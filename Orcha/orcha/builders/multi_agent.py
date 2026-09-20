"""
orcha.builders.multi_agent
===========================
Multi-agent collaboration graph.

This graph decomposes a query into subtasks, assigns each subtask to an
agent node, gathers the agent results, and synthesizes a final answer.
It is designed for complex tasks that benefit from autonomous sub-agent
reasoning.

Topology::

    decompose ──► scatter_agents ──► [agent_0, agent_1, ..., agent_N]
                                              │
                                              ▼ (gather)
                                        synthesize ──► verify ──► END

The scatter node fans out one branch per subtask, each branch runs an
agent on that subtask, and the gather node collects all agent outputs for
synthesis.

Usage::

    from orcha.builders import build_multi_agent_graph
    from orcha.nodes.agent import AgentConfig
    from orcha.graph import GraphRuntime

    graph = build_multi_agent_graph(
        agent_config=AgentConfig(max_iterations=5),
        verify=True,
    )
    rt = GraphRuntime(graph)
    result = await rt.run("Explain quantum computing in simple terms")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..core.packets import OrchaPacket, PacketKind
from ..graph.edge import END
from ..graph.graph import Graph
from ..graph.node import GatherNode, Node, to_node
from ..graph.context import RunContext, ScatterResult
from ..nodes.agent import AgentConfig, AgentNode
from ..nodes.stages import DecomposeNode
from ..nodes.verify import VerifyNode
from ..orchestration.decomposer import get_decomposer


@dataclass
class MultiAgentGraphConfig:
    """
    Configuration for the multi-agent graph.

    Attributes
    ----------
    agent_config     Configuration for agent nodes (shared across branches).
    verify           Whether to include a verification step after synthesis.
    max_cost         Budget cap in USD.
    max_latency_s    Maximum wall-clock latency budget.
    max_iterations   Hard cap on pipeline loop count.
    use_embeddings   Use sentence-transformers for decomposition.
    max_branches     Maximum number of parallel agent branches.
    model_fn         The model function for agents: async (prompt, system) -> str.
    completion_fn    Optional native function-calling completion: async
                     (messages, system, tools) -> dict (assistant message with
                     content + tool_calls). When set, agents use real tool calls
                     instead of the text protocol.
    executor         Optional ToolExecutor shared by every agent branch. When
                     set, the agent validates/executes tool calls through it.
    capabilities     Declared capability names (assembled into the executor by
                     the server; informational here).
    reasoning_level  Reasoning level (fast/light/medium/high/max). When set, the
                     reasoning configuration drives iteration budget and verify.
    """
    agent_config: AgentConfig = field(default_factory=AgentConfig)
    verify: bool = True
    max_cost: float = 1.0
    max_latency_s: float = 120.0
    max_iterations: int = 3
    use_embeddings: bool = False
    max_branches: int = 5
    model_fn: Optional[Callable] = None
    completion_fn: Optional[Callable] = None
    executor: Optional[Any] = None  # ToolExecutor
    capabilities: List[str] = field(default_factory=list)
    reasoning_level: Optional[str] = None


class _ScatterSubtasks(Node):
    """
    Scatter node: decomposes the query into subtasks and fans out one
    agent branch per subtask.
    """

    name = "scatter_agents"
    is_scatter = True

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> ScatterResult:
        subtasks = packet.get_subtasks()
        if not subtasks:
            # No subtasks: create a single branch with the original query.
            subtasks = [
                OrchaPacket.SubTask(
                    id="t0",
                    description=packet.query,
                    domain="general",
                    priority=1.0,
                )
            ]

        # Limit branches.
        limited = subtasks[: self.max_branches]

        branches = []
        for i, task in enumerate(limited):
            child = packet.fork(
                PacketKind.SUBTASKS,
                task=task.description,
                task_id=task.id,
                task_domain=task.domain,
                task_priority=task.priority,
            )
            branches.append((f"branch_{i}", child, "agent"))

        return ScatterResult(branches=branches, gather_to="gather_agents")

    def __init__(self, max_branches: int = 5) -> None:
        self.max_branches = max_branches


class _AgentWorker(Node):
    """
    Per-branch agent worker. Reads the task from the packet and runs
    the agent on it.
    """

    name = "agent"

    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        self.timeout_s = config.max_iterations * 30  # rough budget

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        task: str = packet.payload.get("task", packet.query)
        context: str = ""

        # Add any retrieval context if present.
        retrieval = packet.payload.get("retrieval_results", [])
        if retrieval:
            context = "\n\nRelevant context:\n" + "\n".join(
                f"- {r[:200]}" for r in retrieval[:3]
            )

        agent = AgentNode(
            name="agent_worker",
            config=AgentConfig(
                system_prompt=self._config.system_prompt,
                max_iterations=self._config.max_iterations,
                model_fn=self._config.model_fn,
                completion_fn=self._config.completion_fn,
                stop_phrases=self._config.stop_phrases,
                scratchpad_max_len=self._config.scratchpad_max_len,
                seed_messages=list(self._config.seed_messages),
                tools=self._config.tools,
                executor=self._config.executor,
                capabilities=list(self._config.capabilities),
                reasoning_level=self._config.reasoning_level,
                approval_broker=self._config.approval_broker,
                approval_timeout_s=self._config.approval_timeout_s,
            ),
            timeout_s=self.timeout_s,
        )

        result = await agent.run(
            packet.fork(packet.kind, agent_context=context, task=task),
            ctx,
        )
        return result


class _GatherAgents(GatherNode):
    """
    Gather node: collects all agent branch outputs and prepares for synthesis.
    """

    name = "gather_agents"

    async def gather(
        self, packet: OrchaPacket, children: List[OrchaPacket], ctx: RunContext,
    ) -> OrchaPacket:
        agent_outputs = []
        all_steps: List[Dict[str, Any]] = []
        all_tool_calls: List[Dict[str, Any]] = []
        for i, child in enumerate(children):
            output = child.payload.get("agent_output", "")
            completed = child.payload.get("agent_completed", False)
            iterations = child.payload.get("agent_iterations", 0)
            task_desc = child.payload.get("task", "")
            error = child.payload.get("error")
            steps = child.payload.get("agent_steps", [])
            tcalls = child.payload.get("agent_tool_calls", [])
            for s in steps:
                s["branch"] = i
            all_steps.extend(steps)
            all_tool_calls.extend(tcalls)
            agent_outputs.append({
                "branch": i,
                "task": task_desc[:100],
                "output": output,
                "completed": completed,
                "failed": bool(error),
                "error": error,
                "iterations": iterations,
            })

        # Combine all agent outputs into a single text for synthesis.
        combined = "\n\n".join(
            f"--- Branch {a['branch']}: {a['task'][:60]} ---\n{a['output']}"
            for a in agent_outputs
        )

        return packet.fork(
            packet.kind,
            agent_outputs=agent_outputs,
            synthesis_input=combined,
            branch_count=len(agent_outputs),
            branches_completed=sum(1 for a in agent_outputs if a["completed"]),
            agent_steps=all_steps,
            agent_tool_calls=all_tool_calls,
            agent_completed=all(a["completed"] for a in agent_outputs),
        )


class _Synthesize(Node):
    """
    Synthesis node: combines agent outputs into a final answer.

    Uses a model call if available, otherwise produces a concatenated
    summary.
    """

    name = "synthesize"

    def __init__(self, model_fn: Optional[Callable] = None) -> None:
        self._model_fn = model_fn

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        synthesis_input: str = packet.payload.get("synthesis_input", "")
        agent_outputs: List[dict] = packet.payload.get("agent_outputs", [])

        if self._model_fn is not None:
            prompt = (
                "Synthesize the following agent outputs into a single, "
                "coherent, comprehensive answer.\n\n"
                + synthesis_input
                + "\n\nProvide the final answer prefixed with 'ANSWER:'"
            )
            try:
                response = await self._model_fn(prompt, "You are a synthesis agent.")
                answer = response
                if "ANSWER:" in answer:
                    answer = answer.split("ANSWER:", 1)[-1].strip()
                synthesized = True
            except Exception as exc:
                answer = f"[Synthesis error: {exc}]"
                synthesized = False
        else:
            # Dry run: concatenate agent outputs.
            parts = []
            for a in agent_outputs:
                if a.get("output"):
                    parts.append(a["output"][:500])
            answer = "\n\n".join(parts) if parts else "[No agent outputs to synthesize]"
            synthesized = False

        return packet.fork(
            packet.kind,
            answer=answer,
            confidence=0.7 if synthesized else 0.4,
            synthesized=synthesized,
            contributors=[f"branch_{a['branch']}" for a in agent_outputs],
            agg_mode="agent_synthesis",
            agent_steps=packet.payload.get("agent_steps", []),
            agent_tool_calls=packet.payload.get("agent_tool_calls", []),
            agent_completed=packet.payload.get("agent_completed", False),
        )


class _VerifyRetryGate(Node):
    """
    Routes a failed verification through one re-synthesis before giving up.

    Verification verdicts are heuristic, so a single retry with a fresh
    model call is worthwhile; an unconditional retry loop would risk an
    infinite cycle, so the retry budget is exactly one.
    """

    name = "verify_retry_gate"

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        verified = packet.payload.get("verified", False)
        retries = int(packet.payload.get("verify_retries", 0))
        should_retry = (not verified) and retries < 1
        return packet.fork(
            packet.kind,
            verify_retries=retries + 1 if should_retry else retries,
            retry_verify=should_retry,
        )


def _apply_reasoning_knobs(config: "MultiAgentGraphConfig") -> None:
    """
    Apply reasoning orchestration knobs in place when a level is requested.

    Shared by ``build_multi_agent_graph`` (native) and the LangGraph
    port so both engines derive identical budgets and verify behavior
    from the configured reasoning level.
    """
    if config.reasoning_level:
        from ..capabilities.reasoning import reasoning_config

        rcfg = reasoning_config(config.reasoning_level)
        if config.executor is not None and config.agent_config.max_iterations <= 1:
            config.agent_config.max_iterations = rcfg["max_iterations"]
        config.verify = bool(config.verify or rcfg["verify"])
        config.agent_config.reasoning_level = config.reasoning_level


def build_multi_agent_graph(
    config: Optional[MultiAgentGraphConfig] = None,
    *,
    agent_config: Optional[AgentConfig] = None,
    verify: bool = True,
    max_cost: float = 1.0,
    max_latency_s: float = 120.0,
    max_iterations: int = 3,
    use_embeddings: bool = False,
    max_branches: int = 5,
    model_fn: Optional[Callable] = None,
    completion_fn: Optional[Callable] = None,
    executor: Optional[Any] = None,  # ToolExecutor
    capabilities: Optional[List[str]] = None,
    reasoning_level: Optional[str] = None,
) -> Graph:
    """
    Build the multi-agent collaboration graph.

    Accepts either a ``MultiAgentGraphConfig`` or individual keyword arguments.
    Returns a validated ``Graph`` ready for ``GraphRuntime``.
    """
    if config is None:
        agent_cfg = agent_config or AgentConfig()
        agent_cfg.model_fn = model_fn or agent_cfg.model_fn
        agent_cfg.completion_fn = completion_fn or agent_cfg.completion_fn
        config = MultiAgentGraphConfig(
            agent_config=agent_cfg,
            verify=verify,
            max_cost=max_cost,
            max_latency_s=max_latency_s,
            max_iterations=max_iterations,
            use_embeddings=use_embeddings,
            max_branches=max_branches,
            model_fn=model_fn,
            completion_fn=completion_fn,
            executor=executor,
            capabilities=list(capabilities or []),
            reasoning_level=reasoning_level,
        )

    # Apply reasoning orchestration knobs when a level is requested.
    _apply_reasoning_knobs(config)

    # ── Instantiate nodes ───────────────────────────────────────────────
    decomposer = get_decomposer(use_embeddings=config.use_embeddings)
    decompose_node = DecomposeNode(decomposer)
    scatter_node = _ScatterSubtasks(max_branches=config.max_branches)
    agent_worker = _AgentWorker(config.agent_config)
    gather_node = _GatherAgents()
    synthesize_node = _Synthesize(model_fn=config.model_fn)

    # ── Build topology ──────────────────────────────────────────────────
    g = Graph(name="orcha_multi_agent")

    g.add_node(decompose_node, entry=True)
    g.add_node(scatter_node)
    g.add_node(agent_worker)
    g.add_node(gather_node)
    g.add_node(synthesize_node)

    # decompose → scatter → [agent workers] → gather → synthesize
    g.add_edge("decompose", "scatter_agents")
    g.fan_out("scatter_agents", "gather_agents")
    g.add_edge("gather_agents", "synthesize")

    if config.verify:
        verify_node = VerifyNode(name="verify", timeout_s=10.0)
        gate = _VerifyRetryGate()
        g.add_node(verify_node)
        g.add_node(gate)
        g.add_edge("synthesize", "verify")
        g.add_edge("verify", "verify_retry_gate")
        # One bounded retry through synthesis when verification fails, then
        # the run ends either way (the retry counter lives in the payload).
        g.add_conditional(
            "verify_retry_gate",
            routes={"retry": "synthesize", "pass": END},
            predicate=lambda p: "retry" if p.payload.get("retry_verify") else "pass",
        )
    else:
        g.add_edge("synthesize", END)

    g.validate()

    return g


def build_multi_agent_runner(
    config: Optional["MultiAgentGraphConfig"] = None,
    *,
    engine: Optional[str] = None,
    store: Any = None,
    checkpoint_every: Optional[int] = None,
    max_steps: int = 1000,
    checkpointer: Any = None,
    agent_config: Optional[AgentConfig] = None,
    verify: bool = True,
    max_cost: float = 1.0,
    max_latency_s: float = 120.0,
    max_iterations: int = 3,
    use_embeddings: bool = False,
    max_branches: int = 5,
    model_fn: Optional[Callable] = None,
    completion_fn: Optional[Callable] = None,
    executor: Optional[Any] = None,  # ToolExecutor
    capabilities: Optional[List[str]] = None,
    reasoning_level: Optional[str] = None,
) -> Any:
    """
    Build a multi-agent graph runner on the selected execution engine.

    ``engine`` selects the runtime: ``"langgraph"`` (default) returns a
    ``MultiAgentLangGraphRunner`` driving a genuine LangGraph
    ``StateGraph`` wired with the same nodes (decompose / scatter /
    per-subtask agents / gather / synthesize / verify); ``"native"``
    returns a ``GraphRuntime`` driving the validated ``Graph``. The
    engine may also be set via the ``ORCHA_AGENT_ENGINE`` environment
    variable. Both runners expose the same async surface (``run`` /
    ``replay`` / ``live_emitter``) and return identical ``RunResult``
    objects.
    """
    import os

    if config is None:
        config = MultiAgentGraphConfig(
            agent_config=agent_config or AgentConfig(),
            verify=verify,
            max_cost=max_cost,
            max_latency_s=max_latency_s,
            max_iterations=max_iterations,
            use_embeddings=use_embeddings,
            max_branches=max_branches,
            model_fn=model_fn,
            completion_fn=completion_fn,
            executor=executor,
            capabilities=list(capabilities or []),
            reasoning_level=reasoning_level,
        )
    engine = engine or os.environ.get("ORCHA_AGENT_ENGINE") or "langgraph"

    if engine == "langgraph":
        from ..integrations.langgraph import LangGraphEngine
        from .langgraph_multi_agent import (
            MultiAgentLangGraphRunner, build_multi_agent_graph_langgraph,
        )

        graph = build_multi_agent_graph_langgraph(
            config, max_steps=max_steps, checkpointer=checkpointer,
        )
        return MultiAgentLangGraphRunner(
            LangGraphEngine(graph), graph_name="orcha_multi_agent",
            max_steps=max_steps, config=config,
        )

    if engine != "native":
        raise ValueError(
            f"Unknown engine {engine!r} — expected 'native' or 'langgraph'"
        )

    from ..graph.runtime import GraphRuntime

    graph = build_multi_agent_graph(config)
    return GraphRuntime(
        graph, store=store,
        checkpoint_every=checkpoint_every, max_steps=max_steps,
    )


__all__ = ["build_multi_agent_graph", "build_multi_agent_runner", "MultiAgentGraphConfig"]
