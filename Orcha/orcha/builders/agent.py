"""
orcha.builders.agent
=====================
Single-agent execution graph.

Runs one :class:`~orcha.nodes.agent.AgentNode` on the full query with its
tool executor and native function-calling loop, then maps the agent output
into the packet's ``answer`` so downstream consumers (``RunResult``, the
API) behave exactly like every other graph.

This is the graph used for ``default``/``research`` agent runs when a real
tool surface is available (workspace roots + capabilities/tools + an active
local model) — it is what makes those agents *actually execute* tools
end-to-end instead of describing what they would do. When no tool surface or
model is available the server falls back to the classic pipeline graphs, so
plain chat keeps working unchanged.

Topology (with intent gate)::

    intent_gate (entry) ──read──► agent_readonly ──► finalize_agent ──► END
                     ├──tool──► agent ──► finalize_agent ──► END
                     └──chat──► finalize_agent ──► END

The gate decides conversation intent FIRST (one router completion, no tool
schemas, no tool-use directive, but WITH structured attachment/workspace
metadata so "this folder"/"these files" references are grounded). Chat
intent never reaches a tool loop. Read intents (READ_WORKSPACE,
READ_ATTACHMENT, SEARCH, PROJECT_ANALYSIS, UNKNOWN) enter the READ-ONLY
agent whose executor contains only SAFE tools — workspace inspection can
never trigger approvals or writes. Tool intents (FILE_OPERATION,
TOOL_REQUEST) enter the full agent with the approval policy.

Legacy topology (no gate configured)::

    agent (entry) ──► finalize_agent ──► END
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..core.packets import BudgetState, OrchaPacket, PacketKind
from ..graph.edge import END
from ..graph.graph import Graph
from ..graph.node import Node
from ..graph.context import RunContext
from ..graph.runtime import GraphRuntime
from ..nodes.agent import AgentConfig, AgentNode
from ..agent_runtime.conversation import sanitize_final_answer
from ..nodes.intent import (
    INTENT_CHAT,
    INTENT_FILE_OPERATION,
    INTENT_PROJECT_ANALYSIS,
    INTENT_READ_ATTACHMENT,
    INTENT_READ_WORKSPACE,
    INTENT_SEARCH,
    INTENT_TOOL_REQUEST,
    INTENT_UNKNOWN,
    IntentGateConfig,
    IntentGateNode,
)
from ..nodes.planner import (
    ComplexityGateConfig,
    ComplexityGateNode,
    TaskPlannerNode,
    TaskContextBuilderNode,
    ObserverNode,
    ReplannerNode,
    VerifierNode,
    StepExecutorNode,
)
from ..nodes.task_executor import (
    TaskExecutionLoop,
    TaskStepVerifier,
    FocusedTaskContextBuilder,
    TaskExecutionObserver,
    TaskRetryHandler,
    TaskFinalVerifier,
    AdaptiveReplanner,
)

# Read intents enter the read-only agent (SAFE tools only — no approvals).
READ_INTENTS = {
    INTENT_READ_WORKSPACE,
    INTENT_READ_ATTACHMENT,
    INTENT_SEARCH,
    INTENT_PROJECT_ANALYSIS,
    INTENT_UNKNOWN,
}
# Tool intents enter the full agent (write/execute tools + approval policy).
FULL_INTENTS = {INTENT_FILE_OPERATION, INTENT_TOOL_REQUEST}


@dataclass
class AgentGraphConfig:
    """
    Configuration for the single-agent graph.

    Attributes
    ----------
    agent_config            Agent node configuration (system prompt, executor
                            wiring, completion_fn, seed messages, budget) for
                            the FULL tool loop (write/execute tools + policy).
    executor                Optional ToolExecutor shared with the agent loop.
    readonly_agent_config   Optional AgentConfig for the READ-ONLY agent node.
                            Its executor must contain only SAFE (non-mutating)
                            tools so workspace inspection can never write or
                            request approval. When absent, read intents fall
                            back to the full agent.
    gate_config             Optional IntentGateConfig. When set (and backed by
                             a completion_fn), the graph starts with an intent
                             gate that routes chat-only messages straight to
                             finalize, read intents to the read-only agent, and
                             tool intents to the full agent loop.
    complexity_gate_config  Optional ComplexityGateConfig. When set, adds a
                             complexity analysis gate after the intent gate.
                             Tool intents are further classified as simple or
                             complex. Complex requests are routed through a
                             task planner, step executor, observer, and
                             verifier cycle. Simple requests proceed as before.
    capabilities            Declared capability names (informational + surfaced
                            to the builder so it can assemble the executor).
    reasoning_level         Reasoning level (fast/light/medium/high/max). When
                            set, the reasoning configuration drives the
                            agent's tool-call iteration budget.
    max_cost                Budget cap in USD.
    max_latency_s           Maximum wall-clock latency budget.
    max_iterations          Hard cap on pipeline loop count.
    engine                  Execution engine: "langgraph" (LangGraph
                            StateGraph; the default) or "native"
                            (GraphRuntime). Used by ``build_agent_runner``;
                            overridable per call and via the
                            ORCHA_AGENT_ENGINE env var.
    require_approval        LangGraph engine only: insert an approval gate
                            before the final answer so the run pauses with
                            ``ApprovalPending`` until a human approves or
                            rejects (resume with ``resume_value``). The
                            native engine keeps its existing broker-based
                            per-tool approval mechanism instead.
    """

    agent_config: AgentConfig = field(default_factory=AgentConfig)
    executor: Optional[Any] = None  # ToolExecutor
    readonly_agent_config: Optional[AgentConfig] = None
    gate_config: Optional[IntentGateConfig] = None
    complexity_gate_config: Optional[ComplexityGateConfig] = None
    capabilities: List[str] = field(default_factory=list)
    # Absolute directories the workspace tools operate on. Carried here
    # so the per-step prompt can TELL the model which directory it is
    # working in. Nothing populated the old `workspace_state` field, so
    # the model was never told, invented a 'workspace' subfolder that did
    # not exist, and lost every attempt at running the tests to 'The
    # system cannot find the path specified'.
    workspace_roots: List[str] = field(default_factory=list)
    reasoning_level: Optional[str] = None
    max_cost: float = 1.0
    max_latency_s: float = 120.0
    max_iterations: int = 3
    engine: str = "langgraph"
    require_approval: bool = False
    # Turns on the compact planner/executor prompts, tighter context
    # windows, shorter result truncation and the reduced step cap in
    # TaskPlannerNode/TaskExecutionLoop (see small_model_prompts.py).
    # That whole path already existed but nothing ever set this flag, so
    # every local run — including 1.5B/3B models it was written for — used
    # the full-size prompts. Set from the active model's own name by the
    # API layer (see _is_small_model in orcha/api/server.py).
    small_model: bool = False


class _FinalizeAgent(Node):
    """
    Maps the agent (or intent-gate chat) output into the standard response
    payload so the graph terminates with an ``answer`` like every other graph.

    Chat intent: the answer is the router's own response and no agent work
    happened. Any agent intent: the answer is the agent loop's final output.
    """

    name = "finalize_agent"

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        if packet.payload.get("approval_rejected"):
            # The approval gate (LangGraph engine) rejected the proposed
            # action — surface the refusal instead of the agent's output.
            note = packet.payload.get("approval_note") or "no reason given"
            return self._finalize_with_event_stream(
                packet,
                answer=f"The proposed action was not approved. Reason: {note}",
                confidence=0.9,
                quality_score=0.9,
                iterations=packet.payload.get("agent_iterations", 0),
                steps=packet.payload.get("agent_steps", []),
                tool_calls=packet.payload.get("agent_tool_calls", []),
                agent_completed=True,
            )
        if packet.payload.get("intent_gate_intent") == INTENT_CHAT:
            response = sanitize_final_answer(
                packet.payload.get("intent_gate_response")
            )
            if response is None:
                response = (
                    "I wasn't able to come up with a clear answer for that "
                    "just now — try rephrasing the request."
                )
            return self._finalize_with_event_stream(
                packet,
                answer=response,
                confidence=0.9,
                quality_score=0.9,
                iterations=0,
                steps=[],
                tool_calls=[],
                agent_completed=True,
            )
        # New adaptive execution path: surface verified final response when present
        if packet.payload.get("final_response"):
            fr = packet.payload.get("final_response")
            answer = fr.get("answer", "") if isinstance(fr, dict) else ""
            if not answer:
                answer = packet.payload.get("agent_output") or (
                    packet.payload.get("verification_result") or ""
                )
            return self._finalize_with_event_stream(
                packet,
                answer=answer,
                confidence=0.95,
                quality_score=0.95,
                iterations=packet.payload.get("iterations", 0),
                steps=packet.payload.get("agent_steps", []),
                tool_calls=packet.payload.get("agent_tool_calls", []),
                agent_completed=True,
            )
        output = sanitize_final_answer(packet.payload.get("agent_output"))
        steps = packet.payload.get("agent_steps", [])
        tool_calls = packet.payload.get("agent_tool_calls", [])
        iterations = packet.payload.get("agent_iterations", 0)
        completed = packet.payload.get("agent_completed", False)
        if output is None:
            if completed:
                output = (
                    f"The run finished after {iterations} iteration(s) and "
                    f"{len(tool_calls)} tool call(s) without producing a "
                    "final answer."
                )
            else:
                # The decomposition pipeline reports WHY it stopped in
                # `error` (set by TaskExecutionObserver / the execution
                # loop) but sets no `agent_output` — that field belongs to
                # the single-shot agent path. Without this, a precise
                # diagnostic ("execution produced no result for an
                # incomplete plan") was thrown away and replaced with a
                # generic "try rephrasing", which is both useless to the
                # user and actively misleading: rephrasing does not fix a
                # pipeline that never executed its plan.
                pipeline_error = packet.payload.get("error")
                # Before reporting a bare failure, check what the run
                # ACTUALLY did. A partially-completed plan still wrote real
                # files and finished real steps, and all of it lives in
                # execution_plan — which this branch used to discard
                # wholesale. Measured on a live 3B run: two correct files
                # written and three steps completed, reported to the user as
                # "I wasn't able to complete that request". Telling someone
                # nothing happened when files changed on their disk is worse
                # than saying nothing, because they act on it.
                progress = _describe_partial_progress(packet)
                if progress:
                    output = progress
                    if pipeline_error:
                        output = f"{output}\n\nIt stopped because: {pipeline_error}"
                else:
                    output = (
                        f"I couldn't complete that request. {pipeline_error}"
                        if pipeline_error
                        else (
                            "I wasn't able to complete that request — the agent "
                            "stopped before finishing. Try rephrasing it."
                        )
                    )
        return self._finalize_with_event_stream(
            packet,
            answer=output,
            confidence=0.9 if completed else 0.5,
            quality_score=0.9 if completed else 0.5,
            iterations=iterations,
            steps=steps,
            tool_calls=tool_calls,
            agent_completed=completed,
        )

    def _finalize_with_event_stream(
        self,
        packet: OrchaPacket,
        answer: str,
        confidence: float,
        quality_score: float,
        iterations: int,
        steps: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        agent_completed: bool,
    ) -> OrchaPacket:
        """Finalize response while preserving event stream adapter and emitting completion event."""
        from ..graph.context import EVT_EXEC_RUN_COMPLETED
        from ..nodes.task_executor import EventStreamAdapter

        adapter: Optional[EventStreamAdapter] = packet.payload.get("_event_adapter")
        snapshot = None
        if adapter is not None:
            adapter.emit_run_completed(final_answer=answer)
            snapshot = adapter.get_snapshot()

        return packet.fork(
            PacketKind.RESPONSE,
            answer=answer,
            confidence=confidence,
            quality_score=quality_score,
            synthesized=False,
            primary="agent",
            contributors=["agent"],
            agg_mode="agent",
            iterations=iterations,
            agent_steps=steps,
            agent_tool_calls=tool_calls,
            agent_completed=agent_completed,
            _event_adapter=adapter,
            _event_stream_snapshot=snapshot,
        )


def _describe_partial_progress(packet: OrchaPacket) -> str:
    """Report what a stopped decomposition run actually accomplished.

    Returns "" when there is genuinely nothing to report, so the caller can
    fall through to its plain failure message. Anything else here is drawn
    from the plan's own recorded state — completed steps and the file paths
    their tool calls touched — never from model narration.
    """
    plan_raw = packet.payload.get("execution_plan")
    if not plan_raw:
        return ""
    try:
        from ..core.packets import ExecutionPlan
        from ..nodes.task_executor import _files_written_so_far

        plan = ExecutionPlan(**plan_raw) if isinstance(plan_raw, dict) else plan_raw
        done = [s for s in plan.steps if s.status == "completed"]
        files = _files_written_so_far(plan)
    except Exception:
        return ""
    if not done and not files:
        return ""

    lines = [
        f"I got partway through: {len(done)} of {len(plan.steps)} steps "
        "completed, but not the whole task."
    ]
    if files:
        lines.append("\nFiles written:")
        lines.extend(f"  - {p}" for p in files)
    if done:
        lines.append("\nSteps completed:")
        lines.extend(f"  - {s.title}" for s in done)
    remaining = [s for s in plan.steps if s.status not in ("completed", "skipped")]
    if remaining:
        lines.append("\nStill outstanding:")
        lines.extend(f"  - {s.title}" for s in remaining)
    if files:
        lines.append(
            "\nCheck the files above before re-running — they are already on "
            "disk, so a re-run starts from a changed workspace."
        )
    return "\n".join(lines)


def _intent_route(packet: OrchaPacket) -> str:
    """Conditional-edge label after the gate: read intents enter the read-only
    agent, tool intents enter the full agent, and everything else (including a
    missing/garbled gate outcome) goes straight to finalize as a plain
    conversational answer."""
    intent = packet.payload.get("intent_gate_intent") or INTENT_CHAT
    if intent in READ_INTENTS:
        return "agent_readonly"
    if intent in FULL_INTENTS:
        return "agent"
    return "chat"


def _complexity_route(packet: OrchaPacket) -> str:
    """Conditional-edge label after complexity gate: simple requests go to
    the agent, complex requests go to the task planner."""
    gate = packet.payload.get("complexity_gate")
    if gate == "complex":
        return "complex"
    return "simple"


def _observer_route(packet: OrchaPacket) -> str:
    """Conditional-edge label after observer: replan if needed, verify if
    complete, otherwise continue with next step."""
    if packet.payload.get("replan_needed"):
        return "replan"
    if packet.payload.get("objective_met"):
        return "verify"
    if packet.payload.get("task_complete"):
        return "finalize"
    return "next_step"


def _replanner_route(packet: OrchaPacket) -> str:
    """Conditional-edge label after replanner: replan if needed, otherwise
    finalize."""
    if packet.payload.get("replan_needed"):
        return "replan"
    return "finalize"


def _task_context_route(packet: OrchaPacket) -> str:
    """Conditional-edge label after task context builder: replan if the plan
    stalled, execute if there's a step, complete if the plan is done.

    The "replan" label exists because "no runnable step" and "plan finished"
    are NOT the same state, and collapsing them is how a run ends after two
    of four steps while reporting success. The context builder now separates
    them: ``task_blocked`` means work remains that nothing can currently
    reach (a dependent of a skipped step, a cycle, a dependency the
    replanner renamed), which needs the plan graph repaired rather than a
    final answer written.
    """
    if packet.payload.get("task_blocked"):
        return "replan"
    if packet.payload.get("task_complete"):
        return "complete"
    return "execute"


def _execution_observer_route(packet: OrchaPacket) -> str:
    """Conditional-edge label after execution observer: retry, modify, replan, verify,
    next task, or finalize."""
    action = packet.payload.get("task_action", "finalize")
    if action == "retry":
        return "retry"
    if action == "modify":
        return "modify"
    if action == "replan":
        return "replan"
    if action == "verify":
        return "verify"
    if action == "next_task":
        return "next_task"
    return "finalize"


def _retry_handler_route(packet: OrchaPacket) -> str:
    """Conditional-edge label after retry handler: retry same task, next task,
    or finalize."""
    action = packet.payload.get("task_action", "finalize")
    if action == "retry_same":
        return "retry_same"
    if action == "next_task":
        return "next_task"
    return "finalize"


def _adaptive_replanner_route(packet: OrchaPacket) -> str:
    """Conditional-edge label after adaptive replanner: next task or finalize."""
    action = packet.payload.get("task_action", "finalize")
    if action == "next_task":
        return "next_task"
    return "finalize"


def _apply_reasoning(config: "AgentGraphConfig") -> None:
    """Apply reasoning-orchestration knobs when a level is requested: raise
    the agent's tool-call iteration budget to match the level (bounded by the
    level's own cap) so multi-step file/tool tasks have room to finish."""
    if config.reasoning_level:
        from ..capabilities.reasoning import reasoning_config

        rcfg = reasoning_config(config.reasoning_level)
        if config.executor is not None:
            config.agent_config.max_iterations = max(
                config.agent_config.max_iterations, rcfg["max_iterations"]
            )
        config.agent_config.reasoning_level = config.reasoning_level


def _build_agent_nodes(config: "AgentGraphConfig") -> Dict[str, Any]:
    """
    Instantiate the agent-graph nodes shared by both engines.

    Returns a dict with ``agent``, ``finalize``, and — when an intent gate
    is configured — ``gate`` and ``readonly`` (the read-only agent). The
    native builder and the LangGraph builder wire these same instances
    into their own topologies.
    """
    _apply_reasoning(config)

    agent = AgentNode(
        name="agent",
        config=config.agent_config,
        timeout_s=config.agent_config.max_iterations * 60.0,
    )
    finalize = _FinalizeAgent()

    gate_cfg = config.gate_config
    has_gate = gate_cfg is not None and gate_cfg.completion_fn is not None
    gate: Optional[IntentGateNode] = None
    readonly: Optional[AgentNode] = None
    if has_gate:
        # 300s: the gate may make up to two router completions plus a
        # conversational re-ask, each against a large prompt (tool schemas,
        # attachment context). On local 7B-class GPUs a single cold call can
        # take 30-90s — the old 60s cap killed healthy runs mid-route.
        gate = IntentGateNode(name="intent_gate", config=gate_cfg, timeout_s=300.0)
        if config.readonly_agent_config is not None:
            readonly = AgentNode(
                name="agent_readonly",
                config=config.readonly_agent_config,
                timeout_s=config.readonly_agent_config.max_iterations * 60.0,
            )
    # Complexity gate and task orchestration nodes
    complexity_gate: Optional[ComplexityGateNode] = None
    task_planner: Optional[TaskPlannerNode] = None
    task_context_builder: Optional[TaskContextBuilderNode] = None
    step_executor: Optional[StepExecutorNode] = None
    observer: Optional[ObserverNode] = None
    replanner: Optional[ReplannerNode] = None
    verifier: Optional[VerifierNode] = None
    # New adaptive execution loop components
    execution_loop: Optional[TaskExecutionLoop] = None
    focused_context_builder: Optional[FocusedTaskContextBuilder] = None
    execution_observer: Optional[TaskExecutionObserver] = None
    retry_handler: Optional[TaskRetryHandler] = None
    step_verifier: Optional[TaskStepVerifier] = None
    final_verifier: Optional[TaskFinalVerifier] = None
    adaptive_replanner: Optional[AdaptiveReplanner] = None
    if config.complexity_gate_config is not None:
        complexity_gate = ComplexityGateNode(
            name="complexity_gate",
            config=config.complexity_gate_config,
            timeout_s=120.0,
        )
        task_planner = TaskPlannerNode(
            name="task_planner",
            completion_fn=config.complexity_gate_config.completion_fn,
            # 180s is enough for a fast/cloud model but not for a slow local
            # 7B model producing a full multi-task decomposition plan for a
            # large, detailed spec (confirmed empirically: a real build
            # request timed out here at ~10 tok/s well before the plan
            # finished generating). Generous enough to let a big plan finish
            # on modest local hardware; the node still fails cleanly on a
            # genuine hang.
            timeout_s=600.0,
            # So the planner grounds each step's likely_tools in the real
            # registered tool set instead of inventing plausible-sounding
            # names (e.g. "write_code") that the executing model then can't
            # actually call — see the comment in TaskPlannerNode.run().
            # No small_model here on purpose: TaskPlannerNode has no
            # compact-prompt path. The PLANNER_SMALL prompt belongs to
            # OrchaTaskPlanner (orcha/nodes/planner.py), a separate class
            # this graph does not use — passing small_model here raises
            # TypeError at graph-build time.
            executor=config.executor,
        )
        task_context_builder = TaskContextBuilderNode(
            name="task_context_builder",
            timeout_s=30.0,
        )
        step_executor = StepExecutorNode(
            name="step_executor",
            completion_fn=config.complexity_gate_config.completion_fn,
            executor=config.executor,
            timeout_s=300.0,
        )
        observer = ObserverNode(
            name="observer",
            completion_fn=config.complexity_gate_config.completion_fn,
            timeout_s=120.0,
        )
        replanner = ReplannerNode(
            name="replanner",
            completion_fn=config.complexity_gate_config.completion_fn,
            timeout_s=120.0,
        )
        verifier = VerifierNode(
            name="verifier",
            completion_fn=config.complexity_gate_config.completion_fn,
            timeout_s=120.0,
        )
        # New adaptive execution loop components. NOTE: each `name=` here
        # MUST match the string the graph edges below use to reference this
        # node (Graph looks nodes up by .name, not by the Python variable
        # name) — a prior "task_"-prefix rename of these left the edges
        # pointing at names no node was ever registered under, so the graph
        # failed validation the moment complexity_gate_config was ever
        # actually supplied (it never had been, until now).
        execution_loop = TaskExecutionLoop(
            name="execution_loop",
            workspace_roots=list(config.workspace_roots or []),
            completion_fn=config.complexity_gate_config.completion_fn,
            executor=config.executor,
            timeout_s=600.0,
            max_task_iterations=5,
            small_model=config.small_model,
        )
        focused_context_builder = FocusedTaskContextBuilder(
            name="focused_task_context_builder",
            timeout_s=30.0,
        )
        execution_observer = TaskExecutionObserver(
            name="execution_observer",
            completion_fn=config.complexity_gate_config.completion_fn,
            timeout_s=120.0,
        )
        retry_handler = TaskRetryHandler(
            name="retry_handler",
            completion_fn=config.complexity_gate_config.completion_fn,
            timeout_s=120.0,
        )
        step_verifier = TaskStepVerifier(
            name="step_verifier",
            completion_fn=config.complexity_gate_config.completion_fn,
            workspace_roots=list(config.workspace_roots or []),
            timeout_s=120.0,
        )
        final_verifier = TaskFinalVerifier(
            name="final_verifier",
            completion_fn=config.complexity_gate_config.completion_fn,
            timeout_s=120.0,
        )
        adaptive_replanner = AdaptiveReplanner(
            name="adaptive_replanner",
            completion_fn=config.complexity_gate_config.completion_fn,
            timeout_s=120.0,
        )
    return {
        "agent": agent,
        "finalize": finalize,
        "gate": gate,
        "readonly": readonly,
        "complexity_gate": complexity_gate,
        "task_planner": task_planner,
        "task_context_builder": task_context_builder,
        "step_executor": step_executor,
        "observer": observer,
        "replanner": replanner,
        "verifier": verifier,
        "execution_loop": execution_loop,
        "focused_context_builder": focused_context_builder,
        "execution_observer": execution_observer,
        "retry_handler": retry_handler,
        "step_verifier": step_verifier,
        "final_verifier": final_verifier,
        "adaptive_replanner": adaptive_replanner,
    }


def build_agent_graph(
    config: Optional[AgentGraphConfig] = None,
    *,
    agent_config: Optional[AgentConfig] = None,
    executor: Optional[Any] = None,  # ToolExecutor
    readonly_agent_config: Optional[AgentConfig] = None,
    gate_config: Optional[IntentGateConfig] = None,
    capabilities: Optional[List[str]] = None,
    reasoning_level: Optional[str] = None,
    max_cost: float = 1.0,
    max_latency_s: float = 120.0,
    max_iterations: int = 3,
) -> Graph:
    """
    Build the single-agent execution graph.

    Accepts either an ``AgentGraphConfig`` or individual keyword arguments.
    Returns a validated ``Graph`` ready for ``GraphRuntime``.
    """
    if config is None:
        agent_cfg = agent_config or AgentConfig()
        config = AgentGraphConfig(
            agent_config=agent_cfg,
            executor=executor,
            readonly_agent_config=readonly_agent_config,
            gate_config=gate_config,
            capabilities=list(capabilities or []),
            reasoning_level=reasoning_level,
            max_cost=max_cost,
            max_latency_s=max_latency_s,
            max_iterations=max_iterations,
        )

    nodes = _build_agent_nodes(config)
    agent = nodes["agent"]
    finalize = nodes["finalize"]

    # ── Build topology ──────────────────────────────────────────────────
    g = Graph(name="orcha_agent")
    g.add_node(finalize)

    gate = nodes["gate"]
    complexity_gate = nodes["complexity_gate"]
    if gate is not None:
        # Intent first: the gate answers chat-only messages directly, routes
        # read intents into the READ-ONLY agent (SAFE tools only — workspace
        # inspection can never write or request approval), and routes tool
        # intents into the full agent loop with the approval policy. Without
        # this, tool schemas + the tool directive reach the model on every
        # message and pure conversation can trigger spurious tool calls (and,
        # in approval mode, spurious approval requests).
        g.add_node(gate, entry=True)
        g.add_node(agent)
        g.add_edge("agent", "finalize_agent")

        readonly = nodes["readonly"]
        has_readonly = readonly is not None
        if has_readonly:
            g.add_node(readonly)
            g.add_edge("agent_readonly", "finalize_agent")

        # Complexity gate: after intent gate routes to agent, optionally
        # add complexity analysis for tool intents.
        if complexity_gate is not None:
            g.add_node(complexity_gate)
            g.add_node(nodes["task_planner"])
            g.add_node(nodes["task_context_builder"])
            g.add_node(nodes["step_executor"])
            g.add_node(nodes["observer"])
            g.add_node(nodes["replanner"])
            g.add_node(nodes["verifier"])
            # New adaptive execution loop nodes
            execution_loop = nodes["execution_loop"]
            focused_context_builder = nodes["focused_context_builder"]
            execution_observer = nodes["execution_observer"]
            retry_handler = nodes["retry_handler"]
            step_verifier = nodes["step_verifier"]
            final_verifier = nodes["final_verifier"]
            adaptive_replanner = nodes["adaptive_replanner"]
            if execution_loop is not None:
                g.add_node(execution_loop)
            if focused_context_builder is not None:
                g.add_node(focused_context_builder)
            if execution_observer is not None:
                g.add_node(execution_observer)
            if retry_handler is not None:
                g.add_node(retry_handler)
            if step_verifier is not None:
                g.add_node(step_verifier)
            if final_verifier is not None:
                g.add_node(final_verifier)
            if adaptive_replanner is not None:
                g.add_node(adaptive_replanner)

            # Wire the adaptive execution loop
            if execution_loop is not None and execution_observer is not None:
                # Use new adaptive execution loop (skip step_executor - no double execution)
                g.add_edge("task_context_builder", "execution_loop")
                # EXECUTE -> VERIFY -> OBSERVE. The verifier checks the
                # workspace and hands the observer evidence instead of
                # the step's own account of itself; the observer then
                # overrides a self-report that the evidence contradicts.
                if step_verifier is not None:
                    g.add_edge("execution_loop", "step_verifier")
                    g.add_edge("step_verifier", "execution_observer")
                else:
                    g.add_edge("execution_loop", "execution_observer")
                g.add_conditional(
                    "execution_observer",
                    {
                        "retry": "retry_handler",
                        "modify": "retry_handler",
                        "replan": "adaptive_replanner" if adaptive_replanner is not None else "replanner",
                        "verify": "final_verifier" if final_verifier is not None else "verifier",
                        "next_task": "task_context_builder",
                        "finalize": "finalize_agent",
                    },
                    predicate=_execution_observer_route,
                )
                if retry_handler is not None:
                    g.add_conditional(
                        "retry_handler",
                        {
                            "retry_same": "execution_loop",
                            "next_task": "task_context_builder",
                            "finalize": "finalize_agent",
                        },
                        predicate=_retry_handler_route,
                    )
                if adaptive_replanner is not None:
                    g.add_conditional(
                        "adaptive_replanner",
                        {
                            "next_task": "task_context_builder",
                            "finalize": "finalize_agent",
                        },
                        predicate=_adaptive_replanner_route,
                    )
            else:
                # Fallback to old observer pattern
                g.add_edge("step_executor", "observer")
                g.add_conditional(
                    "observer",
                    {
                        "replan": "replanner",
                        "verify": "verifier",
                        "next_step": "task_context_builder",
                        "finalize": "finalize_agent",
                    },
                    predicate=_observer_route,
                )

            g.add_conditional(
                "replanner",
                {
                    "replan": "task_planner",
                    "finalize": "finalize_agent",
                },
                predicate=_replanner_route,
            )
            if final_verifier is not None:
                g.add_edge("final_verifier", "finalize_agent")
            else:
                g.add_edge("verifier", "finalize_agent")
            g.add_edge("task_planner", "task_context_builder")
            # Route task_context_builder based on whether adaptive loop is active
            if execution_loop is not None and execution_observer is not None:
                # Adaptive loop: task_context_builder -> execution_loop.
                # _task_context_route returns "execute" whenever there's a
                # next task to run — that label must be routed here too (the
                # plain add_edge to "execution_loop" registered above is not
                # consulted once a conditional table exists for this same
                # source node), or every non-"complete" step blows up with
                # NoRoute the first time this path is actually exercised.
                g.add_conditional(
                    "task_context_builder",
                    {
                        "execute": "execution_loop",
                        "replan": (
                            "adaptive_replanner"
                            if adaptive_replanner is not None
                            else "replanner"
                        ),
                        "complete": "finalize_agent",
                    },
                    predicate=_task_context_route,
                )
            else:
                # Fallback: task_context_builder -> step_executor
                g.add_conditional(
                    "task_context_builder",
                    {
                        "execute": "step_executor",
                        "replan": "replanner",
                        "complete": "finalize_agent",
                    },
                    predicate=_task_context_route,
                )
            # Intent gate routes tool intents to complexity gate
            read_dest = "agent_readonly" if has_readonly else "agent"
            g.add_conditional(
                "intent_gate",
                {
                    "chat": "finalize_agent",
                    "agent_readonly": read_dest,
                    "agent": "complexity_gate",
                },
                predicate=_intent_route,
            )
            # Complexity gate routes simple to agent, complex to task_planner
            g.add_conditional(
                "complexity_gate",
                {
                    "simple": "agent",
                    "complex": "task_planner",
                },
                predicate=_complexity_route,
            )
        else:
            # No complexity gate: standard routing
            read_dest = "agent_readonly" if has_readonly else "agent"
            g.add_conditional(
                "intent_gate",
                {
                    "chat": "finalize_agent",
                    "agent_readonly": read_dest,
                    "agent": "agent",
                },
                predicate=_intent_route,
            )
    else:
        # No gate configured (legacy direct construction): agent runs on
        # every message, exactly as before.
        g.add_node(agent, entry=True)
        g.add_edge("agent", "finalize_agent")

    g.add_edge("finalize_agent", END)
    g.validate()

    return g


class _NativeAgentRunner(GraphRuntime):
    """
    Native-engine agent runner produced by ``build_agent_runner``.

    A ``GraphRuntime`` subclass that applies the ``AgentGraphConfig``
    budget knobs to the fresh packet when the caller does not supply
    their own packet/budget (same treatment as the default graph).
    ``require_approval`` is a LangGraph-engine feature; on the native
    engine approvals stay broker-based inside the agent loop.
    """

    def __init__(
        self,
        graph: Graph,
        *,
        config: "AgentGraphConfig",
        store: Any = None,
        checkpoint_every: Optional[int] = None,
        max_steps: int = 1000,
    ) -> None:
        super().__init__(
            graph, store=store, checkpoint_every=checkpoint_every,
            max_steps=max_steps,
        )
        self._config = config

    async def run(
        self, query: str, *,
        budget: Optional["BudgetState"] = None,
        packet: Optional["OrchaPacket"] = None,
        resume_from: Optional[Any] = None,
        run_id: Optional[str] = None,
        on_event: Optional[Any] = None,
        cancel: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if packet is None and budget is None:
            from .default import _config_budget, _initial_packet

            run_id = run_id or str(uuid.uuid4())
            packet = _initial_packet(
                graph_name=self.graph.name,
                run_id=run_id,
                query=query,
                budget=_config_budget(self._config),
                run_all_experts=False,
                metadata=metadata,
            )
        return await super().run(
            query, budget=budget, packet=packet, resume_from=resume_from,
            run_id=run_id, on_event=on_event, cancel=cancel, metadata=metadata,
        )


def build_agent_runner(
    config: Optional["AgentGraphConfig"] = None,
    *,
    engine: Optional[str] = None,
    store: Any = None,
    checkpoint_every: Optional[int] = None,
    max_steps: int = 1000,
    checkpointer: Any = None,
    agent_config: Optional[AgentConfig] = None,
    executor: Optional[Any] = None,  # ToolExecutor
    readonly_agent_config: Optional[AgentConfig] = None,
    gate_config: Optional[IntentGateConfig] = None,
    complexity_gate_config: Optional[ComplexityGateConfig] = None,
    capabilities: Optional[List[str]] = None,
    reasoning_level: Optional[str] = None,
    max_cost: float = 1.0,
    max_latency_s: float = 120.0,
    max_iterations: int = 3,
    require_approval: bool = False,
) -> Any:
    """
    Build a single-agent graph runner on the selected execution engine.

    ``engine`` selects the runtime: ``"langgraph"`` (default) returns an
    ``AgentLangGraphRunner`` driving a genuine LangGraph
    ``StateGraph`` wired with the same agent nodes (and, when
    ``require_approval=True``, an interrupt-based approval gate);
    ``"native"`` returns a ``GraphRuntime`` driving the validated
    ``Graph``. The engine may also be set via the ``ORCHA_AGENT_ENGINE``
    environment variable. Both runners expose the same async surface
    (``run`` / ``replay`` / ``live_emitter``) and return identical
    ``RunResult`` objects.
    """
    import os

    if config is None:
        agent_cfg = agent_config or AgentConfig()
        config = AgentGraphConfig(
            agent_config=agent_cfg,
            executor=executor,
            readonly_agent_config=readonly_agent_config,
            gate_config=gate_config,
            complexity_gate_config=complexity_gate_config,
            capabilities=list(capabilities or []),
            reasoning_level=reasoning_level,
            max_cost=max_cost,
            max_latency_s=max_latency_s,
            max_iterations=max_iterations,
            require_approval=require_approval,
        )
    engine = engine or os.environ.get("ORCHA_AGENT_ENGINE") or config.engine

    if engine == "langgraph":
        from ..integrations.langgraph import LangGraphEngine
        from .langgraph_agent import AgentLangGraphRunner, build_agent_graph_langgraph

        graph = build_agent_graph_langgraph(
            config, max_steps=max_steps, checkpointer=checkpointer,
        )
        return AgentLangGraphRunner(
            LangGraphEngine(graph), graph_name="orcha_agent",
            max_steps=max_steps, config=config,
        )

    if engine != "native":
        raise ValueError(
            f"Unknown engine {engine!r} — expected 'native' or 'langgraph'"
        )

    graph = build_agent_graph(config)
    return _NativeAgentRunner(
        graph, config=config, store=store,
        checkpoint_every=checkpoint_every, max_steps=max_steps,
    )


__all__ = ["build_agent_graph", "build_agent_runner", "AgentGraphConfig", "ComplexityGateConfig", "AdaptiveReplanner"]
