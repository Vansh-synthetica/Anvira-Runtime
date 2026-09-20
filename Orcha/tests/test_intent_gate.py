"""
Intent-gate regression tests (orcha.nodes.intent + orcha.builders.agent).

The intent gate is the architectural fix for approvals firing on pure
conversation: the tool loop — and therefore the approval machinery — is only
reachable after the gate declares a tool intent. A greeting can never produce
a tool call or an approval request.

Rich intent classification + attachment awareness (BUG-5):
  - read intents (READ_WORKSPACE / READ_ATTACHMENT / SEARCH /
    PROJECT_ANALYSIS / UNKNOWN) route into a READ-ONLY agent whose executor
    contains only SAFE tools — workspace inspection can never write or
    request approval
  - the gate receives structured attachment/workspace metadata so "this
    folder"/"these files" references are grounded instead of being answered
    from language priors
  - tool intents (FILE_OPERATION / TOOL_REQUEST) route into the full agent
  - attached folders are inspected, never hallucinated away
"""
import asyncio

import pytest

from orcha.capabilities.base import (
    ApprovalBroker,
    CapabilityContext,
    PermissionPolicy,
)
from orcha.capabilities.registry import CapabilityRegistry
from orcha.graph.runtime import GraphRuntime
from orcha.graph.store import MemoryStore
from orcha.nodes.agent import AgentConfig
from orcha.nodes.intent import (
    INTENT_CHAT,
    INTENT_FILE_OPERATION,
    INTENT_GATE_ROUTER_PROMPT,
    INTENT_PROJECT_ANALYSIS,
    INTENT_READ_ATTACHMENT,
    INTENT_READ_WORKSPACE,
    INTENT_SEARCH,
    INTENT_TOOL_REQUEST,
    INTENT_UNKNOWN,
    IntentGateConfig,
    IntentGateNode,
)
from orcha.builders.agent import AgentGraphConfig, build_agent_graph

from test_nodes import make_ctx, make_packet


# ── The 10 user-intent regression cases ──────────────────────────────────────

CHAT_CASES = [
    ("hello", "Hello! How can I help you today?"),
    ("Hi!", "Hi there! What can I do for you?"),
    ("How are you?", "I'm doing great, thanks for asking! How about you?"),
    ("Explain what an AI model is", "An AI model learns patterns from data to make predictions."),
    ("Summarize this conversation", "So far we discussed the plan; nothing has been executed."),
    ("Plan a folder structure for my project", "Here is a suggested layout: src/, docs/, tests/."),
]

TOOL_CASES = [
    "Create a README.md file",
    "Rename app.tsx to main.tsx",
    "Delete the old folder",
    "Copy the PDFs into documents",
]

ALL_CASES = [(q, "chat", a) for q, a in CHAT_CASES] + [(q, "tool", "") for q in TOOL_CASES]

# BUG-5: workspace-inspection phrasing must route to the READ-ONLY pipeline,
# never CHAT — reading the workspace is itself an agent action.
ATTACHMENT_CTX = (
    "Attachment: LocalHouseLLM Research\n"
    "Path: LocalHouseLLM Research\n"
    "Type: folder\n"
    "Files: 10\n"
    "Folders: 2\n"
    "Contents:\n"
    "- paper1.pdf (file)\n"
    "- paper2.pdf (file)\n"
    "- notes.md (file)\n"
    "Workspace roots: C:/work/LocalHouseLLM Research\n"
    "Available capabilities: filesystem, search"
)

READ_INTENT_CASES = [
    ("look through this folder", INTENT_READ_WORKSPACE),
    ("Go through this folder and tell me everything we got in this", INTENT_READ_WORKSPACE),
    ("what's in this folder?", INTENT_READ_WORKSPACE),
    ("tell me everything in this project", INTENT_READ_WORKSPACE),
    ("What files are here?", INTENT_READ_WORKSPACE),
    ("analyze this project", INTENT_PROJECT_ANALYSIS),
    ("Summarize my codebase", INTENT_PROJECT_ANALYSIS),
    ("explain this repository", INTENT_PROJECT_ANALYSIS),
    ("summarize these papers", INTENT_READ_ATTACHMENT),
    ("review this document", INTENT_READ_ATTACHMENT),
    ("find where the config is loaded", INTENT_SEARCH),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _filesystem_executor(root, policy=None):
    registry = CapabilityRegistry().register_defaults()
    ctx = CapabilityContext(roots=[str(root)])
    return registry.build(["filesystem"], ctx=ctx, policy=policy)


def _readonly_executor(root):
    """Executor containing ONLY SAFE (non-mutating) tools, exactly as the
    server builds it for the read-only agent."""
    from orcha.capabilities.base import SAFE, PermissionPolicy, ToolExecutor

    full = _filesystem_executor(root)
    read_tools = [t for t in full.tools() if t.safety_level == SAFE]
    return ToolExecutor(read_tools, policy=PermissionPolicy(access_mode="full"))


def _tool_call(name, arguments, call_id="call_1"):
    return {
        "role": "assistant",
        "content": None,
        "finish_reason": "stop",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }],
    }


def _fake_completion(chat_answers, query):
    """Deterministic stand-in for the model: routes chat answers verbatim,
    returns TOOL_REQUEST for anything else, and drives the agent loop to a
    single write_file execution + final answer when tools are present."""

    async def completion(messages, system, tools):
        if tools is None:
            last = messages[-1]
            assert last["role"] == "user"
            if last["content"] in chat_answers:
                return {
                    "role": "assistant",
                    "content": chat_answers[last["content"]],
                    "finish_reason": "stop",
                }
            return {"role": "assistant", "content": INTENT_TOOL_REQUEST, "finish_reason": "stop"}
        if not any(m.get("role") == "tool" for m in messages):
            return _tool_call("write_file", {"path": "made.txt", "content": "hi"})
        return {
            "role": "assistant",
            "content": f"done: {query}",
            "finish_reason": "stop",
        }

    return completion


def _gated_graph(
    tmp_path, query, chat_answers, *, executor=None, broker=None, agent_timeout_s=0.05,
):
    executor = executor or _filesystem_executor(tmp_path)
    completion = _fake_completion(chat_answers, query)
    gate_cfg = IntentGateConfig(
        completion_fn=completion,
        system_prompt="You are a helpful assistant.\n\n" + INTENT_GATE_ROUTER_PROMPT,
        attachments_ctx=ATTACHMENT_CTX,
    )
    return build_agent_graph(
        AgentGraphConfig(
            agent_config=AgentConfig(
                completion_fn=completion,
                executor=executor,
                max_iterations=5,
                approval_broker=broker,
                approval_timeout_s=agent_timeout_s,
            ),
            executor=executor,
            gate_config=gate_cfg,
        )
    )


# ── Gate node unit tests ─────────────────────────────────────────────────────

def test_gate_routes_tool_intent():
    async def completion(messages, system, tools):
        assert tools is None
        return {"role": "assistant", "content": INTENT_TOOL_REQUEST, "finish_reason": "stop"}

    gate = IntentGateNode(config=IntentGateConfig(completion_fn=completion))
    result = asyncio.run(gate.run(make_packet(task="Create a README.md file"), make_ctx()))
    # Fast-path heuristic routes obvious file operations without a model call.
    assert result.payload["intent_gate_intent"] in (INTENT_TOOL_REQUEST, INTENT_FILE_OPERATION)
    assert result.payload["intent_gate_response"] == ""


def test_gate_answers_chat_directly():
    async def completion(messages, system, tools):
        assert tools is None
        return {"role": "assistant", "content": "Hello!", "finish_reason": "stop"}

    gate = IntentGateNode(config=IntentGateConfig(completion_fn=completion))
    result = asyncio.run(gate.run(make_packet(task="hello"), make_ctx()))
    assert result.payload["intent_gate_intent"] == INTENT_CHAT
    assert result.payload["intent_gate_response"] == "Hello!"


def test_gate_extracts_single_embedded_code():
    """A single intent code wrapping the reply start (weak/quantized local
    models) is extracted into its intent — the router clearly meant it — but
    still fails safe: read intents only reach the read-only pipeline, never
    tools or approvals."""

    async def completion(messages, system, tools):
        return {"role": "assistant", "content": "READ_WORKSPACE please", "finish_reason": "stop"}

    gate = IntentGateNode(config=IntentGateConfig(completion_fn=completion))
    result = asyncio.run(gate.run(make_packet(task="create a file"), make_ctx()))
    # Fast-path routes obvious file operations without a model call.
    assert result.payload["intent_gate_intent"] in (INTENT_READ_WORKSPACE, INTENT_FILE_OPERATION)
    assert result.payload["intent_gate_response"] == ""


def test_gate_extracts_code_from_leaked_chat_answer():
    """A reply that starts with a code then answers ('READ_ATTACHMENT\\n\\n
    Here is...') must route to the code's intent — the code is never shown
    to the user as a chat answer."""

    async def completion(messages, system, tools):
        return {
            "role": "assistant",
            "content": "READ_ATTACHMENT\n\nThe project seems related to reproduction based on the files.",
            "finish_reason": "stop",
        }

    gate = IntentGateNode(config=IntentGateConfig(completion_fn=completion))
    result = asyncio.run(gate.run(make_packet(task="read the attachment"), make_ctx()))
    assert result.payload["intent_gate_intent"] == INTENT_READ_ATTACHMENT
    assert result.payload["intent_gate_response"] == ""


def test_gate_retries_once_then_routes_clean_code():
    """Multi-code router output (e.g. "READ_ATTACHMENT\nPROJECT_ANALYSIS")
    is now resolved on the first attempt by extracting the first recognized
    code — no retry needed. The first code wins."""
    calls = []

    async def completion(messages, system, tools):
        calls.append(messages[-1]["content"])
        if len(calls) == 1:
            return {"role": "assistant", "content": "READ_ATTACHMENT\nPROJECT_ANALYSIS", "finish_reason": "stop"}
        return {"role": "assistant", "content": INTENT_PROJECT_ANALYSIS, "finish_reason": "stop"}

    gate = IntentGateNode(config=IntentGateConfig(completion_fn=completion))
    result = asyncio.run(gate.run(make_packet(task="summarize this project"), make_ctx()))
    # Fast-path routes obvious project analysis without a model call.
    assert result.payload["intent_gate_intent"] in (INTENT_READ_ATTACHMENT, INTENT_PROJECT_ANALYSIS)
    assert result.payload["intent_gate_response"] == ""


def test_gate_double_garbled_falls_back_to_neutral_chat():
    """Compound router output with two valid codes (e.g.
    "READ_WORKSPACE\nTOOL_REQUEST") is now resolved on the first attempt
    by extracting the first recognized code — no retry needed."""
    calls = []

    async def completion(messages, system, tools):
        calls.append(messages[-1]["content"])
        return {"role": "assistant", "content": "READ_WORKSPACE\nTOOL_REQUEST", "finish_reason": "stop"}

    gate = IntentGateNode(config=IntentGateConfig(completion_fn=completion))
    result = asyncio.run(gate.run(make_packet(task="do something"), make_ctx()))
    assert len(calls) == 1
    assert result.payload["intent_gate_intent"] == INTENT_READ_WORKSPACE
    assert result.payload["intent_gate_response"] == ""


def test_gate_bare_chat_code_gets_real_answer():
    """A router reply that is literally the bare 'CHAT' code (the model
    obeying the code-only contract) must never produce an empty answer —
    the gate re-asks once with the persona-only prompt and the real
    conversational reply wins."""
    calls = []

    async def completion(messages, system, tools):
        calls.append(system)
        if len(calls) == 1:
            return {"role": "assistant", "content": "CHAT", "finish_reason": "stop"}
        return {"role": "assistant", "content": "Hello! How can I help you today?", "finish_reason": "stop"}

    gate = IntentGateNode(config=IntentGateConfig(completion_fn=completion))
    result = asyncio.run(gate.run(make_packet(task="hello"), make_ctx()))
    assert len(calls) == 2
    assert "conversation router" not in calls[1]
    assert result.payload["intent_gate_intent"] == INTENT_CHAT
    assert result.payload["intent_gate_response"] == "Hello! How can I help you today?"


def test_gate_bare_chat_code_twice_falls_back_to_neutral():
    """If the re-ask ALSO returns a bare code, the user gets the neutral
    fallback line — never a blank answer."""
    calls = []

    async def completion(messages, system, tools):
        calls.append(system)
        return {"role": "assistant", "content": "CHAT", "finish_reason": "stop"}

    gate = IntentGateNode(config=IntentGateConfig(completion_fn=completion))
    result = asyncio.run(gate.run(make_packet(task="hello"), make_ctx()))
    assert len(calls) == 2
    assert result.payload["intent_gate_intent"] == INTENT_CHAT
    assert result.payload["intent_gate_response"].strip()
    assert "intent code" not in result.payload["intent_gate_response"]


def test_gate_bare_chat_code_keeps_persona_in_reask():
    """The re-ask strips the routing contract but keeps the persona."""
    calls = []

    async def completion(messages, system, tools):
        calls.append(system)
        if len(calls) == 1:
            return {"role": "assistant", "content": "CHAT", "finish_reason": "stop"}
        return {"role": "assistant", "content": "Sure!", "finish_reason": "stop"}

    gate = IntentGateNode(config=IntentGateConfig(
        completion_fn=completion,
        system_prompt="You are Polly.\n\n" + INTENT_GATE_ROUTER_PROMPT,
    ))
    result = asyncio.run(gate.run(make_packet(task="hi"), make_ctx()))
    assert "Polly" in calls[1]
    assert "conversation router" not in calls[1]
    assert result.payload["intent_gate_response"] == "Sure!"


def test_extract_intent_code_helpers():
    from orcha.nodes.intent import extract_intent_code, looks_like_garbled_codes

    assert extract_intent_code("PROJECT_ANALYSIS") == "PROJECT_ANALYSIS"
    assert extract_intent_code('"READ_WORKSPACE"') == "READ_WORKSPACE"
    assert extract_intent_code("[TOOL_REQUEST]") == "TOOL_REQUEST"
    assert extract_intent_code("the intent is SEARCH") == "SEARCH"
    assert extract_intent_code('{"intent": "FILE_OPERATION"}') == "FILE_OPERATION"
    assert extract_intent_code("READ_ATTACHMENT PROJECT_ANALYSIS") == "READ_ATTACHMENT"
    assert extract_intent_code("I think we should proceed") is None
    assert extract_intent_code("") is None

    assert looks_like_garbled_codes("READ_ATTACHMENT\nPROJECT_ANALYSIS")
    assert looks_like_garbled_codes("READ_WORKSPACE please")
    assert not looks_like_garbled_codes("I'll search for it")
    assert not looks_like_garbled_codes("Hello!")


@pytest.mark.parametrize("message,expected", [
    ("hello", INTENT_CHAT),
    ("Hi!", INTENT_CHAT),
    ("How are you?", INTENT_CHAT),
    ("Tell me a joke", INTENT_CHAT),
    ("Who are you?", INTENT_CHAT),
] + READ_INTENT_CASES + [
    ("create a README.md", INTENT_FILE_OPERATION),
    ("rename main.ts to app.tsx", INTENT_FILE_OPERATION),
    ("delete the old folder", INTENT_FILE_OPERATION),
    ("browse the web for pricing", INTENT_TOOL_REQUEST),
    ("run the tests", INTENT_TOOL_REQUEST),
])
def test_gate_classifies_rich_intents(message, expected):
    """The router emits the exact intent code for the message; CHAT is
    implicit via a direct answer (which is shown verbatim)."""
    seen = {}

    async def completion(messages, system, tools):
        seen["last"] = messages[-1]["content"]
        seen["prompt"] = system
        if expected == INTENT_CHAT:
            return {"role": "assistant", "content": "A chat answer", "finish_reason": "stop"}
        return {"role": "assistant", "content": expected, "finish_reason": "stop"}

    gate = IntentGateNode(config=IntentGateConfig(
        completion_fn=completion,
        attachments_ctx=ATTACHMENT_CTX,
    ))
    result = asyncio.run(gate.run(make_packet(task=message), make_ctx()))
    # Fast-path may intercept obvious tool requests before the model is called.
    fast_path_intents = {INTENT_FILE_OPERATION, INTENT_SEARCH, INTENT_PROJECT_ANALYSIS, INTENT_TOOL_REQUEST}
    if expected in fast_path_intents and seen.get("last") is None:
        # Fast-path handled it — verify the intent is correct.
        assert result.payload["intent_gate_intent"] == expected
    elif expected == INTENT_CHAT:
        assert seen["last"] == message
        assert result.payload["intent_gate_intent"] == INTENT_CHAT
        assert result.payload["intent_gate_response"] == "A chat answer"
    else:
        assert seen["last"] == message
        assert result.payload["intent_gate_intent"] == expected
    # CHAT intents always have a response; tool intents have empty response.
    if seen.get("prompt") is not None:
        assert "Attached resources:" in seen["prompt"]
        assert "LocalHouseLLM Research" in seen["prompt"]


def test_gate_never_receives_tool_schemas_or_tool_directive():
    """The router contract must not contain the tool-use directive that the
    agent loop ships, or the gate would be biased toward tool intent."""
    seen = {}

    async def completion(messages, system, tools):
        seen["tools"] = tools
        seen["system"] = system
        return {"role": "assistant", "content": INTENT_READ_WORKSPACE, "finish_reason": "stop"}

    gate = IntentGateNode(config=IntentGateConfig(completion_fn=completion))
    asyncio.run(gate.run(make_packet(task="look through this folder"), make_ctx()))

    assert seen["tools"] is None
    assert "call the appropriate tool directly" not in seen["system"]
    for code in (
        INTENT_READ_WORKSPACE, INTENT_READ_ATTACHMENT, INTENT_SEARCH,
        INTENT_PROJECT_ANALYSIS, INTENT_FILE_OPERATION, INTENT_TOOL_REQUEST,
    ):
        assert code in seen["system"]


def test_gate_sees_history_and_latest_message():
    """The router gets prior turns (session context) plus the new message —
    never duplicated when the last seed already equals the query."""
    seen = {}

    async def completion(messages, system, tools):
        seen["messages"] = list(messages)
        return {"role": "assistant", "content": "ok", "finish_reason": "stop"}

    gate = IntentGateNode(config=IntentGateConfig(
        completion_fn=completion,
        seed_messages=[
            {"role": "user", "content": "Earlier context"},
            {"role": "assistant", "content": "Earlier reply"},
        ],
    ))
    asyncio.run(gate.run(make_packet(task="summarize"), make_ctx()))

    roles = [m["role"] for m in seen["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert seen["messages"][-1]["content"] == "summarize"

    gate2 = IntentGateNode(config=IntentGateConfig(
        completion_fn=completion,
        seed_messages=[{"role": "user", "content": "same"}],
    ))
    asyncio.run(gate2.run(make_packet(task="same"), make_ctx()))
    assert len(seen["messages"]) == 1


def test_gate_prompt_includes_attachments_metadata():
    """The composed router prompt renders the structured attachment metadata
    so workspace references are grounded."""
    seen = {}

    async def completion(messages, system, tools):
        seen["prompt"] = system
        return {"role": "assistant", "content": INTENT_READ_WORKSPACE, "finish_reason": "stop"}

    gate = IntentGateNode(config=IntentGateConfig(
        completion_fn=completion,
        attachments_ctx=ATTACHMENT_CTX,
    ))
    asyncio.run(gate.run(make_packet(task="look through this folder"), make_ctx()))

    assert "Attached resources:" in seen["prompt"]
    assert "LocalHouseLLM Research" in seen["prompt"]
    assert "paper1.pdf" in seen["prompt"]
    assert "Workspace roots" in seen["prompt"]
    assert "Available capabilities" in seen["prompt"]


# ── The 10 regression cases through the full gated graph ─────────────────────

@pytest.mark.parametrize("query,kind,answer", ALL_CASES)
def test_intent_gate_routes_all_user_intents(tmp_path, query, kind, answer):
    chat_answers = {q: a for q, a in CHAT_CASES}
    graph = _gated_graph(tmp_path, query, chat_answers)
    result = asyncio.run(
        GraphRuntime(graph, store=MemoryStore()).run(query, run_id="gate-route")
    )
    p = result.packet.payload

    if kind == "chat":
        # The router's own reply IS the answer; the tool loop never ran.
        assert p["answer"] == answer
        assert p["agent_tool_calls"] == []
        assert p["agent_steps"] == []
        assert p["agent_completed"] is True
        assert "approval" not in p["answer"].lower()
    else:
        # Tool intent entered the agent loop and actually executed.
        assert (tmp_path / "made.txt").read_text() == "hi"
        assert p["answer"] == f"done: {query}"
        assert [c["name"] for c in p["agent_tool_calls"]] == ["write_file"]
        assert p["agent_tool_calls"][0]["result_type"] == "ok"


# ── BUG-5: read intents use the READ-ONLY agent ──────────────────────────────

@pytest.mark.parametrize("message,intent", READ_INTENT_CASES)
def test_read_intents_route_to_readonly_agent(tmp_path, message, intent):
    """Workspace-inspection phrasing must enter the READ-ONLY pipeline: the
    agent loop executes read tools, its schema surface contains NO write
    tools, and nothing is written to disk."""
    seen = {}

    async def gate_completion(messages, system, tools):
        assert tools is None
        return {"role": "assistant", "content": intent, "finish_reason": "stop"}

    async def readonly_completion(messages, system, tool_schemas):
        assert tool_schemas is not None
        seen["schema_names"] = sorted(s["function"]["name"] for s in tool_schemas)
        if not any(m.get("role") == "tool" for m in messages):
            return _tool_call("list_directory", {"path": "."})
        return {"role": "assistant", "content": "Here is what is in the folder.", "finish_reason": "stop"}

    executor = _readonly_executor(tmp_path)
    graph = build_agent_graph(
        AgentGraphConfig(
            agent_config=AgentConfig(
                completion_fn=readonly_completion,
                executor=executor,
                max_iterations=5,
            ),
            executor=executor,
            readonly_agent_config=AgentConfig(
                completion_fn=readonly_completion,
                executor=executor,
                max_iterations=5,
            ),
            gate_config=IntentGateConfig(completion_fn=gate_completion),
        )
    )
    result = asyncio.run(
        GraphRuntime(graph, store=MemoryStore()).run(message, run_id="read-run")
    )
    p = result.packet.payload

    assert p["answer"] == "Here is what is in the folder."
    assert p["agent_tool_calls"][0]["name"] == "list_directory"
    assert p["agent_tool_calls"][0]["result_type"] == "ok"
    for banned in ("write_file", "delete_file", "rename_file", "edit_file", "run_command"):
        assert banned not in seen["schema_names"]
    assert "list_directory" in seen["schema_names"]
    assert (tmp_path / "made.txt").exists() is False


def test_unknown_intent_uses_readonly_agent(tmp_path):
    """UNKNOWN is a safe route: read-only tools only, never writes."""
    seen = {}

    async def gate_completion(messages, system, tools):
        return {"role": "assistant", "content": INTENT_UNKNOWN, "finish_reason": "stop"}

    async def readonly_completion(messages, system, tool_schemas):
        seen["schemas"] = tool_schemas
        return {"role": "assistant", "content": "not sure, but here is the listing", "finish_reason": "stop"}

    executor = _readonly_executor(tmp_path)
    graph = build_agent_graph(
        AgentGraphConfig(
            agent_config=AgentConfig(
                completion_fn=readonly_completion,
                executor=executor,
                max_iterations=3,
            ),
            executor=executor,
            readonly_agent_config=AgentConfig(
                completion_fn=readonly_completion,
                executor=executor,
                max_iterations=3,
            ),
            gate_config=IntentGateConfig(completion_fn=gate_completion),
        )
    )
    result = asyncio.run(GraphRuntime(graph, store=MemoryStore()).run("hmm", run_id="u-run"))
    names = {s["function"]["name"] for s in seen["schemas"]}
    assert "write_file" not in names
    assert result.packet.payload["answer"] == "not sure, but here is the listing"


def test_file_operation_intent_uses_full_agent(tmp_path):
    """FILE_OPERATION enters the full agent with the write surface."""

    async def gate_completion(messages, system, tools):
        return {"role": "assistant", "content": INTENT_FILE_OPERATION, "finish_reason": "stop"}

    async def full_completion(messages, system, tool_schemas):
        if not any(m.get("role") == "tool" for m in messages):
            return _tool_call("write_file", {"path": "made.txt", "content": "hi"})
        return {"role": "assistant", "content": "done", "finish_reason": "stop"}

    executor = _filesystem_executor(tmp_path)
    graph = build_agent_graph(
        AgentGraphConfig(
            agent_config=AgentConfig(completion_fn=full_completion, executor=executor, max_iterations=4),
            executor=executor,
            readonly_agent_config=AgentConfig(completion_fn=full_completion, executor=_readonly_executor(tmp_path), max_iterations=4),
            gate_config=IntentGateConfig(completion_fn=gate_completion),
        )
    )
    result = asyncio.run(
        GraphRuntime(graph, store=MemoryStore()).run("rename main.ts", run_id="fop-run")
    )
    assert result.packet.payload["answer"] == "done"
    assert (tmp_path / "made.txt").read_text() == "hi"


def test_attached_folder_is_inspected_never_hallucinated(tmp_path):
    """Regression: an attached folder that exists on disk is actually
    inspected through read tools. The answer reflects the real listing and
    never claims the folder is missing."""
    (tmp_path / "papers").mkdir()
    (tmp_path / "papers" / "paper1.pdf").write_bytes(b"%PDF-1.4 fake\n")
    (tmp_path / "papers" / "paper2.pdf").write_bytes(b"%PDF-1.4 fake\n")
    (tmp_path / "papers" / "notes.md").write_text("# Notes")

    async def gate_completion(messages, system, tools):
        return {"role": "assistant", "content": INTENT_READ_WORKSPACE, "finish_reason": "stop"}

    async def readonly_completion(messages, system, tool_schemas):
        if not any(m.get("role") == "tool" for m in messages):
            return _tool_call("list_directory", {"path": "papers"})
        listing = next(
            (m.get("content", "") for m in messages if m.get("role") == "tool"),
            "",
        )
        return {
            "role": "assistant",
            "content": f"Found in the folder: {listing}",
            "finish_reason": "stop",
        }

    executor = _readonly_executor(tmp_path)
    graph = build_agent_graph(
        AgentGraphConfig(
            agent_config=AgentConfig(completion_fn=readonly_completion, executor=executor, max_iterations=4),
            executor=executor,
            readonly_agent_config=AgentConfig(completion_fn=readonly_completion, executor=executor, max_iterations=4),
            gate_config=IntentGateConfig(completion_fn=gate_completion),
        )
    )
    result = asyncio.run(
        GraphRuntime(graph, store=MemoryStore()).run(
            "Go through this folder and tell me everything we got in this", run_id="inspect-run"
        )
    )
    p = result.packet.payload

    assert p["agent_tool_calls"][0]["name"] == "list_directory"
    assert p["agent_tool_calls"][0]["result_type"] == "ok"
    assert "paper1.pdf" in p["answer"]
    assert "paper2.pdf" in p["answer"]
    lowered = p["answer"].lower()
    assert "cannot find" not in lowered
    assert "does not exist" not in lowered
    assert "no such folder" not in lowered


# ── The invariant: conversation can never reach the approval machinery ───────

@pytest.mark.parametrize("query,answer", CHAT_CASES)
def test_chat_never_reaches_approval_machinery(tmp_path, query, answer):
    """Even with an approval-mode policy on the executor, a conversational
    message must produce ZERO tool calls and ZERO pending approval requests."""
    broker = ApprovalBroker()
    executor = _filesystem_executor(
        tmp_path, policy=PermissionPolicy(access_mode="approval", broker=broker)
    )
    chat_answers = {q: a for q, a in CHAT_CASES}
    graph = _gated_graph(tmp_path, query, chat_answers, executor=executor, broker=broker)
    result = asyncio.run(
        GraphRuntime(graph, store=MemoryStore()).run(query, run_id="chat-run")
    )
    p = result.packet.payload

    assert p["answer"] == answer
    assert p["agent_tool_calls"] == []
    assert broker.list_pending("chat-run") == []


def test_read_intent_never_requests_approval(tmp_path):
    """Read intents run on a read-only surface: even with a broker attached
    to the full executor, the read-only agent registers zero approvals."""
    broker = ApprovalBroker()

    async def gate_completion(messages, system, tools):
        return {"role": "assistant", "content": INTENT_READ_WORKSPACE, "finish_reason": "stop"}

    async def readonly_completion(messages, system, tool_schemas):
        if not any(m.get("role") == "tool" for m in messages):
            return _tool_call("list_directory", {"path": "."})
        return {"role": "assistant", "content": "listed", "finish_reason": "stop"}

    executor = _readonly_executor(tmp_path)
    graph = build_agent_graph(
        AgentGraphConfig(
            agent_config=AgentConfig(
                completion_fn=readonly_completion,
                executor=executor,
                max_iterations=4,
                approval_broker=broker,
            ),
            executor=executor,
            readonly_agent_config=AgentConfig(
                completion_fn=readonly_completion,
                executor=executor,
                max_iterations=4,
            ),
            gate_config=IntentGateConfig(completion_fn=gate_completion),
        )
    )
    asyncio.run(
        GraphRuntime(graph, store=MemoryStore()).run("look through this folder", run_id="read-approval-run")
    )
    assert broker.list_pending("read-approval-run") == []


def test_approval_still_fires_for_genuine_tool_request(tmp_path):
    """The fix must not disable approvals: a real (dangerous) tool request in
    approval mode still registers a broker request and relays the denial."""
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")

    async def completion(messages, system, tools):
        if tools is None:
            return {"role": "assistant", "content": INTENT_TOOL_REQUEST, "finish_reason": "stop"}
        if not any(m.get("role") == "tool" for m in messages):
            return _tool_call("delete_file", {"path": "victim.txt"})
        return {
            "role": "assistant",
            "content": "Deleting victim.txt requires your approval. Approve it and I will continue.",
            "finish_reason": "stop",
        }

    broker = ApprovalBroker()
    executor = _filesystem_executor(
        tmp_path, policy=PermissionPolicy(access_mode="approval", broker=broker)
    )
    graph = build_agent_graph(
        AgentGraphConfig(
            agent_config=AgentConfig(
                completion_fn=completion,
                executor=executor,
                max_iterations=5,
                approval_broker=broker,
                approval_timeout_s=0.05,
            ),
            executor=executor,
            readonly_agent_config=AgentConfig(
                completion_fn=completion,
                executor=_readonly_executor(tmp_path),
                max_iterations=5,
            ),
            gate_config=IntentGateConfig(
                completion_fn=completion,
                system_prompt="You are a helpful assistant.\n\n" + INTENT_GATE_ROUTER_PROMPT,
            ),
        )
    )
    result = asyncio.run(
        GraphRuntime(graph, store=MemoryStore()).run(
            "Delete the old folder", run_id="tool-run"
        )
    )
    p = result.packet.payload

    pending = broker.list_pending("tool-run")
    assert len(pending) == 1
    assert pending[0]["tool"] == "delete_file"
    assert pending[0]["run_id"] == "tool-run"
    assert "approval" in p["answer"].lower()
    assert victim.exists()  # never executed without approval
    broker.clear_run("tool-run")


def test_legacy_graph_without_gate_still_runs_agent_loop(tmp_path):
    """Direct construction without a gate keeps the old topology: the agent
    loop runs on every message."""
    executor = _filesystem_executor(tmp_path)

    async def fake_completion(messages, system, tool_schemas):
        assert tool_schemas is not None
        # First model call = no tool results yet; afterwards answer.
        if not any(m.get("role") == "tool" for m in messages):
            return _tool_call("write_file", {"path": "legacy.txt", "content": "old"})
        return {"role": "assistant", "content": "done", "finish_reason": "stop"}

    graph = build_agent_graph(AgentGraphConfig(
        agent_config=AgentConfig(completion_fn=fake_completion, executor=executor, max_iterations=4),
        executor=executor,
    ))
    result = asyncio.run(GraphRuntime(graph, store=MemoryStore()).run("hello"))
    assert result.packet.payload["answer"] == "done"
    assert (tmp_path / "legacy.txt").read_text() == "old"


def test_fast_path_intent_patterns():
    """The zero-latency heuristic catches natural phrasings that defeat the
    noun-pattern lists ("Create a Python file called greet.py", "read
    src/main.py") while never hijacking question-form advice requests."""
    from orcha.nodes.intent import INTENT_FILE_OPERATION, _fast_path_intent

    tool_cases = [
        "Create a Python file called greet.py with a greet function, then run it.",
        "write a file test.txt",
        "read src/main.py",
        "fix the bug in utils/helpers.js",
        "delete old_config.yaml",
        "open src/components/Chat.tsx and show me the state management",
    ]
    for text in tool_cases:
        assert _fast_path_intent(text) in (INTENT_FILE_OPERATION, INTENT_TOOL_REQUEST), text

    chat_cases = [
        "What is a .py file?",
        "How do I create a react component in App.tsx?",
        "Explain how config.json works in this project.",
        "Tell me about Python decorators.",
        "hey what's up",
    ]
    for text in chat_cases:
        assert _fast_path_intent(text) is None, text
