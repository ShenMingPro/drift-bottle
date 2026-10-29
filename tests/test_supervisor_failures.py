import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

import app.agents.supervisor_agent as supervisor
from app.config import settings


class AsyncTool:
    def __init__(self, callback):
        self.coroutine = callback

    async def ainvoke(self, arguments):
        return await self.coroutine(arguments)


class Model:
    def __init__(self, callback):
        self.callback = callback
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, _messages):
        self.calls += 1
        return await self.callback(self.calls)


def tool_call():
    return AIMessage(content="", tool_calls=[{
        "name": "call_match_agent", "args": {}, "id": "match-1", "type": "tool_call",
    }])


def state_with_tool(**kwargs):
    state = supervisor.create_supervisor_state(user_id=7, action="pick", **kwargs)
    state["messages"] = [tool_call()]
    return state


@pytest.fixture(autouse=True)
def fast_policy(monkeypatch):
    monkeypatch.setitem(settings.__dict__, "AGENT_MAX_ATTEMPTS", 2)
    monkeypatch.setitem(settings.__dict__, "AGENT_CALL_TIMEOUT_SECONDS", 0.02)
    monkeypatch.setattr(supervisor, "RETRY_DELAY_SECONDS", 0, raising=False)


def test_transient_worker_failure_retries_and_clears_error(monkeypatch):
    attempts = []

    async def worker(_args):
        attempts.append(1)
        if len(attempts) == 1:
            raise TimeoutError("temporary failure")
        return {"agent": "match", "status": "success", "best_match": {"bottle_id": 12, "user_id": 23}}

    monkeypatch.setattr(supervisor, "supervisor_tool_map", {"call_match_agent": AsyncTool(worker)})
    result = asyncio.run(supervisor.execute_tools(state_with_tool()))

    assert len(attempts) == 2
    assert result["status"] == "running"
    assert result["error"] is None
    assert result["retry_count"] == 1
    assert result["step_count"] == 2
    assert result["completed_steps"] == ["match"]


def test_exhausted_worker_failure_stops_graph_without_false_success(monkeypatch):
    attempts = []

    async def worker(_args):
        attempts.append(1)
        raise TimeoutError("private provider details")

    async def planner(_call):
        return tool_call()

    model = Model(planner)
    monkeypatch.setattr(supervisor, "_get_llm", lambda: model)
    monkeypatch.setattr(supervisor, "supervisor_tool_map", {"call_match_agent": AsyncTool(worker)})
    result = asyncio.run(asyncio.wait_for(supervisor.supervisor_app.ainvoke(
        supervisor.create_supervisor_state(user_id=7, action="pick")), timeout=1))

    assert len(attempts) == 2
    assert model.calls == 1
    assert result["status"] == "failed"
    assert result["guard_passed"] is False
    assert result["error"]["error_code"] == "CALL_TIMEOUT"
    assert result["retry_count"] == 1
    assert result["messages"][-1].status == "error"


def test_permanent_worker_error_is_not_retried(monkeypatch):
    attempts = []

    async def worker(_args):
        attempts.append(1)
        raise ValueError("bad arguments")

    monkeypatch.setattr(supervisor, "supervisor_tool_map", {"call_match_agent": AsyncTool(worker)})
    result = asyncio.run(supervisor.execute_tools(state_with_tool()))

    assert len(attempts) == 1
    assert result["status"] == "failed"
    assert result["retry_count"] == 0
    assert result["error"]["retryable"] is False


def test_failed_payload_is_terminal(monkeypatch):
    async def worker(_args):
        return {"agent": "match", "status": "failed", "error": "upstream failed"}

    monkeypatch.setattr(supervisor, "supervisor_tool_map", {"call_match_agent": AsyncTool(worker)})
    result = asyncio.run(supervisor.execute_tools(state_with_tool()))

    assert result["status"] == "failed"
    assert result["completed_steps"] == []
    assert result["messages"][0].status == "error"


def test_supervisor_timeout_is_captured_and_bounded(monkeypatch):
    async def planner(_call):
        await asyncio.Event().wait()

    model = Model(planner)
    monkeypatch.setattr(supervisor, "_get_llm", lambda: model)
    result = asyncio.run(asyncio.wait_for(supervisor.supervisor_app.ainvoke(
        supervisor.create_supervisor_state(user_id=7, action="pick")), timeout=1))

    assert model.calls == 2
    assert result["status"] == "failed"
    assert result["guard_passed"] is False
    assert result["error"]["error_code"] == "CALL_TIMEOUT"
    assert result["retry_count"] == 1


def test_supervisor_recovers_after_transient_model_failure(monkeypatch):
    async def planner(call):
        if call == 1:
            raise ConnectionError("connection reset")
        if call == 2:
            return tool_call()
        return AIMessage(content="done")

    async def worker(_args):
        return {"agent": "match", "status": "no_result", "best_match": None}

    model = Model(planner)
    monkeypatch.setattr(supervisor, "_get_llm", lambda: model)
    monkeypatch.setattr(supervisor, "supervisor_tool_map", {"call_match_agent": AsyncTool(worker)})
    result = asyncio.run(supervisor.supervisor_app.ainvoke(
        supervisor.create_supervisor_state(user_id=7, action="pick")))

    assert result["status"] == "no_result"
    assert result["error"] is None
    assert result["retry_count"] == 1


def test_extract_result_never_promotes_unresolved_failure_to_success():
    state = state_with_tool()
    state["artifacts"] = {"match": {"agent": "match", "status": "retryable_error", "error": "timeout"}}
    state["error"] = state["artifacts"]["match"]
    result = supervisor.extract_result(state)

    assert result["status"] == "failed"
    assert result["guard_passed"] is False


def test_cancellation_propagates_without_retry(monkeypatch):
    async def worker(_args):
        raise asyncio.CancelledError()

    monkeypatch.setattr(supervisor, "supervisor_tool_map", {"call_match_agent": AsyncTool(worker)})
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(supervisor.execute_tools(state_with_tool()))


def test_time_budget_exceeded_skips_new_worker_calls(monkeypatch):
    """总时间预算耗尽后，不再调用任何新的子 Agent。"""
    calls = []

    async def worker(_args):
        calls.append(1)
        return {"agent": "match", "status": "success"}

    state = state_with_tool()
    state["deadline_at"] = time.monotonic() - 1
    monkeypatch.setattr(supervisor, "supervisor_tool_map", {"call_match_agent": AsyncTool(worker)})
    result = asyncio.run(supervisor.execute_tools(state))

    assert calls == []
    assert result["status"] == "budget_exceeded"
    assert result["error"]["error_code"] == "TIME_BUDGET_EXCEEDED"
    assert result["messages"][0].status == "error"


def test_time_budget_exceeded_stops_supervisor_llm(monkeypatch):
    """预算耗尽时 Supervisor 不应再发起模型推理。"""
    model = Model(lambda _call: tool_call())
    monkeypatch.setattr(supervisor, "_get_llm", lambda: model)

    state = supervisor.create_supervisor_state(user_id=7, action="pick")
    state["deadline_at"] = time.monotonic() - 1
    result = asyncio.run(supervisor.agent_node(state))

    assert model.calls == 0
    assert result["status"] == "budget_exceeded"
    assert result["error"]["error_code"] == "TIME_BUDGET_EXCEEDED"


def test_research_failure_does_not_become_successful_evidence(monkeypatch):
    import app.agents.ai_chat_agent as ai_chat

    async def invoke(_state):
        return {"status": "failed", "error": {"error": "MCP unavailable"}, "final_response": "friendly error text"}

    monkeypatch.setattr(ai_chat, "ai_chat_app", SimpleNamespace(ainvoke=invoke))
    payload = json.loads(asyncio.run(supervisor.call_research_agent.ainvoke({"user_id": 7, "query": "热点"})))
    assert payload["status"] in {"failed", "fatal_error", "retryable_error"}
    assert not payload.get("evidence")


def test_model_cannot_invent_a_successful_match_without_tool_artifact():
    state = supervisor.create_supervisor_state(user_id=7, action="pick")
    state["messages"] = [AIMessage(content=json.dumps({
        "action": "pick", "result": {"best_match": {"bottle_id": 12, "user_id": 23}},
    }))]
    result = supervisor.extract_result(state)
    assert result["status"] == "failed"
    assert result["error"]["error_code"] == "MISSING_AGENT_RESULT"
    assert result["guard_passed"] is False


def test_partial_full_workflow_cannot_report_success():
    state = supervisor.create_supervisor_state(user_id=7, action="pick_full")
    state["artifacts"] = {"match": {
        "agent": "match", "status": "success", "best_match": {"bottle_id": 12, "user_id": 23},
    }}
    result = supervisor.extract_result(state)
    assert result["status"] == "failed"
    assert result["error"]["error_code"] == "MISSING_AGENT_RESULT"
    assert result["guard_passed"] is False


def test_success_on_last_available_step_is_allowed(monkeypatch):
    async def worker(_args):
        return {"agent": "match", "status": "success", "best_match": {"bottle_id": 12, "user_id": 23}}

    monkeypatch.setattr(supervisor, "supervisor_tool_map", {"call_match_agent": AsyncTool(worker)})
    state = state_with_tool(max_steps=1)
    updates = asyncio.run(supervisor.execute_tools(state))
    result = supervisor.extract_result({**state, **updates})
    assert result["status"] == "success"
    assert result["error"] is None
    assert result["step_count"] == 1
