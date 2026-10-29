"""真实 MCP → Agent → API 链路中的故障不能被包装成成功。"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage

import app.agents.ai_chat_agent as agent
import app.mcp_client as mcp
import app.routers.ai_chat as route
from app.config import settings


def run(coroutine):
    return asyncio.run(coroutine)


def client_with(handler):
    client = mcp.MCPClient()
    client.session = SimpleNamespace(call_tool=handler)
    client._tools_cache = [{"name": "lookup", "description": "query", "inputSchema": {"type": "object"}}]
    return client


def test_mcp_error_flag_raises_instead_of_returning_success_text():
    async def handler(*_args):
        return SimpleNamespace(isError=True, content=[SimpleNamespace(text="upstream secret")])

    with pytest.raises(Exception) as failure:
        run(client_with(handler).call_tool("lookup"))
    assert failure.value.error_code == "MCP_TOOL_ERROR"
    assert "upstream secret" not in str(failure.value)


def test_mcp_timeout_raises_without_automatic_reexecution(monkeypatch):
    calls = []

    async def handler(*_args):
        calls.append(1)
        await asyncio.Event().wait()

    monkeypatch.setattr(settings, "AGENT_CALL_TIMEOUT_SECONDS", 0.01, raising=False)
    with pytest.raises(Exception) as failure:
        run(client_with(handler).call_tool("lookup"))
    assert failure.value.error_code == "MCP_TOOL_TIMEOUT"
    assert calls == [1]


def test_mcp_success_preserves_text():
    async def handler(*_args):
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text="result")])

    assert run(client_with(handler).call_tool("lookup")) == "result"


class ToolCallingModel:
    def __init__(self):
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, _messages):
        self.calls += 1
        return AIMessage(content="", tool_calls=[{
            "name": "lookup", "args": {}, "id": f"call-{self.calls}", "type": "tool_call",
        }])


def test_mcp_failure_terminates_graph_without_fabricated_answer(monkeypatch):
    async def handler(*_args):
        raise RuntimeError("private-provider-token")

    client = client_with(handler)
    monkeypatch.setattr(mcp.mcp_manager, "_clients", {"test": client})
    class FabricatingModel(ToolCallingModel):
        async def ainvoke(self, messages):
            if self.calls:
                self.calls += 1
                return AIMessage(content="fabricated answer after failed lookup")
            return await super().ainvoke(messages)

    model = FabricatingModel()
    monkeypatch.setattr(agent, "_get_llm", lambda: model)

    result = run(agent.ai_chat_app.ainvoke({"user_id": 7, "messages": [HumanMessage(content="query")]}))

    assert result["status"] == "failed"
    assert result["final_response"] == ""
    assert result["error"]["error_code"] == "MCP_TOOL_FAILED"
    assert result["messages"][-1].status == "error"
    assert "private-provider-token" not in result["messages"][-1].content
    assert model.calls == 1


def test_tool_call_budget_ends_actual_graph(monkeypatch):
    calls = []

    async def handler(*_args):
        calls.append(1)
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text="result")])

    monkeypatch.setattr(mcp.mcp_manager, "_clients", {"test": client_with(handler)})
    monkeypatch.setattr(agent, "AI_CHAT_MAX_TOOL_CALLS", 2, raising=False)
    monkeypatch.setattr(agent, "_get_llm", lambda: ToolCallingModel())

    result = run(agent.ai_chat_app.ainvoke({"user_id": 7, "messages": [HumanMessage(content="query")]}))

    assert result["status"] == "failed"
    assert result["error"]["error_code"] == "TOOL_BUDGET_EXCEEDED"
    assert len(calls) == 2


def test_model_timeout_returns_structured_failure(monkeypatch):
    class SlowModel:
        async def ainvoke(self, _messages):
            await asyncio.Event().wait()

    monkeypatch.setattr(settings, "AGENT_CALL_TIMEOUT_SECONDS", 0.01, raising=False)
    monkeypatch.setattr(mcp.mcp_manager, "_clients", {})
    monkeypatch.setattr(agent, "_get_llm", SlowModel)

    result = run(agent.ai_chat_app.ainvoke({"user_id": 7, "messages": [HumanMessage(content="query")]}))
    assert result["status"] == "failed"
    assert result["error"]["error_code"] == "MODEL_TIMEOUT"


def test_nonstream_route_rejects_failed_state_without_leaking_detail(monkeypatch):
    async def invoke(*_args):
        return {"status": "failed", "error": {"error_code": "MCP_TOOL_FAILED", "error": "private-provider-token"}}

    monkeypatch.setattr(route, "ai_chat_app", SimpleNamespace(ainvoke=invoke))
    with pytest.raises(HTTPException) as failure:
        run(route.chat_with_ai(route.ChatRequest(userId=7, message="query"), current_user_id=7))
    assert failure.value.status_code == 502
    assert "private-provider-token" not in str(failure.value.detail)


def test_stream_route_emits_error_without_done_on_failed_final_state(monkeypatch):
    class FailedGraph:
        async def astream_events(self, *_args, **_kwargs):
            yield {"event": "on_chain_start", "parent_ids": [], "run_id": "root"}
            yield {"event": "on_chain_end", "run_id": "root", "data": {"output": {
                "status": "failed", "error": {"error_code": "MCP_TOOL_FAILED", "error": "private-provider-token"},
                "final_response": "",
            }}}

    monkeypatch.setattr(route, "ai_chat_app", FailedGraph())
    monkeypatch.setattr("sse_starlette.sse.EventSourceResponse", lambda generator: SimpleNamespace(generator=generator))

    async def collect():
        response = await route.stream_chat_with_ai(userId=7, message="query", current_user_id=7)
        return [event async for event in response.generator]

    events = run(collect())
    assert [event["event"] for event in events] == ["progress", "error"]
    assert "private-provider-token" not in json.dumps(events)


def test_stream_ending_without_final_state_is_reported_as_failure(monkeypatch):
    class InterruptedGraph:
        async def astream_events(self, *_args, **_kwargs):
            yield {"event": "on_chain_start", "parent_ids": [], "run_id": "root"}

    monkeypatch.setattr(route, "ai_chat_app", InterruptedGraph())
    monkeypatch.setattr("sse_starlette.sse.EventSourceResponse", lambda generator: SimpleNamespace(generator=generator))

    async def collect():
        response = await route.stream_chat_with_ai(userId=7, message="query", current_user_id=7)
        return [event async for event in response.generator]

    events = run(collect())
    assert [event["event"] for event in events] == ["progress", "error"]


def test_normal_chat_succeeds_and_clears_error(monkeypatch):
    class NormalModel:
        async def ainvoke(self, _messages):
            return AIMessage(content="你好")

    monkeypatch.setattr(mcp.mcp_manager, "_clients", {})
    monkeypatch.setattr(agent, "_get_llm", NormalModel)
    result = run(agent.ai_chat_app.ainvoke({"user_id": 7, "messages": [HumanMessage(content="hello")]}))
    assert result["status"] == "success"
    assert result["error"] is None
    assert result["final_response"] == "你好"


@pytest.mark.parametrize("response", [
    SimpleNamespace(content="not an AIMessage"),
    AIMessage(content="not a valid tool result", invalid_tool_calls=[{
        "name": "lookup", "args": "{", "id": "invalid-call", "error": "invalid JSON",
    }]),
    AIMessage(content=[{"type": "image", "url": "https://example.com/image.png"}]),
])
def test_invalid_model_response_never_becomes_success(monkeypatch, response):
    class InvalidModel:
        async def ainvoke(self, _messages):
            return response

    monkeypatch.setattr(mcp.mcp_manager, "_clients", {})
    monkeypatch.setattr(agent, "_get_llm", InvalidModel)
    result = run(agent.ai_chat_app.ainvoke({"user_id": 7, "messages": [HumanMessage(content="hello")]}))
    assert result["status"] == "failed"
    assert result["error"]["error_code"] == "INVALID_MODEL_RESPONSE"
    assert result["final_response"] == ""


def test_empty_model_response_is_reported_as_failure(monkeypatch):
    class EmptyModel:
        async def ainvoke(self, _messages):
            return AIMessage(content="")

    monkeypatch.setattr(mcp.mcp_manager, "_clients", {})
    monkeypatch.setattr(agent, "_get_llm", EmptyModel)
    result = run(agent.ai_chat_app.ainvoke({"user_id": 7, "messages": [HumanMessage(content="hello")]}))
    assert result["status"] == "failed"
    assert result["error"]["error_code"] == "EMPTY_MODEL_RESPONSE"
