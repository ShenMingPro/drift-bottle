"""AI 聊天 SSE 只能执行一次 LangGraph，避免重复调用模型和 MCP 工具。"""

import asyncio
import json
from types import SimpleNamespace

import app.routers.ai_chat as route


class FakeGraph:
    def __init__(self):
        self.stream_calls = 0
        self.invoke_calls = 0

    async def astream_events(self, _state, _config, version):
        self.stream_calls += 1
        yield {"event": "on_chain_start", "parent_ids": [], "run_id": "root"}
        yield {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": SimpleNamespace(
                    content="你好",
                    tool_calls=[],
                    additional_kwargs={},
                )
            },
        }
        yield {
            "event": "on_chain_end",
            "run_id": "root",
            "data": {"output": {"final_response": "你好"}},
        }

    async def ainvoke(self, _state, _config):
        self.invoke_calls += 1
        raise AssertionError("流式请求不应再次 ainvoke")


class FakeEventSourceResponse:
    def __init__(self, generator, headers=None):
        self.generator = generator
        self.headers = headers


def test_stream_reuses_same_graph_execution(monkeypatch):
    graph = FakeGraph()
    monkeypatch.setattr(route, "ai_chat_app", graph)
    monkeypatch.setattr(
        "sse_starlette.sse.EventSourceResponse",
        FakeEventSourceResponse,
    )

    async def run():
        response = await route.stream_chat_with_ai(
            userId=7,
            message="你好",
            current_user_id=7,
        )
        events = [event async for event in response.generator]
        return events

    events = asyncio.run(run())
    done = next(event for event in events if event["event"] == "done")
    done_data = json.loads(done["data"])

    assert graph.stream_calls == 1
    assert graph.invoke_calls == 0
    assert done_data["response"] == "你好"
