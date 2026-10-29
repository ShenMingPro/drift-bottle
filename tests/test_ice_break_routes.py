"""真实 SQLite 检查点与 HTTP 路由集成；不启动 MySQL、MCP 或远端模型。"""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

import app.agents.ice_break_agent as ice_break
import app.ice_break_checkpoint as checkpoint
import app.routers.ice_break as routes
from app.auth_jwt import create_access_token
from app.config import settings
from app.database import get_db


def headers(user_id=7):
    return {"Authorization": f"Bearer {create_access_token(user_id)}"}


def parameters(request_id=None, user_id=7, target_user_id=8):
    params = {"userId": user_id, "targetUserId": target_user_id}
    if request_id is not None:
        params["requestId"] = request_id
    return params


def events(response):
    result = []
    for block in response.text.replace("\r\n", "\n").split("\n\n"):
        event_name = "message"
        data_lines = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.partition(":")[2].strip()
            elif line.startswith("data:"):
                data_lines.append(line.partition(":")[2].lstrip())
        if data_lines:
            result.append((event_name, json.loads("\n".join(data_lines))))
    return result


@pytest.fixture
def route_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setattr(settings, "JWT_SECRET", "ice-break-route-tests-only-secret-key-2026")
    monkeypatch.setattr(checkpoint, "_thread_locks", {})
    observations = SimpleNamespace(
        model_inputs=[], queries=[], failures_left=0, reply_delay=0,
        chat_text="对方：最近喜欢打篮球", graph=None,
    )

    async def reply(messages):
        observations.model_inputs.append(list(messages))
        number = len(observations.model_inputs)
        if observations.reply_delay:
            await asyncio.sleep(observations.reply_delay)
        if observations.failures_left:
            observations.failures_left -= 1
            raise RuntimeError("模拟临时模型错误")
        return AIMessage(content=json.dumps({
            "hints": [f"第{number}次调用建议{index}" for index in range(1, 4)],
            "scenario": "reply",
        }, ensure_ascii=False))

    async def profile(target_user_id):
        observations.queries.append(("profile", target_user_id))
        return json.dumps({"昵称": f"同学{target_user_id}", "爱好": "篮球"}, ensure_ascii=False)

    async def history(user_id, target_user_id):
        observations.queries.append(("history", user_id, target_user_id))
        return json.dumps({"场景": "回复建议", "聊天记录": observations.chat_text}, ensure_ascii=False)

    async def unused_database():
        # 兼容普通接口现有依赖；业务数据库操作全部由上面的查询替身处理。
        yield object()

    monkeypatch.setattr(ice_break, "_get_llm", lambda: RunnableLambda(reply))
    monkeypatch.setattr(ice_break, "_query_user_profile", profile)
    monkeypatch.setattr(ice_break, "_query_chat_history", history)

    @asynccontextmanager
    async def client():
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "routes.sqlite")) as saver:
            graph = ice_break.build_ice_break_graph().compile(checkpointer=saver)
            observations.graph = graph
            monkeypatch.setattr(routes, "get_ice_break_app", lambda: graph)
            app = FastAPI()
            app.include_router(routes.router, prefix="/api")
            app.dependency_overrides[get_db] = unused_database
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as session:
                yield session

    yield SimpleNamespace(client=client, observations=observations)
    assert checkpoint._thread_locks == {}


def test_stream_executes_model_once_and_sends_one_result(route_runtime):
    async def run():
        async with route_runtime.client() as client:
            response = await client.get(
                "/api/ice-break/stream", params=parameters(str(uuid4())), headers=headers()
            )
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            received = events(response)
            results = [data for name, data in received if name == "result"]
            assert len(results) == 1
            assert results[0]["hints"] == [f"第1次调用建议{index}" for index in range(1, 4)]
            assert results[0]["scenario"] == "reply"
            assert received[-1][0] == "done"
            assert not any(name == "error" for name, _data in received)

    asyncio.run(run())
    assert len(route_runtime.observations.model_inputs) == 1
    assert len(route_runtime.observations.queries) == 2


def test_normal_fallback_reuses_completed_stream_request(route_runtime):
    request_id = str(uuid4())

    async def run():
        async with route_runtime.client() as client:
            streamed = await client.get(
                "/api/ice-break/stream", params=parameters(request_id), headers=headers()
            )
            original = next(data for name, data in events(streamed) if name == "result")
            fallback = await client.get(
                "/api/ice-break", params=parameters(request_id), headers=headers()
            )
            assert fallback.status_code == 200
            assert fallback.json() == original
            assert fallback.headers["cache-control"] == "no-store"

    asyncio.run(run())
    assert len(route_runtime.observations.model_inputs) == 1
    assert len(route_runtime.observations.queries) == 2


def test_new_chat_request_does_not_reuse_workflow_analysis_from_checkpoint(route_runtime):
    async def run():
        async with route_runtime.client() as client:
            graph = route_runtime.observations.graph
            await graph.aupdate_state(
                routes._config(7, 8),
                {
                    "user_id": 7, "target_user_id": 8, "request_id": "previous-request",
                    "analysis_context": {"compatibility": "旧工作流的独立结论"},
                },
                as_node="fetch_context",
            )
            response = await client.get(
                "/api/ice-break", params=parameters(str(uuid4())), headers=headers()
            )
            assert response.status_code == 200
            prompt = route_runtime.observations.model_inputs[-1][0].content
            assert "旧工作流的独立结论" not in prompt
            saved = await graph.aget_state(routes._config(7, 8))
            assert saved.values["analysis_context"] is None

    asyncio.run(run())


def test_concurrent_requests_with_same_id_generate_only_once(route_runtime):
    request_id = str(uuid4())
    route_runtime.observations.reply_delay = 0.03

    async def run():
        async with route_runtime.client() as client:
            responses = await asyncio.gather(*[
                client.get("/api/ice-break", params=parameters(request_id), headers=headers())
                for _ in range(2)
            ])
            assert all(response.status_code == 200 for response in responses)
            assert responses[0].json() == responses[1].json()

    asyncio.run(run())
    assert len(route_runtime.observations.model_inputs) == 1
    assert len(route_runtime.observations.queries) == 2


def test_same_id_recovers_failed_generation_without_requerying_context(route_runtime):
    request_id = str(uuid4())
    route_runtime.observations.failures_left = 1

    async def run():
        async with route_runtime.client() as client:
            failed = await client.get(
                "/api/ice-break/stream", params=parameters(request_id), headers=headers()
            )
            assert failed.status_code == 200
            assert any(name == "error" for name, _data in events(failed))
            assert not any(name == "result" for name, _data in events(failed))
            assert len(route_runtime.observations.queries) == 2

            recovered = await client.get(
                "/api/ice-break", params=parameters(request_id), headers=headers()
            )
            assert recovered.status_code == 200
            assert recovered.json()["hints"] == [f"第2次调用建议{index}" for index in range(1, 4)]

    asyncio.run(run())
    assert len(route_runtime.observations.model_inputs) == 2
    assert len(route_runtime.observations.queries) == 2


@pytest.mark.parametrize("user_id,target_user_id", [(7, 9), (9, 8), (8, 7)])
def test_request_cache_is_isolated_by_user_and_target(route_runtime, user_id, target_user_id):
    request_id = str(uuid4())

    async def run():
        async with route_runtime.client() as client:
            first = await client.get(
                "/api/ice-break", params=parameters(request_id), headers=headers()
            )
            other = await client.get(
                "/api/ice-break", params=parameters(request_id, user_id, target_user_id),
                headers=headers(user_id),
            )
            assert first.status_code == other.status_code == 200
            assert other.json()["hints"] != first.json()["hints"]
            other_prompt = "\n".join(
                str(message.content) for message in route_runtime.observations.model_inputs[1]
            )
            assert all(hint not in other_prompt for hint in first.json()["hints"])

    asyncio.run(run())
    assert len(route_runtime.observations.model_inputs) == 2


def test_new_request_reads_latest_chat_and_prior_suggestions(route_runtime):
    async def run():
        async with route_runtime.client() as client:
            first = await client.get(
                "/api/ice-break", params=parameters(str(uuid4())), headers=headers()
            )
            route_runtime.observations.chat_text = "对方：周末打算去看电影"
            second = await client.get(
                "/api/ice-break", params=parameters(str(uuid4())), headers=headers()
            )
            assert first.status_code == second.status_code == 200
            new_prompt = "\n".join(
                str(message.content) for message in route_runtime.observations.model_inputs[1]
            )
            assert "对方：周末打算去看电影" in new_prompt
            assert "对方：最近喜欢打篮球" not in new_prompt
            assert all(hint in new_prompt for hint in first.json()["hints"])

    asyncio.run(run())
    assert len(route_runtime.observations.model_inputs) == 2
    assert len(route_runtime.observations.queries) == 4


def test_earlier_request_returns_its_own_cached_result_after_new_request(route_runtime):
    first_id, second_id = str(uuid4()), str(uuid4())

    async def run():
        async with route_runtime.client() as client:
            first = await client.get("/api/ice-break", params=parameters(first_id), headers=headers())
            second = await client.get("/api/ice-break", params=parameters(second_id), headers=headers())
            replay = await client.get("/api/ice-break", params=parameters(first_id), headers=headers())
            assert first.status_code == second.status_code == replay.status_code == 200
            assert replay.json() == first.json()
            assert replay.json() != second.json()

    asyncio.run(run())
    assert len(route_runtime.observations.model_inputs) == 2
    assert len(route_runtime.observations.queries) == 4


def test_omitting_request_id_still_starts_independent_generations(route_runtime):
    async def run():
        async with route_runtime.client() as client:
            first = await client.get("/api/ice-break", params=parameters(), headers=headers())
            second = await client.get("/api/ice-break", params=parameters(), headers=headers())
            assert first.status_code == second.status_code == 200
            assert first.json()["hints"] != second.json()["hints"]

    asyncio.run(run())
    assert len(route_runtime.observations.model_inputs) == 2


@pytest.mark.parametrize("path", ["/api/ice-break", "/api/ice-break/stream"])
def test_http_authentication_and_uuid_validation_do_not_run_graph(route_runtime, path):
    async def run():
        async with route_runtime.client() as client:
            unauthenticated = await client.get(path, params=parameters(str(uuid4())))
            assert unauthenticated.status_code == 401

            forbidden = await client.get(
                path, params=parameters(str(uuid4()), user_id=9), headers=headers(7)
            )
            assert forbidden.status_code == 403

            invalid_id = await client.get(path, params=parameters("invalid-uuid"), headers=headers())
            assert invalid_id.status_code == 422

    asyncio.run(run())
    assert route_runtime.observations.model_inputs == []
    assert route_runtime.observations.queries == []
