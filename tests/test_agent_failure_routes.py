"""Agent 的技术失败必须返回错误响应，不能提交业务或伪装成空结果。"""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

import app.routers.bottle as bottle
import app.routers.chat as chat


class NoWriteDatabase:
    def __init__(self):
        self.executions = []

    @asynccontextmanager
    async def session(self, *, transaction=False):
        assert not transaction, "Agent 失败后不能开始业务写入事务"
        yield self

    async def execute(self, query, args):
        self.executions.append(query)
        assert query.startswith("SELECT"), "Agent 失败后不能扣次数或写入记录"

    async def fetchone(self):
        return {"user1_id": 7, "user2_id": 8}


def request_for(name):
    return {
        "pick": ("GET", "/api/pick?userId=7", None),
        "full": ("GET", "/api/pick/full?userId=7", None),
        "write": ("POST", "/api/write", {"userId": 7, "content": "你好"}),
        "chat": ("POST", "/api/messages", {"conversationId": 1, "senderId": 7, "content": "你好"}),
    }[name]


def call_route(name, monkeypatch, callback):
    database = NoWriteDatabase()
    graph = SimpleNamespace(ainvoke=callback)
    monkeypatch.setattr(bottle, "supervisor_app", graph)
    monkeypatch.setattr(chat, "supervisor_app", graph)
    app = FastAPI()
    app.include_router(bottle.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    app.dependency_overrides[bottle.get_current_user] = lambda: 7
    app.dependency_overrides[bottle.get_database] = lambda: database
    method, url, body = request_for(name)

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.request(method, url, json=body)

    return asyncio.run(run())


@pytest.mark.parametrize("route", ["pick", "full", "write", "chat"])
@pytest.mark.parametrize("status", ["failed", "budget_exceeded", "success"])
def test_failed_agent_returns_service_error_without_writing(monkeypatch, route, status):
    async def failed(*_args):
        return {
            "status": status,
            "error": {"error": "internal-secret", "error_code": "UPSTREAM_UNAVAILABLE", "retryable": True},
            "guard_passed": True,
            "final_result": json.dumps({"result": {
                "best_match": {"bottle_id": 12, "user_id": 23},
                "ice_break": {"hints": ["不能展示"]},
            }}),
        }

    response = call_route(route, monkeypatch, failed)
    assert response.status_code == 503
    assert isinstance(response.json()["detail"], str)
    assert "internal-secret" not in response.text


@pytest.mark.parametrize("route", ["pick", "full", "write", "chat"])
@pytest.mark.parametrize(("error", "expected"), [(TimeoutError("internal-secret"), 504), (RuntimeError("internal-secret"), 502)])
def test_uncaught_agent_exception_is_safe_http_error(monkeypatch, route, error, expected):
    async def failed(*_args):
        raise error

    response = call_route(route, monkeypatch, failed)
    assert response.status_code == expected
    assert "internal-secret" not in response.text


def test_legitimate_no_result_remains_successful_empty_pick(monkeypatch):
    async def no_result(*_args):
        return {"status": "no_result", "error": None, "final_result": json.dumps({"result": {"best_match": None}})}

    response = call_route("pick", monkeypatch, no_result)
    assert response.status_code == 200
    assert response.json()["hasBottle"] is False


@pytest.mark.parametrize("payload", ["not json", "[]", '{"result": []}'])
@pytest.mark.parametrize("route", ["pick", "full"])
def test_invalid_agent_result_is_not_no_result(monkeypatch, route, payload):
    async def invalid(*_args):
        return {"status": "success", "final_result": payload}

    assert call_route(route, monkeypatch, invalid).status_code == 502


@pytest.mark.parametrize("candidate", [{"bottle_id": 12}, {"bottle_id": "invalid", "user_id": 23}, {"bottle_id": -1, "user_id": 23}])
@pytest.mark.parametrize("route", ["pick", "full"])
def test_invalid_candidate_is_rejected_before_business_commit(monkeypatch, route, candidate):
    async def invalid(*_args):
        match = {"status": "success", "best_match": candidate}
        return {
            "status": "success", "guard_passed": True,
            "final_result": json.dumps({"result": match if route == "pick" else {"match": match}}),
        }

    assert call_route(route, monkeypatch, invalid).status_code == 502


@pytest.mark.parametrize("result", [None, {"status": None}, {"status": "unknown"}, {"status": []}])
def test_invalid_agent_envelope_does_not_allow_business_action(monkeypatch, result):
    async def invalid(*_args):
        return result

    assert call_route("write", monkeypatch, invalid).status_code == 502
