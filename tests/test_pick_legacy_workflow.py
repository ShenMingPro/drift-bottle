import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage

import app.agents.match_agent as match_agent
import app.agents.supervisor_agent as supervisor
import app.routers.bottle as bottle_router
from backend_fakes import PickDatabase as FakeDatabase


def _run(coro):
    return asyncio.run(coro)


def test_legacy_pick_returns_flat_response_and_records_bottle(monkeypatch):
    captured = []
    payload = {
        "best_match": {
            "bottle_id": 12,
            "user_id": 23,
            "content": "最近喜欢摄影",
            "created_at": "2026-10-06 10:30:00",
            "nickname": "小林",
            "gender": "女",
            "age": 21,
            "hobbies": "摄影、旅行",
            "contact_type": "微信",
            "contact_value": "xiaolin",
            "province": "测试省",
            "city": "测试市",
            "university": "测试大学",
        },
        "match_reason": "匹配到同校用户",
        "match_score": 0.86,
        "match_level": "university",
    }

    async def fake_supervisor(state):
        captured.append(state)
        return {
            "final_result": json.dumps(
                {"action": "pick", "result": payload}, ensure_ascii=False
            ),
            "guard_passed": True,
        }

    monkeypatch.setattr(
        bottle_router, "supervisor_app", SimpleNamespace(ainvoke=fake_supervisor)
    )
    db = FakeDatabase()

    response = _run(
        bottle_router.pick_bottle(
            userId=7,
            university="测试大学",
            city="测试市",
            province="测试省",
            db=db,
            current_user_id=7,
        )
    )

    assert response.hasBottle is True
    assert response.bottleUserId == 23
    assert response.content == "最近喜欢摄影"
    assert response.createdAt == "2026-10-06 10:30:00"
    assert response.nickname == "小林"
    assert response.matchLevel == "university"
    assert response.matchReason == "匹配到同校用户"
    assert response.matchScore == 0.86
    assert captured[0]["action"] == "pick"
    assert captured[0]["target_university"] == "测试大学"
    assert captured[0]["target_city"] == "测试市"
    assert captured[0]["target_province"] == "测试省"
    assert captured[0]["guard_passed"] is True
    assert any(
        "UPDATE users SET pick_chances" in query and args == [7]
        for query, args in db.executions
    )
    assert any(
        "INSERT INTO pick_records" in query and args == [7, 12]
        for query, args in db.executions
    )


def test_legacy_pick_without_candidate_does_not_write_record(monkeypatch):
    async def fake_supervisor(_state):
        return {
            "final_result": json.dumps(
                {
                    "action": "pick",
                    "result": {
                        "best_match": None,
                        "match_reason": "暂时没有纸团",
                        "match_score": 0,
                        "match_level": "",
                    },
                },
                ensure_ascii=False,
            ),
            "guard_passed": True,
        }

    monkeypatch.setattr(
        bottle_router, "supervisor_app", SimpleNamespace(ainvoke=fake_supervisor)
    )
    db = FakeDatabase()

    response = _run(
        bottle_router.pick_bottle(
            userId=7,
            university="测试大学",
            city="测试市",
            province="测试省",
            db=db,
            current_user_id=7,
        )
    )

    assert response.hasBottle is False
    assert db.executions == []


def test_legacy_pick_rejects_user_id_mismatch_before_workflow(monkeypatch):
    calls = []

    async def must_not_run(_state):
        calls.append(True)
        raise AssertionError("身份不匹配时不能调用 Supervisor")

    monkeypatch.setattr(
        bottle_router, "supervisor_app", SimpleNamespace(ainvoke=must_not_run)
    )
    db = FakeDatabase()

    with pytest.raises(HTTPException) as error:
        _run(
            bottle_router.pick_bottle(
                userId=7,
                university="测试大学",
                city="测试市",
                province="测试省",
                db=db,
                current_user_id=8,
            )
        )

    assert error.value.status_code == 403
    assert calls == []
    assert db.executions == []


class _FakeTool:
    def __init__(self, callback):
        self.func = callback
        self.coroutine = None

    def invoke(self, arguments):
        return self.func(arguments)


def test_supervisor_pick_aggregates_match_directly_and_only_allows_match(monkeypatch):
    calls = []
    payload = {
        "agent": "match",
        "status": "success",
        "result": "success",
        "best_match": {"bottle_id": 12, "user_id": 23},
        "match_reason": "匹配到同校用户",
        "match_score": 0.86,
        "match_level": "university",
    }

    def fake_match(arguments):
        calls.append(arguments)
        return json.dumps(payload, ensure_ascii=False)

    monkeypatch.setattr(
        supervisor,
        "supervisor_tool_map",
        {"call_match_agent": _FakeTool(fake_match)},
    )
    state = supervisor.create_supervisor_state(
        user_id=7,
        action="pick",
        university="测试大学",
        city="测试市",
        province="测试省",
    )
    state["messages"] = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "call_match_agent",
                    "args": {},
                    "id": "match-call",
                    "type": "tool_call",
                }
            ],
        )
    ]

    updates = _run(supervisor.execute_tools(state))
    merged = supervisor._aggregate_artifacts({**state, **updates})

    assert calls == [
        {
            "user_id": 7,
            "university": "测试大学",
            "city": "测试市",
            "province": "测试省",
            "reuse_profile": False,
        }
    ]
    assert updates["artifacts"]["match"] == payload
    assert merged["result"] == payload
    assert supervisor.ACTION_TOOL_ALLOWLIST["pick"] == {"call_match_agent"}


def test_legacy_match_execution_does_not_inject_profile_into_find_best_match(monkeypatch):
    captured_match_args = []

    async def fake_profile(_args):
        return json.dumps(
            {
                "user_id": 7,
                "nickname": "小明",
                "gender": "男",
                "age": 20,
                "hobbies": "摄影、旅行",
                "province": "测试省",
                "city": "测试市",
                "university": "测试大学",
            },
            ensure_ascii=False,
        )

    async def fake_match(args):
        captured_match_args.append(args)
        return json.dumps(
            {
                "result": "no_candidates",
                "best_match": None,
                "match_reason": "",
                "match_score": 0,
                "match_level": "",
            },
            ensure_ascii=False,
        )

    class AsyncFakeTool:
        def __init__(self, callback):
            self.coroutine = callback
            self.func = None

        async def ainvoke(self, args):
            return await self.coroutine(args)

    monkeypatch.setattr(
        match_agent,
        "match_tool_map",
        {
            "query_user_profile": AsyncFakeTool(fake_profile),
            "find_best_match": AsyncFakeTool(fake_match),
        },
    )

    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "query_user_profile",
                "args": {"user_id": 7},
                "id": "profile-call",
                "type": "tool_call",
            },
            {
                "name": "find_best_match",
                "args": {
                    "user_id": 7,
                    "university": "测试大学",
                    "city": "测试市",
                    "province": "测试省",
                },
                "id": "match-call",
                "type": "tool_call",
            },
        ],
    )

    _run(
        match_agent.execute_tools(
            {
                "user_id": 7,
                "messages": [message],
                "user_profile": None,
                "reuse_profile": False,
            }
        )
    )

    assert len(captured_match_args) == 1
    assert "user_profile" not in captured_match_args[0]

