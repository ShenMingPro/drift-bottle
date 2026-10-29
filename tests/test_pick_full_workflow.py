import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import app.routers.bottle as bottle_router
from backend_fakes import PickDatabase as FakeDatabase


def test_pick_full_endpoint_runs_supervisor_workflow(monkeypatch):
    calls = []

    async def fake_supervisor(state, config=None):
        calls.append((state, config))
        final_result = {
            "action": "pick_full",
            "result": {
                "match": {
                    "agent": "match",
                    "status": "success",
                    "best_match": {
                        "bottle_id": 12,
                        "user_id": 23,
                        "content": "喜欢摄影",
                    },
                },
                "profile_analysis": {"agent": "profile_analysis", "status": "success"},
                "ice_break": {"agent": "ice_break", "status": "success", "hints": ["你也喜欢摄影吗？"]},
                "guard": {"agent": "guard", "status": "success", "passed": True},
            },
        }
        return {
            "final_result": json.dumps(final_result, ensure_ascii=False),
            "status": "success",
            "guard_passed": True,
            "completed_steps": ["match", "profile_analysis", "ice_break", "guard"],
            "step_count": 4,
        }

    monkeypatch.setattr(bottle_router, "supervisor_app", SimpleNamespace(ainvoke=fake_supervisor))
    database = FakeDatabase()

    response = asyncio.run(
        bottle_router.pick_full_bottle(
            userId=7,
            university="测试大学",
            city="测试市",
            province="测试省",
            db=database,
            current_user_id=7,
        )
    )

    body = json.loads(response.body)
    assert calls[0][0]["action"] == "pick_full"
    assert calls[0][0]["target_university"] == "测试大学"
    assert calls[0][1]["configurable"]["thread_id"].startswith("pick-full:7:")
    assert body["hasMatch"] is True
    assert body["iceBreak"]["hints"] == ["你也喜欢摄影吗？"]
    assert any(
        "UPDATE users SET pick_chances" in query and args == [7]
        for query, args in database.executions
    )
    assert any(
        "INSERT INTO pick_records" in query and args == [7, 12]
        for query, args in database.executions
    )


def test_pick_full_endpoint_does_not_commit_unapproved_ice_break(monkeypatch):
    async def rejected_supervisor(_state, _config=None):
        return {
            "final_result": json.dumps(
                {
                    "action": "pick_full",
                    "result": {
                        "match": {"best_match": {"bottle_id": 12, "user_id": 23}},
                        "ice_break": {"hints": ["不安全内容"]},
                    },
                },
                ensure_ascii=False,
            ),
            "status": "success",
            "guard_passed": False,
            "guard_reason": "内容审核未通过",
            "completed_steps": ["match", "ice_break", "guard"],
            "step_count": 3,
        }

    monkeypatch.setattr(
        bottle_router,
        "supervisor_app",
        SimpleNamespace(ainvoke=rejected_supervisor),
    )
    database = FakeDatabase()

    response = asyncio.run(
        bottle_router.pick_full_bottle(
            userId=7,
            university="测试大学",
            city="测试市",
            province="测试省",
            db=database,
            current_user_id=7,
        )
    )

    assert response.status_code == 422
    assert json.loads(response.body)["status"] == "rejected"
    assert database.executions == []


def test_pick_full_endpoint_does_not_commit_when_workflow_fails(monkeypatch):
    async def failed_supervisor(_state, _config=None):
        return {
            "final_result": json.dumps(
                {
                    "action": "pick_full",
                    "result": {
                        "match": {
                            "status": "success",
                            "best_match": {"bottle_id": 12, "user_id": 23},
                        },
                        "profile_analysis": {
                            "status": "fatal_error",
                            "error": "画像服务不可用",
                        },
                    },
                },
                ensure_ascii=False,
            ),
            "status": "failed",
            "guard_passed": False,
            "guard_reason": "画像服务不可用",
            "completed_steps": ["match"],
            "step_count": 2,
        }

    monkeypatch.setattr(
        bottle_router,
        "supervisor_app",
        SimpleNamespace(ainvoke=failed_supervisor),
    )
    database = FakeDatabase()

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            bottle_router.pick_full_bottle(
                userId=7,
                university="测试大学",
                city="测试市",
                province="测试省",
                db=database,
                current_user_id=7,
            )
        )

    assert error.value.status_code == 502
    assert database.executions == []


def test_legacy_pick_route_is_restored():
    assert hasattr(bottle_router, "pick_bottle")
