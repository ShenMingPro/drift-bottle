"""拾取任务 API 的状态映射和幂等审批测试。"""

import asyncio
import json
from types import SimpleNamespace

import app.routers.pick_runs as routes
import pytest
from fastapi import HTTPException
from app.models.pick_run import PickRunRequest, PickRunResumeRequest
from backend_fakes import PickDatabase as FakeDatabase


class FakeGraph:
    def __init__(self):
        self.values = {}
        self.next = ()
        self.resume_count = 0

    async def ainvoke(self, incoming, _config):
        if isinstance(incoming, dict):
            self.values = dict(incoming)
            self.values.update(
                {
                    "status": "running",
                    "match": {
                        "status": "success",
                        "best_match": {"bottle_id": 12, "user_id": 23},
                    },
                    "approved_hints": ["安全建议"],
                    "guard": {"passed": True},
                }
            )
            self.next = ("approval",)
            return {**self.values, "__interrupt__": [{"type": "pick_approval"}]}

        resume = incoming.resume or {}
        self.resume_count += 1
        self.values["status"] = "approved" if resume.get("approved") else "canceled"
        self.values["selected_hint"] = resume.get("hint", "")
        self.next = ()
        return self.values

    async def aget_state(self, _config):
        return SimpleNamespace(values=dict(self.values), next=self.next, tasks=())


def test_start_run_returns_awaiting_approval(monkeypatch):
    graph = FakeGraph()
    monkeypatch.setattr(routes, "get_pick_workflow_app", lambda: graph)

    response = asyncio.run(
        routes.start_pick_run(
            PickRunRequest(userId=7, university="测试大学"),
            current_user_id=7,
        )
    )
    body = json.loads(response.body)

    assert body["status"] == "awaiting_approval"
    assert body["canResume"] is True
    assert body["match"]["best_match"]["bottle_id"] == 12


def test_resume_approval_writes_pick_once(monkeypatch):
    graph = FakeGraph()
    monkeypatch.setattr(routes, "get_pick_workflow_app", lambda: graph)
    database = FakeDatabase()

    async def run():
        started = await routes.start_pick_run(
            PickRunRequest(userId=7),
            current_user_id=7,
        )
        run_id = json.loads(started.body)["runId"]
        resumed = await routes.resume_pick_run(
            run_id,
            PickRunResumeRequest(userId=7, approved=True, hint="安全建议"),
            db=database,
            current_user_id=7,
        )
        repeated = await routes.resume_pick_run(
            run_id,
            PickRunResumeRequest(userId=7, approved=True, hint="安全建议"),
            db=database,
            current_user_id=7,
        )
        return json.loads(resumed.body), json.loads(repeated.body)

    first, second = asyncio.run(run())
    assert first["status"] == "approved"
    assert first["pickRecorded"] is True
    assert second["pickRecorded"] is True
    assert database.chances == 9
    assert len(database.records) == 1
    assert len(database.commits) == 1
    assert graph.resume_count == 1
    assert first["commitStatus"] == second["commitStatus"] == "committed"
    assert any(
        "UPDATE users SET pick_chances" in query for query, _args in database.executions
    )
    assert any(
        "INSERT INTO pick_records" in query and args == [7, 12]
        for query, args in database.executions
    )


def test_unknown_run_is_hidden_from_other_users(monkeypatch):
    graph = FakeGraph()
    graph.values = {"user_id": 8, "status": "approved"}
    monkeypatch.setattr(routes, "get_pick_workflow_app", lambda: graph)

    with pytest.raises(HTTPException) as error:
        asyncio.run(routes.get_pick_run("run-1", current_user_id=7))
    assert error.value.status_code == 404


def test_run_response_never_exposes_unapproved_hints(monkeypatch):
    graph = FakeGraph()
    graph.values = {
        "user_id": 7,
        "status": "approved",
        "approved_hints": ["安全建议"],
        "ice_break": {"hints": ["安全建议", "未通过审核的建议"]},
    }
    monkeypatch.setattr(routes, "get_pick_workflow_app", lambda: graph)

    response = asyncio.run(routes.get_pick_run("run-1", db=FakeDatabase(), current_user_id=7))
    assert json.loads(response.body)["iceBreak"]["hints"] == ["安全建议"]


def test_failed_business_commit_is_pending_and_retry_does_not_repeat_graph(monkeypatch):
    graph = FakeGraph()
    database = FakeDatabase()
    database.fail_next_insert = True
    monkeypatch.setattr(routes, "get_pick_workflow_app", lambda: graph)

    async def run():
        response = await routes.start_pick_run(PickRunRequest(userId=7), current_user_id=7)
        run_id = json.loads(response.body)["runId"]
        request = PickRunResumeRequest(userId=7, approved=True, hint="安全建议")
        with pytest.raises(RuntimeError, match="injected"):
            await routes.resume_pick_run(run_id, request, db=database, current_user_id=7)
        status = await routes.get_pick_run(run_id, db=database, current_user_id=7)
        assert json.loads(status.body)["commitStatus"] == "pending"
        assert json.loads(status.body)["pickRecorded"] is False
        assert database.chances == 10
        assert database.records == {}
        result = await routes.resume_pick_run(run_id, request, db=database, current_user_id=7)
        status = await routes.get_pick_run(run_id, db=database, current_user_id=7)
        assert json.loads(result.body)["commitStatus"] == "committed"
        assert json.loads(status.body)["pickRecorded"] is True

    asyncio.run(run())
    assert graph.resume_count == 1
    assert database.rollbacks == 1
    assert database.chances == 9
    assert len(database.records) == 1


def test_legacy_approved_run_reads_existing_record_without_recharging(monkeypatch):
    graph = FakeGraph()
    graph.values = {"user_id": 7, "status": "approved", "match": {"best_match": {"bottle_id": 12}}}
    database = FakeDatabase()
    database.records[7, 12] = 31
    monkeypatch.setattr(routes, "get_pick_workflow_app", lambda: graph)
    response = asyncio.run(routes.get_pick_run("old-run", db=database, current_user_id=7))
    body = json.loads(response.body)
    assert body["commitStatus"] == "committed"
    assert body["pickRecorded"] is True
    assert database.chances == 10
    assert database.commits == {}


def test_cancel_after_approval_does_not_submit_pending_business(monkeypatch):
    graph = FakeGraph()
    graph.values = {"user_id": 7, "status": "approved", "match": {"best_match": {"bottle_id": 12}}}
    database = FakeDatabase()
    monkeypatch.setattr(routes, "get_pick_workflow_app", lambda: graph)
    response = asyncio.run(routes.resume_pick_run(
        "pending-run", PickRunResumeRequest(userId=7, approved=False), db=database, current_user_id=7
    ))
    assert json.loads(response.body)["commitStatus"] == "pending"
    assert database.chances == 10
    assert database.records == {}
    assert database.commits == {}
