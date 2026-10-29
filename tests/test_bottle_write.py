import asyncio
from contextlib import asynccontextmanager
from datetime import date

import pytest
from fastapi import HTTPException

import app.routers.bottle as bottle_router
from app.ai_content_proof import create_ai_content_proof
from app.models.bottle import WriteRequest


class FakeWriteDatabase:
    def __init__(self):
        self.executions = []
        self.rowcount = 1
        self._next_row = None

    @asynccontextmanager
    async def session(self, *, transaction=False):
        yield self

    async def execute(self, query, args):
        self.executions.append((query, args))

    async def fetchone(self):
        if self._next_row is not None:
            self._next_row.setdefault("business_date", date(2026, 10, 25))
        return self._next_row


def test_exact_ai_suggestion_skips_supervisor(monkeypatch):
    content = "今晚有人一起跑步吗"
    proof = create_ai_content_proof(7, content)
    database = FakeWriteDatabase()

    async def supervisor_must_not_run(_state):
        raise AssertionError("未修改的 AI 内容不应调用 Supervisor/Guard Agent")

    monkeypatch.setattr(bottle_router.supervisor_app, "ainvoke", supervisor_must_not_run)

    response = asyncio.run(
        bottle_router.write_bottle(
            WriteRequest(userId=7, content=content, aiProof=proof),
            db=database,
            current_user_id=7,
        )
    )

    assert response.success is True
    assert any(
        "UPDATE users SET write_chances" in query and args == [7]
        for query, args in database.executions
    )
    assert any(
        "INSERT INTO bottles" in query and args == [7, content]
        for query, args in database.executions
    )


@pytest.mark.parametrize(
    ("content", "proof"),
    [
        ("原始 AI 内容 + 用户添加", create_ai_content_proof(7, "原始 AI 内容")),
        ("纯手写纸团", None),
        ("原始 AI 内容", create_ai_content_proof(8, "原始 AI 内容")),
    ],
    ids=["modified", "manual", "wrong-user"],
)
def test_modified_manual_or_wrong_user_content_runs_guard(monkeypatch, content, proof):
    database = FakeWriteDatabase()
    states = []

    async def record_supervisor(state):
        states.append(state)
        return {"guard_passed": True, "guard_reason": ""}

    monkeypatch.setattr(bottle_router.supervisor_app, "ainvoke", record_supervisor)

    asyncio.run(
        bottle_router.write_bottle(
            WriteRequest(userId=7, content=content, aiProof=proof),
            db=database,
            current_user_id=7,
        )
    )

    assert len(states) == 1
    assert states[0]["action"] == "write"
    assert states[0]["content"] == content


def test_invalid_proof_falls_back_to_fail_closed_guard(monkeypatch):
    database = FakeWriteDatabase()

    async def reject_content(_state):
        return {"guard_passed": False, "guard_reason": "内容审核未通过"}

    monkeypatch.setattr(bottle_router.supervisor_app, "ainvoke", reject_content)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            bottle_router.write_bottle(
                WriteRequest(userId=7, content="被修改的内容", aiProof="invalid-proof"),
                db=database,
                current_user_id=7,
            )
        )

    assert exc_info.value.status_code == 400
    assert database.executions == []


def test_bonus_write_does_not_consume_regular_write_chance(monkeypatch):
    database = FakeWriteDatabase()
    database._next_row = {"write_bonus_used": 0, "last_write_bonus_date": None}

    async def allow(_state):
        return {"guard_passed": True, "guard_reason": ""}

    monkeypatch.setattr(bottle_router.supervisor_app, "ainvoke", allow)

    response = asyncio.run(
        bottle_router.write_bottle(
            WriteRequest(userId=7, content="手写奖励纸团", useBonus=True),
            db=database,
            current_user_id=7,
        )
    )

    assert response.writeBonusUsed == 1
    assert response.writeBonusRemaining == 2
    assert not any(
        "write_chances = write_chances - 1" in query
        for query, _args in database.executions
    )
    assert any(
        "write_bonus_used" in query and "pick_chances = pick_chances + 1" in query
        for query, _args in database.executions
    )
