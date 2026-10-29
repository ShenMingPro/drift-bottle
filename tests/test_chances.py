"""服务端次数规则回归测试。"""

import asyncio
from contextlib import asynccontextmanager
from datetime import date

import pytest
from fastapi import HTTPException

import app.routers.task as task
from app.models.task import UpdateChancesRequest


class FakeDatabase:
    def __init__(self, rows=None):
        self.rows = list(rows or [])
        self.executions = []
        self.sessions = []

    @asynccontextmanager
    async def session(self, *, transaction=False):
        self.sessions.append((transaction, "begin"))
        try:
            yield self
        except BaseException:
            self.sessions.append((transaction, "rollback"))
            raise
        else:
            self.sessions.append((transaction, "commit"))

    async def execute(self, query, args):
        self.executions.append((query, args))

    async def fetchone(self):
        return self.rows.pop(0) if self.rows else None


def test_update_chances_is_disabled():
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            task.update_chances(
                UpdateChancesRequest(userId=7, writeChances=999, pickChances=999),
                db=FakeDatabase(),
                current_user_id=7,
            )
        )
    assert error.value.status_code == 410


def test_checkin_grants_both_chances_server_side():
    db = FakeDatabase([{"last_checkin_date": None, "business_date": date(2099, 1, 1)}])

    result = asyncio.run(
        task.checkin({"userId": 7}, db=db, current_user_id=7)
    )

    assert result.success is True
    assert db.sessions == [(True, "begin"), (True, "commit")]
    assert any(
        "write_chances = write_chances + 1" in query
        and "pick_chances = pick_chances + 1" in query
        for query, _args in db.executions
    )


def test_invite_grants_pick_chance_server_side():
    db = FakeDatabase([{
        "last_invite_date": None, "invite_count_today": 0,
        "business_date": date(2099, 1, 1),
    }])

    result = asyncio.run(
        task.invite({"userId": 7}, db=db, current_user_id=7)
    )

    assert result.inviteRemaining == 2
    assert db.sessions == [(True, "begin"), (True, "commit")]
    assert any(
        "pick_chances = pick_chances + 1" in query
        for query, _args in db.executions
    )


def test_write_bonus_requires_a_write_today():
    db = FakeDatabase([{
        "last_write_bonus_date": None, "write_bonus_used": 0,
        "business_date": date(2099, 1, 1),
    }, {"cnt": 0}])

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            task.write_bonus({"userId": 7}, db=db, current_user_id=7)
        )

    assert error.value.status_code == 400
    assert "请先完成一次写纸团" in error.value.detail
    assert db.sessions == [(True, "begin"), (True, "rollback")]


def test_user_chances_refreshes_cooldown(monkeypatch):
    db = FakeDatabase([{"write_chances": 1, "pick_chances": 1}])
    calls = []

    async def refresh(_db, user_id):
        calls.append(user_id)

    monkeypatch.setattr(task, "refresh_cooldown_chances", refresh)
    result = asyncio.run(
        task.get_user_chances(userId=7, db=db, current_user_id=7)
    )

    assert calls == [7]
    assert result.writeChances == 1
    assert db.sessions == [(False, "begin"), (False, "commit")]


def test_checkin_status_uses_database_date_for_all_daily_quotas():
    today = date(2099, 1, 1)
    db = FakeDatabase([{
        "last_checkin_date": today,
        "last_invite_date": today,
        "invite_count_today": 2,
        "last_write_bonus_date": date(2098, 12, 31),
        "write_bonus_used": 3,
        "business_date": today,
    }])

    result = asyncio.run(task.get_checkin_status(userId=7, db=db, current_user_id=7))

    assert result.canCheckin is False
    assert result.inviteUsed == 2
    assert result.inviteRemaining == 1
    assert result.writeBonusUsed == 0
    assert result.writeBonusRemaining == 3
    assert db.sessions == [(False, "begin"), (False, "commit")]
