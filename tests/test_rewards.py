"""每日奖励的数据库日期、额度与前置条件。"""

import asyncio
from datetime import date

import pytest
from fastapi import HTTPException

from app.services import rewards


BUSINESS_DATE = date(2026, 10, 25)


class RewardCursor:
    """数据库边界替身：按查询返回行并记录写入。"""

    def __init__(self, *rows):
        self.rows = list(rows)
        self.updates = []

    async def execute(self, query, args=None):
        if query.lstrip().upper().startswith("UPDATE"):
            self.updates.append((query, args))

    async def fetchone(self):
        return self.rows.pop(0) if self.rows else None


def test_checkin_rejects_repeat_on_database_business_date():
    cursor = RewardCursor({
        "last_checkin_date": BUSINESS_DATE,
        "business_date": BUSINESS_DATE,
    })

    with pytest.raises(HTTPException) as error:
        asyncio.run(rewards.claim_checkin(cursor, 7))

    assert error.value.detail == "今日已签到"
    assert cursor.updates == []


def test_invite_resets_previous_database_day_before_granting():
    cursor = RewardCursor({
        "last_invite_date": date(2026, 10, 24),
        "invite_count_today": 3,
        "business_date": BUSINESS_DATE,
    })

    used = asyncio.run(rewards.claim_invite(cursor, 7))

    assert used == 1
    assert cursor.updates[0][1] == [BUSINESS_DATE, 1, 7]


def test_write_bonus_requires_an_existing_write_for_standalone_claim():
    cursor = RewardCursor({
        "last_write_bonus_date": None,
        "write_bonus_used": 0,
        "business_date": BUSINESS_DATE,
    }, {"cnt": 0})

    with pytest.raises(HTTPException) as error:
        asyncio.run(rewards.claim_write_bonus(cursor, 7))

    assert error.value.detail == "请先完成一次写纸团"
    assert cursor.updates == []


@pytest.mark.parametrize("claim", [
    rewards.claim_checkin, rewards.claim_invite, rewards.claim_write_bonus,
])
def test_missing_user_cannot_receive_a_reward(claim):
    cursor = RewardCursor(None)

    with pytest.raises(HTTPException) as error:
        asyncio.run(claim(cursor, 7))

    assert error.value.detail == "用户不存在"
    assert cursor.updates == []


@pytest.mark.parametrize("claim,row,kwargs,expected", [
    (
        rewards.claim_invite,
        {"last_invite_date": BUSINESS_DATE, "invite_count_today": 3},
        {},
        "今日邀请次数已用完",
    ),
    (
        rewards.claim_write_bonus,
        {"last_write_bonus_date": BUSINESS_DATE, "write_bonus_used": 3},
        {"require_write": False},
        "今日写纸团奖励次数已用完",
    ),
])
def test_daily_reward_cap_cannot_be_exceeded(claim, row, kwargs, expected):
    cursor = RewardCursor({**row, "business_date": BUSINESS_DATE})

    with pytest.raises(HTTPException) as error:
        asyncio.run(claim(cursor, 7, **kwargs))

    assert error.value.detail == expected
    assert cursor.updates == []


def test_bonus_for_new_bottle_uses_the_same_daily_quota_without_prior_write():
    cursor = RewardCursor({
        "last_write_bonus_date": BUSINESS_DATE,
        "write_bonus_used": 2,
        "business_date": BUSINESS_DATE,
    })

    used = asyncio.run(rewards.claim_write_bonus(cursor, 7, require_write=False))

    assert used == 3
    assert cursor.updates[0][1] == [BUSINESS_DATE, 3, 7]
    assert "pick_chances = pick_chances + 1" in cursor.updates[0][0]


def test_standalone_bonus_resets_previous_day_after_valid_write():
    cursor = RewardCursor({
        "last_write_bonus_date": date(2026, 10, 24),
        "write_bonus_used": 3,
        "business_date": BUSINESS_DATE,
    }, {"cnt": 1})

    used = asyncio.run(rewards.claim_write_bonus(cursor, 7))

    assert used == 1
    assert cursor.updates[0][1] == [BUSINESS_DATE, 1, 7]


def test_checkin_writes_the_database_business_date_after_rollover():
    cursor = RewardCursor({
        "last_checkin_date": date(2026, 10, 24),
        "business_date": BUSINESS_DATE,
    })

    result = asyncio.run(rewards.claim_checkin(cursor, 7))

    assert result is None
    assert cursor.updates[0][1] == [BUSINESS_DATE, 7]
