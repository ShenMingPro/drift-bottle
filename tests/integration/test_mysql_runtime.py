"""显式启用的真实 MySQL 事务、并发及回滚测试。

设置 RUN_MYSQL_INTEGRATION=1，并指定独立的专用测试库。
仅允许 paper_bottle_backend_test_<后缀> 或 CI 专用 paper_bottle_test_it。
测试会重新创建并最终删除该测试库，不能指向业务数据库。
"""

from __future__ import annotations

import asyncio
import os
import re
from contextlib import asynccontextmanager
from typing import Awaitable, Callable

import aiomysql
import pytest
from fastapi import HTTPException

import app.database as database
import app.routers.bottle as bottle_routes
import app.routers.task as task
from app.ai_content_proof import create_ai_content_proof
from app.config import settings
from app.models.bottle import WriteRequest
from app.services.chances import consume_chance, grant_chances, refresh_cooldown_chances
from app.services.picks import commit_pick

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_MYSQL_INTEGRATION") != "1",
        reason="需要设置 RUN_MYSQL_INTEGRATION=1 才运行真实 MySQL 测试",
    ),
]


def _sync_settings_from_env() -> None:
    settings.DB_HOST = os.environ.get("DB_HOST", settings.DB_HOST)
    settings.DB_PORT = int(os.environ.get("DB_PORT", settings.DB_PORT))
    settings.DB_USER = os.environ.get("DB_USER", settings.DB_USER)
    settings.DB_PASSWORD = os.environ.get("DB_PASSWORD", settings.DB_PASSWORD)
    settings.DB_NAME = os.environ.get("DB_NAME", settings.DB_NAME)


def _checked_test_database_name() -> str:
    name = settings.DB_NAME
    if name != "paper_bottle_test_it" and not re.fullmatch(
        r"paper_bottle_backend_test_[A-Za-z0-9_]+", name
    ):
        raise RuntimeError("只允许使用明确的独立 MySQL 集成测试库名")
    if len(name) > 64:
        raise RuntimeError("MySQL 集成测试库名不能超过 64 字符")
    return name


async def _drop_test_database() -> None:
    name = _checked_test_database_name()
    connection = await aiomysql.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        charset="utf8mb4",
        autocommit=True,
    )
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = %s",
                [name],
            )
            if await cursor.fetchone() is not None:
                # name 已通过严格标识符白名单，无动态 SQL 注入。
                await cursor.execute(f"DROP DATABASE {name}")
    finally:
        # Connection.close() 是同步方法；只有 Pool 才提供 wait_closed()。
        connection.close()


async def _prepare_database() -> None:
    _sync_settings_from_env()
    _checked_test_database_name()
    os.environ["DB_NAME"] = settings.DB_NAME
    await _drop_test_database()
    from init_db import init_db

    await init_db()
    await database.init_pool()


async def _teardown_database() -> None:
    await database.close_pool()
    await _drop_test_database()


def run_case(case: Callable[[], Awaitable[None]]) -> None:
    async def main() -> None:
        await _prepare_database()
        try:
            await case()
        finally:
            await _teardown_database()

    asyncio.run(main())


async def _insert_user(cursor, username: str, *, write: int = 1, pick: int = 1) -> int:
    await cursor.execute(
        "INSERT INTO users (username, password, nickname, write_chances, pick_chances) "
        "VALUES (%s, %s, %s, %s, %s)",
        [username, "synthetic-test-password-hash", username, write, pick],
    )
    return int(cursor.lastrowid)


async def _insert_bottle(cursor, user_id: int, content: str = "测试纸团") -> int:
    await cursor.execute(
        "INSERT INTO bottles (user_id, content) VALUES (%s, %s)", [user_id, content]
    )
    return int(cursor.lastrowid)


async def _user_row(db: database.Database, user_id: int) -> dict:
    async with db.session() as cursor:
        await cursor.execute(
            "SELECT write_chances, pick_chances, last_checkin_date, last_invite_date, "
            "invite_count_today, write_bonus_used, last_write_bonus_date "
            "FROM users WHERE id = %s",
            [user_id],
        )
        return await cursor.fetchone()


async def _count(db: database.Database, table: str, user_id: int) -> int:
    if table not in {"bottles", "pick_records", "pick_commits"}:
        raise ValueError("非法测试表名")
    async with db.session() as cursor:
        await cursor.execute(f"SELECT COUNT(*) AS cnt FROM {table} WHERE user_id = %s", [user_id])
        return int((await cursor.fetchone())["cnt"])


class InjectedWriteFailure(RuntimeError):
    """在真实事务内部模拟数据库写入前故障。"""


class _FailingCursor:
    def __init__(self, cursor, owner):
        self._cursor = cursor
        self._owner = owner

    def __getattr__(self, name):
        return getattr(self._cursor, name)

    async def execute(self, query, args=None):
        self._owner.executions.append(query)
        if query.lstrip().startswith(self._owner.insert_prefix):
            self._owner.injected = True
            raise InjectedWriteFailure("模拟业务写入失败")
        return await self._cursor.execute(query, args)


class _FailingDatabase:
    """只代理游标；连接、begin/rollback/commit 仍使用真实 Database。"""

    def __init__(self, wrapped: database.Database, insert_prefix: str):
        self.wrapped = wrapped
        self.insert_prefix = insert_prefix
        self.injected = False
        self.executions = []

    @asynccontextmanager
    async def session(self, *, transaction: bool = False):
        async with self.wrapped.session(transaction=transaction) as cursor:
            yield _FailingCursor(cursor, self)


def test_chance_service_against_mysql():
    async def case():
        db = database.get_database()
        async with db.session() as cursor:
            user_id = await _insert_user(cursor, "chance_user", write=1, pick=1)
            await consume_chance(cursor, user_id, "write_chances")
        assert (await _user_row(db, user_id))["write_chances"] == 0

        async with db.session() as cursor:
            with pytest.raises(HTTPException):
                await consume_chance(cursor, user_id, "write_chances")
            await grant_chances(cursor, user_id, write_delta=1, pick_delta=2)
        row = await _user_row(db, user_id)
        assert (row["write_chances"], row["pick_chances"]) == (1, 3)

        async with db.session() as cursor:
            await cursor.execute("UPDATE users SET write_chances = 0 WHERE id = %s", [user_id])
            await cursor.execute(
                "INSERT INTO bottles (user_id, content, created_at) "
                "VALUES (%s, %s, NOW() - INTERVAL 4 HOUR)",
                [user_id, "四小时前的测试纸团"],
            )
            await refresh_cooldown_chances(cursor, user_id)
        assert (await _user_row(db, user_id))["write_chances"] == 1

    run_case(case)


def test_task_routes_update_counts_against_mysql():
    async def case():
        db = database.get_database()
        async with db.session() as cursor:
            user_id = await _insert_user(cursor, "task_user", write=1, pick=1)
            await _insert_bottle(cursor, user_id)
        await task.checkin({"userId": user_id}, db=db, current_user_id=user_id)
        row = await _user_row(db, user_id)
        assert (row["write_chances"], row["pick_chances"]) == (2, 2)
        await task.invite({"userId": user_id}, db=db, current_user_id=user_id)
        assert (await _user_row(db, user_id))["pick_chances"] == 3
        await task.write_bonus({"userId": user_id}, db=db, current_user_id=user_id)
        assert (await _user_row(db, user_id))["pick_chances"] == 4

    run_case(case)


def test_concurrent_pick_request_commits_once_against_mysql():
    async def case():
        db = database.get_database()
        async with db.session() as cursor:
            picker = await _insert_user(cursor, "concurrent_picker", pick=3)
            author = await _insert_user(cursor, "concurrent_author")
            bottle = await _insert_bottle(cursor, author)
        results = await asyncio.wait_for(
            asyncio.gather(
                *(commit_pick(db, picker, bottle, request_key="run:concurrent") for _ in range(20))
            ),
            timeout=30,
        )
        assert sum(result.created for result in results) == 1
        assert len({result.record_id for result in results}) == 1
        assert (await _user_row(db, picker))["pick_chances"] == 2
        assert await _count(db, "pick_records", picker) == 1
        assert await _count(db, "pick_commits", picker) == 1

    run_case(case)


def test_pick_write_failure_rolls_back_then_retry_commits_against_mysql():
    async def case():
        db = database.get_database()
        async with db.session() as cursor:
            picker = await _insert_user(cursor, "rollback_picker", pick=2)
            author = await _insert_user(cursor, "rollback_author")
            bottle = await _insert_bottle(cursor, author)
        failing = _FailingDatabase(db, "INSERT INTO pick_records")
        with pytest.raises(InjectedWriteFailure):
            await commit_pick(failing, picker, bottle, request_key="run:rollback")
        assert failing.injected
        assert any("UPDATE users SET pick_chances" in query for query in failing.executions)
        assert (await _user_row(db, picker))["pick_chances"] == 2
        assert await _count(db, "pick_records", picker) == 0
        assert await _count(db, "pick_commits", picker) == 0

        committed = await commit_pick(db, picker, bottle, request_key="run:rollback")
        assert committed.created
        assert (await _user_row(db, picker))["pick_chances"] == 1
        assert await _count(db, "pick_records", picker) == 1
        assert await _count(db, "pick_commits", picker) == 1

    run_case(case)


def test_cross_run_pick_rules_against_mysql():
    async def case():
        db = database.get_database()
        async with db.session() as cursor:
            picker = await _insert_user(cursor, "cross_run_picker", pick=3)
            author = await _insert_user(cursor, "cross_run_author")
            other_author = await _insert_user(cursor, "other_author")
            bottle = await _insert_bottle(cursor, author, "作者一的第一条")
            same_author_bottle = await _insert_bottle(cursor, author, "作者一的第二条")
            other_bottle = await _insert_bottle(cursor, other_author)

        first = await commit_pick(db, picker, bottle, request_key="run:first")
        repeated = await commit_pick(db, picker, bottle, request_key="run:second")
        assert first.created and not repeated.created
        assert first.record_id == repeated.record_id
        assert (await _user_row(db, picker))["pick_chances"] == 2

        with pytest.raises(HTTPException) as error:
            await commit_pick(db, picker, same_author_bottle, request_key="run:stale")
        assert error.value.status_code == 409
        assert (await _user_row(db, picker))["pick_chances"] == 2

        other = await commit_pick(db, picker, other_bottle, request_key="run:other")
        assert other.created
        assert (await _user_row(db, picker))["pick_chances"] == 1
        assert await _count(db, "pick_records", picker) == 2
        assert await _count(db, "pick_commits", picker) == 3

    run_case(case)


def test_concurrent_distinct_runs_share_existing_bottle_against_mysql():
    async def case():
        db = database.get_database()
        async with db.session() as cursor:
            picker = await _insert_user(cursor, "distinct_run_picker", pick=3)
            author = await _insert_user(cursor, "distinct_run_author")
            bottle = await _insert_bottle(cursor, author)
        results = await asyncio.wait_for(
            asyncio.gather(
                *(commit_pick(db, picker, bottle, request_key=f"run:{index}") for index in range(20))
            ),
            timeout=30,
        )
        assert sum(result.created for result in results) == 1
        assert len({result.record_id for result in results}) == 1
        assert (await _user_row(db, picker))["pick_chances"] == 2
        assert await _count(db, "pick_records", picker) == 1
        assert await _count(db, "pick_commits", picker) == 20

    run_case(case)


def test_reusing_request_key_for_different_bottle_is_rejected_against_mysql():
    async def case():
        db = database.get_database()
        async with db.session() as cursor:
            picker = await _insert_user(cursor, "request_key_picker", pick=2)
            author = await _insert_user(cursor, "request_key_author")
            other_author = await _insert_user(cursor, "request_key_other_author")
            bottle = await _insert_bottle(cursor, author)
            other_bottle = await _insert_bottle(cursor, other_author)
        await commit_pick(db, picker, bottle, request_key="run:same-key")
        with pytest.raises(HTTPException) as error:
            await commit_pick(db, picker, other_bottle, request_key="run:same-key")
        assert error.value.status_code == 409
        assert (await _user_row(db, picker))["pick_chances"] == 1
        assert await _count(db, "pick_records", picker) == 1
        assert await _count(db, "pick_commits", picker) == 1

    run_case(case)


@pytest.mark.parametrize(
    ("action", "expected_successes", "expected_write", "expected_pick", "counter"),
    [
        ("checkin", 1, 1, 1, None),
        ("invite", 3, 0, 3, "invite_count_today"),
        ("write_bonus", 3, 0, 3, "write_bonus_used"),
    ],
)
def test_concurrent_rewards_respect_daily_limits_against_mysql(
    action, expected_successes, expected_write, expected_pick, counter
):
    async def case():
        db = database.get_database()
        async with db.session() as cursor:
            user_id = await _insert_user(cursor, f"concurrent_{action}", write=0, pick=0)
            if action == "write_bonus":
                await _insert_bottle(cursor, user_id)

        async def claim():
            try:
                return await getattr(task, action)(
                    {"userId": user_id}, db=db, current_user_id=user_id
                )
            except HTTPException as error:
                assert error.status_code == 400
                return None

        results = await asyncio.wait_for(asyncio.gather(*(claim() for _ in range(20))), timeout=30)
        assert sum(result is not None for result in results) == expected_successes
        row = await _user_row(db, user_id)
        assert row["write_chances"] == expected_write
        assert row["pick_chances"] == expected_pick
        if counter:
            assert row[counter] == expected_successes
        else:
            assert row["last_checkin_date"] is not None

    run_case(case)


def test_daily_rewards_reset_using_database_business_date_against_mysql():
    async def case():
        db = database.get_database()
        async with db.session() as cursor:
            user_id = await _insert_user(cursor, "cross_day_rewards", write=0, pick=0)
            await _insert_bottle(cursor, user_id)
            await cursor.execute(
                "UPDATE users SET last_checkin_date = CURDATE() - INTERVAL 1 DAY, "
                "last_invite_date = CURDATE() - INTERVAL 1 DAY, invite_count_today = 3, "
                "last_write_bonus_date = CURDATE() - INTERVAL 1 DAY, write_bonus_used = 3 "
                "WHERE id = %s",
                [user_id],
            )
            await cursor.execute("SELECT CURDATE() AS business_date")
            today = (await cursor.fetchone())["business_date"]

        await task.checkin({"userId": user_id}, db=db, current_user_id=user_id)
        invitation = await task.invite({"userId": user_id}, db=db, current_user_id=user_id)
        bonus = await task.write_bonus({"userId": user_id}, db=db, current_user_id=user_id)
        row = await _user_row(db, user_id)
        assert (row["write_chances"], row["pick_chances"]) == (1, 3)
        assert row["invite_count_today"] == row["write_bonus_used"] == 1
        assert row["last_checkin_date"] == row["last_invite_date"] == today
        assert row["last_write_bonus_date"] == today
        assert invitation.inviteRemaining == bonus.writeBonusRemaining == 2
        status = await task.get_checkin_status(userId=user_id, db=db, current_user_id=user_id)
        assert not status.canCheckin
        assert status.inviteUsed == status.writeBonusUsed == 1
        assert status.inviteRemaining == status.writeBonusRemaining == 2

    run_case(case)


@pytest.mark.parametrize("use_bonus", [False, True])
def test_bottle_insert_failure_rolls_back_quota_and_rewards_against_mysql(use_bonus):
    async def case():
        db = database.get_database()
        async with db.session() as cursor:
            user_id = await _insert_user(cursor, "failed_bottle_writer", write=2, pick=1)
        content = "集成测试的安全 AI 纸团"
        request = WriteRequest(
            userId=user_id,
            content=content,
            aiProof=create_ai_content_proof(user_id, content),
            useBonus=use_bonus,
        )
        failing = _FailingDatabase(db, "INSERT INTO bottles")
        with pytest.raises(InjectedWriteFailure):
            await bottle_routes.write_bottle(request, db=failing, current_user_id=user_id)
        assert failing.injected
        row = await _user_row(db, user_id)
        assert (row["write_chances"], row["pick_chances"]) == (2, 1)
        assert row["write_bonus_used"] == 0
        assert row["last_write_bonus_date"] is None
        assert await _count(db, "bottles", user_id) == 0

        response = await bottle_routes.write_bottle(request, db=db, current_user_id=user_id)
        assert response.success
        row = await _user_row(db, user_id)
        assert await _count(db, "bottles", user_id) == 1
        if use_bonus:
            assert (row["write_chances"], row["pick_chances"]) == (2, 2)
            assert row["write_bonus_used"] == 1
            assert row["last_write_bonus_date"] is not None
        else:
            assert (row["write_chances"], row["pick_chances"]) == (1, 1)
            assert row["write_bonus_used"] == 0

    run_case(case)
