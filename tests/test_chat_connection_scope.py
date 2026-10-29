"""聊天请求在等待模型时释放连接，并以短事务写入消息。"""

import asyncio
from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi import FastAPI

import app.database as database
import app.routers.chat as chat


class FakePool:
    """在数据库边界模拟有限连接与提交/回滚，不替换 Database。"""

    def __init__(self, capacity=2):
        self.capacity = capacity
        self.slots = asyncio.Semaphore(capacity)
        self.in_use = 0
        self.next_message_id = 1
        self.messages = {}
        self.conversation_updates = []
        self.participants = {"user1_id": 7, "user2_id": 8}
        self.fail_conversation_update = False
        self.commits = 0
        self.rollbacks = 0

    @asynccontextmanager
    async def acquire(self):
        await self.slots.acquire()
        self.in_use += 1
        try:
            yield FakeConnection(self)
        finally:
            self.in_use -= 1
            self.slots.release()


class FakeConnection:
    def __init__(self, pool):
        self.pool = pool
        self.in_transaction = False
        self.pending_messages = {}
        self.pending_conversation_updates = []

    @asynccontextmanager
    async def cursor(self, *_args):
        yield FakeCursor(self)

    async def begin(self):
        self.in_transaction = True

    async def commit(self):
        self.pool.messages.update(self.pending_messages)
        self.pool.conversation_updates.extend(self.pending_conversation_updates)
        self.pending_messages.clear()
        self.pending_conversation_updates.clear()
        self.in_transaction = False
        self.pool.commits += 1

    async def rollback(self):
        self.pending_messages.clear()
        self.pending_conversation_updates.clear()
        self.in_transaction = False
        self.pool.rollbacks += 1


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.lastrowid = None
        self.row = None

    async def execute(self, query, args):
        pool = self.connection.pool
        statement = " ".join(query.split())
        if statement.startswith("SELECT user1_id, user2_id FROM conversations"):
            self.row = dict(pool.participants)
        elif statement.startswith("INSERT INTO messages"):
            self.lastrowid = pool.next_message_id
            pool.next_message_id += 1
            row = {
                "id": self.lastrowid,
                "conversation_id": args[0],
                "sender_id": args[1],
                "content": args[2],
                "is_read": 0,
                "created_at": "2026-10-25 12:00:00",
            }
            target = self.connection.pending_messages if self.connection.in_transaction else pool.messages
            target[row["id"]] = row
        elif statement.startswith("UPDATE conversations SET last_message_at"):
            if pool.fail_conversation_update:
                raise RuntimeError("simulated conversation update failure")
            target = (
                self.connection.pending_conversation_updates
                if self.connection.in_transaction
                else pool.conversation_updates
            )
            target.append(args[0])
        elif statement.startswith("SELECT id, sender_id, content, is_read, created_at FROM messages WHERE id"):
            self.row = self.connection.pending_messages.get(args[0]) or pool.messages.get(args[0])
        else:
            raise AssertionError(f"未处理的 SQL: {statement}")
        return 1

    async def fetchone(self):
        return self.row


def make_app():
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    app.dependency_overrides[chat.get_current_user] = lambda: 7
    return app


def test_message_releases_database_connection_while_waiting_for_guard(monkeypatch):
    async def scenario():
        pool = FakePool()
        monkeypatch.setattr(database, "pool", pool)

        async def allow(_state):
            assert pool.in_use == 0, "等待模型审核时不应占用数据库连接"
            return {"guard_passed": True}

        monkeypatch.setattr(chat.supervisor_app, "ainvoke", allow)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=make_app()), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/messages", json={"conversationId": 1, "senderId": 7, "content": "你好"}
            )

        assert response.status_code == 200
        assert response.json()["content"] == "你好"
        assert pool.in_use == 0
        assert list(pool.messages) == [1]
        assert pool.conversation_updates == [1]
        assert pool.commits == 1

    asyncio.run(scenario())


def test_concurrent_messages_allow_guard_tools_to_read_from_small_pool(monkeypatch):
    async def scenario():
        pool = FakePool(capacity=2)
        monkeypatch.setattr(database, "pool", pool)
        request_count = 4
        guards_waiting = 0
        tools_completed = 0
        all_guards_waiting = asyncio.Event()

        async def allow(_state):
            nonlocal guards_waiting, tools_completed
            guards_waiting += 1
            if guards_waiting == request_count:
                all_guards_waiting.set()
            # 所有请求都先到达模型等待点，再模拟图内工具借用同一连接池。
            await all_guards_waiting.wait()
            async with database.get_database().session() as cursor:
                await cursor.execute(
                    "SELECT user1_id, user2_id FROM conversations WHERE id = %s", [1]
                )
                assert (await cursor.fetchone())["user1_id"] == 7
                tools_completed += 1
            return {"guard_passed": True}

        monkeypatch.setattr(chat.supervisor_app, "ainvoke", allow)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=make_app()), base_url="http://test"
        ) as client:
            responses = await asyncio.wait_for(
                asyncio.gather(
                    *[
                        client.post(
                            "/api/messages",
                            json={"conversationId": 1, "senderId": 7, "content": f"消息{index}"},
                        )
                        for index in range(request_count)
                    ]
                ),
                timeout=5,
            )

        assert [response.status_code for response in responses] == [200] * request_count
        assert tools_completed == request_count
        assert len(pool.messages) == request_count
        assert pool.in_use == 0

    asyncio.run(scenario())


def test_conversation_update_failure_does_not_leave_inserted_message(monkeypatch):
    async def scenario():
        pool = FakePool()
        pool.fail_conversation_update = True
        monkeypatch.setattr(database, "pool", pool)

        async def allow(_state):
            return {"guard_passed": True}

        monkeypatch.setattr(chat.supervisor_app, "ainvoke", allow)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=make_app(), raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/messages", json={"conversationId": 1, "senderId": 7, "content": "你好"}
            )

        assert response.status_code == 500
        assert pool.messages == {}
        assert pool.conversation_updates == []
        assert pool.rollbacks == 1
        assert pool.in_use == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("guard_outcome, expected_status", [("rejected", 400), ("failed", 502), ("member_changed", 403)])
def test_guard_rejection_failure_or_changed_membership_does_not_write(
    monkeypatch, guard_outcome, expected_status
):
    async def scenario():
        pool = FakePool()
        monkeypatch.setattr(database, "pool", pool)

        async def guard(_state):
            if guard_outcome == "failed":
                raise RuntimeError("simulated model failure")
            if guard_outcome == "member_changed":
                pool.participants = {"user1_id": 8, "user2_id": 9}
            return {"guard_passed": guard_outcome != "rejected"}

        monkeypatch.setattr(chat.supervisor_app, "ainvoke", guard)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=make_app(), raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/messages", json={"conversationId": 1, "senderId": 7, "content": "你好"}
            )

        assert response.status_code == expected_status
        assert pool.messages == {}
        assert pool.conversation_updates == []
        assert pool.in_use == 0

    asyncio.run(scenario())
