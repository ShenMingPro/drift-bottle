"""聊天会话授权和消息审核回归测试。"""

import asyncio
from contextlib import asynccontextmanager

import pytest
from fastapi import HTTPException

import app.routers.chat as chat
from app.models.chat import SendMessageRequest


class FakeDatabase:
    def __init__(self, fetches):
        self.fetches = list(fetches)
        self.executions = []
        self.lastrowid = 1

    @asynccontextmanager
    async def session(self, *, transaction=False):
        yield self

    async def execute(self, query, args):
        self.executions.append((query, args))

    async def fetchone(self):
        return self.fetches.pop(0) if self.fetches else None

    async def fetchall(self):
        value = self.fetches.pop(0) if self.fetches else []
        return value if isinstance(value, list) else []


def conversation():
    return {"user1_id": 7, "user2_id": 8}


def message_row():
    return {
        "id": 1,
        "sender_id": 7,
        "content": "你好",
        "is_read": 0,
        "created_at": "2026-10-19 10:00:00",
    }


def test_non_member_cannot_send_or_read(monkeypatch):
    called = []

    async def must_not_guard(_state):
        called.append(True)
        return {"guard_passed": True}

    monkeypatch.setattr(chat.supervisor_app, "ainvoke", must_not_guard)

    send_db = FakeDatabase([conversation()])
    with pytest.raises(HTTPException) as send_error:
        asyncio.run(
            chat.send_message(
                SendMessageRequest(conversationId=1, senderId=9, content="越权消息"),
                db=send_db,
                current_user_id=9,
            )
        )
    assert send_error.value.status_code == 403
    assert called == []

    read_db = FakeDatabase([conversation()])
    with pytest.raises(HTTPException) as read_error:
        asyncio.run(
            chat.get_messages(
                conversationId=1,
                userId=9,
                db=read_db,
                current_user_id=9,
            )
        )
    assert read_error.value.status_code == 403


def test_member_message_always_runs_guard_and_is_inserted(monkeypatch):
    states = []

    async def allow(state):
        states.append(state)
        return {"guard_passed": True}

    monkeypatch.setattr(chat.supervisor_app, "ainvoke", allow)
    db = FakeDatabase([conversation(), conversation(), message_row()])

    result = asyncio.run(
        chat.send_message(
            SendMessageRequest(conversationId=1, senderId=7, content="你好"),
            db=db,
            current_user_id=7,
        )
    )

    assert result.content == "你好"
    assert states[0]["action"] == "send_message"
    assert any("INSERT INTO messages" in query for query, _args in db.executions)


def test_member_message_is_rejected_when_guard_fails(monkeypatch):
    async def reject(_state):
        return {"guard_passed": False, "guard_reason": "包含联系方式"}

    monkeypatch.setattr(chat.supervisor_app, "ainvoke", reject)
    db = FakeDatabase([conversation()])

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            chat.send_message(
                SendMessageRequest(conversationId=1, senderId=7, content="微信：abc123"),
                db=db,
                current_user_id=7,
            )
        )

    assert error.value.status_code == 400
    assert not any("INSERT INTO messages" in query for query, _args in db.executions)
