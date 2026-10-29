import asyncio

import pytest
from fastapi import HTTPException

from app.models.user import LoginRequest, RegisterRequest
from app.routers.auth import login, register


class FakeAuthDatabase:
    def __init__(self):
        self.users = {}
        self.lastrowid = 0
        self._next_row = None

    async def execute(self, query, args):
        if query.startswith("SELECT id FROM users WHERE username"):
            user = self.users.get(args[0])
            self._next_row = {"id": user["id"]} if user else None
        elif query.startswith("SELECT id, password"):
            self._next_row = self.users.get(args[0])
        elif query.startswith("INSERT INTO users"):
            self.lastrowid += 1
            self.users[args[0]] = {
                "id": self.lastrowid,
                "password": args[1],
                "nickname": args[2],
                "gender": args[3],
                "age": args[4],
                "hobbies": args[5],
                "contact_type": args[6],
                "contact_value": args[7],
                "province": args[8],
                "city": args[9],
                "university": args[10],
                "write_chances": 1,
                "pick_chances": 1,
            }
            self._next_row = None
        elif query.startswith("UPDATE users SET password"):
            for user in self.users.values():
                if user["id"] == args[1]:
                    user["password"] = args[0]
                    break
            self._next_row = None
        else:
            raise AssertionError(f"未处理的 SQL: {query}")

    async def fetchone(self):
        return self._next_row


def register_request(username="alice_01", password="abcdefgh"):
    return RegisterRequest(
        username=username,
        password=password,
        gender="女",
        age=20,
        hobbies=["编程"],
        contactValue="123456",
        province="上海市",
        city="上海市",
        university="测试大学",
    )


def test_register_hashes_password_and_login_succeeds():
    db = FakeAuthDatabase()
    response = asyncio.run(register(register_request(), db=db))

    assert response.userId == 1
    assert response.nickname == "alice_01"
    assert db.users["alice_01"]["password"] != "abcdefgh"
    assert db.users["alice_01"]["nickname"] == "alice_01"

    login_response = asyncio.run(
        login(LoginRequest(username="ALICE_01", password="abcdefgh"), db=db)
    )
    assert login_response.userId == 1


def test_duplicate_username_returns_conflict():
    db = FakeAuthDatabase()
    asyncio.run(register(register_request(), db=db))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(register(register_request(username="Alice_01"), db=db))

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "该用户名已被使用"


def test_legacy_plaintext_password_is_upgraded_after_login():
    db = FakeAuthDatabase()
    db.users["legacy_user"] = {
        "id": 1,
        "password": "legacy-pass",
        "nickname": "Legacy",
        "gender": "",
        "age": 0,
        "hobbies": "",
        "contact_type": "",
        "contact_value": "",
        "province": "",
        "city": "",
        "university": "",
        "write_chances": 1,
        "pick_chances": 1,
    }

    asyncio.run(login(LoginRequest(username="legacy_user", password="legacy-pass"), db=db))

    assert db.users["legacy_user"]["password"].startswith("$argon2")


def test_wrong_password_returns_generic_unauthorized_error():
    db = FakeAuthDatabase()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(login(LoginRequest(username="missing", password="wrong"), db=db))

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "用户名或密码错误"
