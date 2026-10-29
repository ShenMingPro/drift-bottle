import json
import asyncio

from langchain_core.messages import AIMessage

import app.agents.match_agent as match_module


class _FakeTool:
    def __init__(self, name, handler):
        self.name = name
        self.coroutine = handler
        self.func = None
        self._handler = handler

    async def ainvoke(self, args):
        return await self._handler(args)


def test_execute_tools_carries_profile_into_match_call(monkeypatch):
    captured = {}

    async def fake_profile(args):
        return json.dumps({
            "user_id": args["user_id"],
            "nickname": "小明",
            "gender": "男",
            "age": 20,
            "hobbies": "摄影,跑步",
            "province": "广东",
            "city": "广州",
            "university": "A大学",
        }, ensure_ascii=False)

    async def fake_match(args):
        captured.update(args)
        return json.dumps({
            "result": "no_candidates",
            "best_match": None,
            "match_reason": "",
            "match_score": 0,
            "match_level": "",
        })

    monkeypatch.setattr(match_module, "match_tool_map", {
        "query_user_profile": _FakeTool("query_user_profile", fake_profile),
        "find_best_match": _FakeTool("find_best_match", fake_match),
    })

    # 故意把匹配调用放在前面，验证执行器会先完成画像查询。
    message = AIMessage(content="", tool_calls=[
        {
            "name": "find_best_match",
            "args": {"user_id": 7, "university": "A大学", "city": "广州", "province": "广东"},
            "id": "match-call",
            "type": "tool_call",
        },
        {
            "name": "query_user_profile",
            "args": {"user_id": 7},
            "id": "profile-call",
            "type": "tool_call",
        },
    ])

    updates = asyncio.run(match_module.execute_tools({
        "user_id": 7,
        "messages": [message],
        "user_profile": None,
    }))

    assert captured["user_profile"]["user_id"] == 7
    assert captured["user_profile"]["hobbies"] == "摄影,跑步"
    assert updates["profile_status"] == "success"
    assert updates["match_status"] == "no_candidates"


class _FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.executed = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def execute(self, sql, params):
        self.executed.append((sql, params))

    async def fetchall(self):
        return self.rows


class _FakeConnection:
    def __init__(self, rows):
        self.cursor_obj = _FakeCursor(rows)

    def cursor(self):
        return self.cursor_obj

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakePool:
    def __init__(self, rows):
        self.connection = _FakeConnection(rows)

    def acquire(self):
        return self.connection


def test_find_best_match_reuses_supplied_profile(monkeypatch):
    candidate_row = (
        101, 8, "喜欢摄影", "2026-01-01", "小红", "女", 21, "摄影,旅行",
        "", "", "广东", "广州", "A大学",
    )
    pool = _FakePool([candidate_row])
    async def fail_profile_load(user_id):
        raise AssertionError("不应重复查询画像")

    monkeypatch.setattr(match_module, "_load_user_profile", fail_profile_load)
    monkeypatch.setattr("app.database.pool", pool)

    profile = {
        "user_id": 7,
        "nickname": "小明",
        "gender": "男",
        "age": 20,
        "hobbies": "摄影,跑步",
        "province": "广东",
        "city": "广州",
        "university": "A大学",
    }
    raw = asyncio.run(match_module.find_best_match.ainvoke({
        "user_id": 7,
        "university": "A大学",
        "city": "广州",
        "province": "广东",
        "user_profile": profile,
    }))
    result = json.loads(raw)

    assert result["result"] == "success"
    assert result["profile_reused"] is True
    assert result["best_match"]["user_id"] == 8
    assert result["content_score"] > 0
