"""通过真实 HTTP 路由和生成状态图验证连续刷新，不访问远端模型或数据库。"""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import app.agents.content_gen_agent as agent
import app.routers.content_gen as routes
from app.auth_jwt import create_access_token
from app.ai_content_proof import verify_ai_content_proof
from app.config import settings
from app.database import get_db


OLD = [
    {"content": "王者dd 星耀卡三天了 求个野王带带 我辅助贼稳", "style": "找人玩"},
    {"content": "兴安盟这天也太冷了 宿舍躺一天了 啥也不想干", "style": "闲聊吐槽"},
    {"content": "宿舍躺一天了 有没有人聊天 随便说点啥都行", "style": "想聊天"},
]
NEW = [
    {"content": "有人想约羽毛球吗 傍晚有空一起去动一动", "style": "找人玩"},
    {"content": "刚洗完衣服又开始下雨了 谁懂晒不干的痛", "style": "闲聊吐槽"},
    {"content": "最近想试试自己做饭 有人愿意分享拿手菜吗", "style": "想聊天"},
]
ENDPOINTS = ["/api/content-gen", "/api/content-gen/stream"]


def model_reply(suggestions):
    return json.dumps({"suggestions": suggestions}, ensure_ascii=False)


def request(runtime, endpoint, previous=None, *, headers=None, method="GET"):
    async def run():
        params = [("userId", "7")]
        params.extend(("previousSuggestions", content) for content in previous or [])
        auth = {"Authorization": f"Bearer {create_access_token(7)}"} if headers is None else headers
        async with runtime.client() as client:
            if method == "POST":
                return await client.post(endpoint, json={"userId": 7, "previousSuggestions": previous or []}, headers=auth)
            return await client.get(endpoint, params=params, headers=auth)

    return asyncio.run(run())


def result_suggestions(response, endpoint):
    assert response.status_code == 200
    if endpoint.endswith("/stream"):
        events = sse_events(response)
        assert not any(event == "error" for event, _ in events), events
        results = [data for event, data in events if event == "result"]
        assert len(results) == 1
        payload = results[0]
    else:
        payload = response.json()
    suggestions = payload["suggestions"]
    assert len(suggestions) == 3
    return suggestions


def sse_events(response):
    events = []
    for block in response.text.replace("\r\n", "\n").split("\n\n"):
        name = "message"
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line.partition(":")[2].strip()
            elif line.startswith("data:"):
                events.append((name, json.loads(line.partition(":")[2])))
    return events


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setattr(settings, "JWT_SECRET", "content-generation-tests-only-secret-2026")
    observations = SimpleNamespace(inputs=[], replies=[])

    async def reply(messages):
        observations.inputs.append(messages)
        if observations.replies:
            return AIMessage(content=observations.replies.pop(0))
        # 模拟模型在无历史约束时总是给出同一批建议。
        has_history = any(OLD[0]["content"] in str(message.content) for message in messages)
        return AIMessage(content=json.dumps({"suggestions": NEW if has_history else OLD}, ensure_ascii=False))

    async def profile(_user_id):
        return json.dumps({"爱好": "王者荣耀、羽毛球", "城市": "兴安盟"}, ensure_ascii=False)

    async def topics():
        return "暂无热门话题"

    async def unused_database():
        yield object()

    monkeypatch.setattr(agent, "_get_llm", lambda: RunnableLambda(reply))
    monkeypatch.setattr(agent, "_query_user_profile", profile)
    monkeypatch.setattr(agent, "_query_recent_hot_topics", topics)
    monkeypatch.setattr(routes, "content_gen_app", agent.build_content_gen_graph().compile())
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    app.dependency_overrides[get_db] = unused_database

    @asynccontextmanager
    async def client():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as session:
            yield session

    return SimpleNamespace(client=client, observations=observations)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_refresh_excludes_previous_suggestions(runtime, endpoint):
    async def run():
        headers = {"Authorization": f"Bearer {create_access_token(7)}"}
        async with runtime.client() as client:
            first = await client.get(endpoint, params={"userId": 7}, headers=headers)
            params = [("userId", "7"), *[("previousSuggestions", item["content"]) for item in OLD]]
            second = await client.get(endpoint, params=params, headers=headers)
        assert first.status_code == second.status_code == 200
        if endpoint.endswith("/stream"):
            original = next(data for event, data in sse_events(first) if event == "result")
            refreshed = next(data for event, data in sse_events(second) if event == "result")
        else:
            original, refreshed = first.json(), second.json()
        old_contents = {item["content"] for item in original["suggestions"]}
        new_contents = {item["content"] for item in refreshed["suggestions"]}
        assert len(new_contents) == 3
        assert not old_contents.intersection(new_contents), "点击 AI 生成后仍返回上一批相同建议"

    asyncio.run(run())


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_initial_generation_without_history_remains_compatible_and_signed(runtime, endpoint):
    response = request(runtime, endpoint)
    suggestions = result_suggestions(response, endpoint)
    assert [item["content"] for item in suggestions] == [item["content"] for item in OLD]
    assert len(runtime.observations.inputs) == 1
    for item in suggestions:
        assert verify_ai_content_proof(item["proof"], 7, item["content"])
        assert not verify_ai_content_proof(item["proof"], 8, item["content"])
        assert not verify_ai_content_proof(item["proof"], 7, item["content"] + "！")


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_history_is_in_prompt_and_repeated_model_response_is_retried(runtime, endpoint):
    runtime.observations.replies = [model_reply(OLD), model_reply(NEW)]
    response = request(runtime, endpoint, [item["content"] for item in OLD])
    suggestions = result_suggestions(response, endpoint)
    assert [item["content"] for item in suggestions] == [item["content"] for item in NEW]
    assert len(runtime.observations.inputs) == 2
    for messages in runtime.observations.inputs:
        prompt = "\n".join(str(message.content) for message in messages)
        assert all(item["content"] in prompt for item in OLD)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize(
    "invalid_reply",
    [
        "生成失败，暂时没有 JSON",
        "[]",
        "null",
        '{"suggestions": null}',
        '{"suggestions": "这不是数组"}',
        model_reply(NEW[:2]),
        model_reply(NEW + OLD[:1]),
        model_reply([NEW[0], NEW[1], {"content": "", "style": "想聊天"}]),
        model_reply([NEW[0], NEW[1], {"content": 123, "style": "想聊天"}]),
        model_reply([NEW[0], NEW[1], {"style": "想聊天"}]),
        model_reply([NEW[0], NEW[1], {"content": "今天想和大家聊聊最近看的电影", "style": ""}]),
        model_reply([NEW[0], NEW[1], {"content": "今天想和大家聊聊最近看的电影", "style": "未知类型"}]),
        model_reply([NEW[0], NEW[1], {"content": "今天想和大家聊聊最近看的电影", "style": ["想聊天"]}]),
        model_reply([NEW[0], NEW[1], {"content": "今天想和大家聊聊最近看的电影", "style": {"name": "想聊天"}}]),
        model_reply([NEW[0], NEW[1], {"content": ["今天想和大家聊聊最近看的电影"], "style": "想聊天"}]),
        model_reply([NEW[0], NEW[1], None]),
        model_reply([NEW[0], NEW[1], {"content": "电影推荐" * 51, "style": "想聊天"}]),
    ],
    ids=[
        "invalid-json", "array-root", "null-root", "null-suggestions", "string-suggestions",
        "too-few", "too-many", "empty-content", "numeric-content", "missing-content",
        "empty-style", "unknown-style", "list-style", "object-style", "list-content", "null-item", "oversized-content",
    ],
)
def test_invalid_model_output_is_retried_instead_of_using_fixed_fallback(runtime, endpoint, invalid_reply):
    runtime.observations.replies = [invalid_reply, model_reply(NEW)]
    suggestions = result_suggestions(request(runtime, endpoint), endpoint)
    assert [item["content"] for item in suggestions] == [item["content"] for item in NEW]
    assert len(runtime.observations.inputs) == 2


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize(
    "variant",
    [
        "王者dd，星耀卡三天了！求个野王带带；我辅助贼稳。",
        "王者ｄｄ　星耀卡三天了，求个野王带带，我辅助贼稳",
        "王者dd 星耀卡三天了 求个野王带带 我辅助很稳",
    ],
    ids=["punctuation", "unicode-width", "near-duplicate"],
)
def test_history_near_duplicates_are_rejected(runtime, endpoint, variant):
    repeated = [{**OLD[0], "content": variant}, *NEW[1:]]
    runtime.observations.replies = [model_reply(repeated), model_reply(NEW)]
    suggestions = result_suggestions(request(runtime, endpoint, [OLD[0]["content"]]), endpoint)
    assert [item["content"] for item in suggestions] == [item["content"] for item in NEW]
    assert len(runtime.observations.inputs) == 2


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_duplicates_inside_one_batch_are_retried(runtime, endpoint):
    repeated = [NEW[0], {**NEW[1], "content": NEW[0]["content"] + "！"}, NEW[2]]
    runtime.observations.replies = [model_reply(repeated), model_reply(NEW)]
    suggestions = result_suggestions(request(runtime, endpoint), endpoint)
    assert [item["content"] for item in suggestions] == [item["content"] for item in NEW]
    assert len(runtime.observations.inputs) == 2


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_third_attempt_can_succeed_after_two_repeated_batches(runtime, endpoint):
    runtime.observations.replies = [model_reply(OLD), model_reply(OLD), model_reply(NEW)]
    suggestions = result_suggestions(request(runtime, endpoint, [item["content"] for item in OLD]), endpoint)
    assert [item["content"] for item in suggestions] == [item["content"] for item in NEW]
    assert len(runtime.observations.inputs) == 3


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_unsafe_content_is_retried_before_signing(runtime, endpoint, monkeypatch):
    unsafe = "加我微信：abcdef123"
    runtime.observations.replies = [model_reply([NEW[0], NEW[1], {"content": unsafe, "style": "想聊天"}]), model_reply(NEW)]
    signed = []
    original_sign = routes.create_ai_content_proof

    def track_sign(user_id, content):
        signed.append(content)
        return original_sign(user_id, content)

    monkeypatch.setattr(routes, "create_ai_content_proof", track_sign)
    suggestions = result_suggestions(request(runtime, endpoint), endpoint)
    assert [item["content"] for item in suggestions] == [item["content"] for item in NEW]
    assert signed == [item["content"] for item in NEW]
    assert unsafe not in signed
    assert len(runtime.observations.inputs) == 2


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("failure", ["duplicate", "invalid", "unsafe"])
def test_retry_exhaustion_returns_error_without_old_or_default_suggestions(runtime, endpoint, failure, monkeypatch):
    failed_reply = {
        "duplicate": model_reply(OLD),
        "invalid": "not json",
        "unsafe": model_reply([NEW[0], NEW[1], {"content": "加我微信：abcdef123", "style": "想聊天"}]),
    }[failure]
    runtime.observations.replies = [failed_reply] * 3
    signed = []
    monkeypatch.setattr(routes, "create_ai_content_proof", lambda *args: signed.append(args))
    response = request(runtime, endpoint, [item["content"] for item in OLD])
    assert len(runtime.observations.inputs) == 3
    assert not signed, "失败的批次不能签发内容凭证"
    if endpoint.endswith("/stream"):
        assert response.status_code == 200
        events = sse_events(response)
        assert any(event == "error" for event, _ in events)
        assert not any(event == "result" for event, _ in events)
    else:
        assert response.status_code == 503
        assert "suggestions" not in response.json()
    assert all(item["content"] not in response.text for item in OLD)
    assert "有没有人一起打游戏 来不坑的" not in response.text
    assert "食堂的饭太难吃了 救命" not in response.text
    assert "好无聊 谁来聊天" not in response.text


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize(
    "history",
    [
        [f"第{index}个已经看过的不同话题" for index in range(31)],
        ["历史消息" * 50 + "超"],
    ],
    ids=["more-than-30-items", "more-than-200-characters"],
)
def test_oversized_history_is_rejected_before_model_call(runtime, endpoint, history):
    response = request(runtime, endpoint, history)
    assert response.status_code == 422
    assert runtime.observations.inputs == []


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_history_limits_accept_exactly_30_items_and_200_characters(runtime, endpoint):
    history = [f"历史话题{index}" for index in range(29)] + ["春夏秋冬" * 50]
    response = request(runtime, endpoint, history)
    result_suggestions(response, endpoint)
    assert len(runtime.observations.inputs) == 1
    prompt = "\n".join(str(message.content) for message in runtime.observations.inputs[0])
    assert all(content in prompt for content in history)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("authentication", ["missing", "invalid", "different-user"])
def test_authentication_is_checked_before_model_call(runtime, endpoint, authentication):
    headers = {
        "missing": {},
        "invalid": {"Authorization": "Bearer invalid-token"},
        "different-user": {"Authorization": f"Bearer {create_access_token(8)}"},
    }[authentication]
    response = request(runtime, endpoint, [OLD[0]["content"]], headers=headers)
    assert response.status_code == (403 if authentication == "different-user" else 401)
    assert runtime.observations.inputs == []


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_post_transports_large_history_in_body_and_returns_new_signed_suggestions(runtime, endpoint):
    history = [item["content"] for item in OLD] + [
        (f"已展示的历史话题{index}" + "春夏秋冬" * 50)[:200] for index in range(27)
    ]
    response = request(runtime, endpoint, history, method="POST")
    suggestions = result_suggestions(response, endpoint)
    assert [item["content"] for item in suggestions] == [item["content"] for item in NEW]
    assert response.request.method == "POST"
    assert not response.request.url.query
    assert response.headers["cache-control"] == "no-store"
    prompt = "\n".join(str(message.content) for message in runtime.observations.inputs[0])
    assert all(content in prompt for content in history)
    assert all(verify_ai_content_proof(item["proof"], 7, item["content"]) for item in suggestions)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("history", [["历史内容"] * 31, ["字" * 201], [""]])
def test_post_history_validation_happens_before_model_call(runtime, endpoint, history):
    assert request(runtime, endpoint, history, method="POST").status_code == 422
    assert runtime.observations.inputs == []


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_post_rejects_mismatched_identity_before_model_call(runtime, endpoint):
    response = request(runtime, endpoint, method="POST", headers={"Authorization": f"Bearer {create_access_token(8)}"})
    assert response.status_code == 403
    assert runtime.observations.inputs == []
