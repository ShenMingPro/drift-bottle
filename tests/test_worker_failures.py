"""子 Agent 的基础设施和输出错误不能伪装成正常业务结果。"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

import app.agents.content_gen_agent as content_gen
import app.agents.content_guard_agent as guard
import app.agents.conversation_guide_agent as guide
import app.agents.ice_break_agent as ice_break
import app.agents.match_agent as match
import app.agents.profile_analysis_agent as profile
import app.agents.supervisor_agent as supervisor


def state_for(payload):
    return {"user_id": 7, "messages": [AIMessage(content=json.dumps(payload))]}


@pytest.mark.parametrize("payload", [
    {"passed": "false", "reason": "不通过", "risk_level": "high"},
    {"passed": 1, "reason": "通过", "risk_level": "low"},
    [], {},
])
def test_guard_invalid_result_is_failure_instead_of_approval_or_rejection(payload):
    with pytest.raises(ValueError):
        guard.extract_result(state_for(payload))


def test_guard_real_boolean_false_remains_business_rejection():
    result = guard.extract_result(state_for({"passed": False, "reason": "违规", "risk_level": "high"}))
    assert result == {"passed": False, "reason": "违规", "risk_level": "high"}


@pytest.mark.parametrize("module,payload", [
    (profile, {}),
    (profile, {"tags": ["摄影"], "completeness_score": "bad", "suggestions": []}),
    (guide, {}),
    (guide, {"conversation_status": "new", "topic_suggestions": None, "reply_strategies": [], "ice_breaker": "你好"}),
    (ice_break, {"hints": [], "scenario": "ice_break"}),
    (ice_break, {"hints": ["只有一条"], "scenario": "ice_break"}),
])
def test_invalid_worker_output_does_not_return_default_success(module, payload):
    with pytest.raises(ValueError):
        module.extract_result(state_for(payload))


@pytest.mark.parametrize("module", [profile, guide, ice_break])
def test_non_json_worker_output_is_explicit_failure(module):
    with pytest.raises(ValueError):
        module.extract_result({"user_id": 7, "messages": [AIMessage(content="not json")]})


@pytest.mark.parametrize("module,name,args", [
    (profile, "_query_user_profile", (7,)),
    (guide, "_query_user_profile", (7,)),
    (guide, "_query_chat_history", (7, 8)),
    (ice_break, "_query_user_profile", (7,)),
    (ice_break, "_query_chat_history", (7, 8)),
    (content_gen, "_query_user_profile", (7,)),
    (content_gen, "_query_recent_hot_topics", ()),
])
def test_database_failure_propagates_instead_of_becoming_prompt_text(monkeypatch, module, name, args):
    def unavailable():
        raise ConnectionError("数据库临时不可用")
    monkeypatch.setattr("app.database.pool", SimpleNamespace(acquire=unavailable))
    with pytest.raises(ConnectionError, match="数据库临时不可用"):
        asyncio.run(getattr(module, name)(*args))


@pytest.mark.parametrize("through_supervisor_tool", [False, True])
def test_match_tool_failure_cannot_be_reported_as_no_candidates(monkeypatch, through_supervisor_tool):
    class Model:
        def bind_tools(self, _tools):
            return self
        async def ainvoke(self, messages):
            if len(messages) == 2:
                return AIMessage(content="", tool_calls=[{
                    "name": "find_best_match", "args": {}, "id": "match-1", "type": "tool_call",
                }])
            return AIMessage(content='{"best_match":null}')

    async def failed(_args):
        raise ConnectionError("匹配数据库不可用")
    monkeypatch.setattr(match, "_get_llm", lambda: Model())
    monkeypatch.setattr(match, "match_tool_map", {"find_best_match": SimpleNamespace(coroutine=failed, ainvoke=failed)})
    with pytest.raises(ConnectionError, match="匹配数据库不可用"):
        if through_supervisor_tool:
            asyncio.run(supervisor.call_match_agent.ainvoke({"user_id": 7, "university": "", "city": "", "province": ""}))
        else:
            asyncio.run(match.match_app.ainvoke({"user_id": 7, "messages": []}))


def test_match_failed_checkpoint_state_cannot_use_a_model_invented_result():
    state = state_for({"best_match": {"bottle_id": 1, "user_id": 8}})
    state["match_status"] = "tool_error"
    with pytest.raises(ValueError, match="工具执行失败"):
        match.extract_result(state)


def test_invalid_ice_break_can_regenerate_on_same_checkpoint_request(monkeypatch):
    replies = iter([AIMessage(content="invalid"), AIMessage(content=json.dumps({
        "hints": ["聊篮球", "聊音乐", "聊电影"], "scenario": "ice_break",
    }))])
    calls = []
    async def reply(messages):
        calls.append(messages)
        return next(replies)
    async def query(*_args):
        return "{}"
    monkeypatch.setattr(ice_break, "_get_llm", lambda: SimpleNamespace(ainvoke=reply))
    monkeypatch.setattr(ice_break, "_query_user_profile", query)
    monkeypatch.setattr(ice_break, "_query_chat_history", query)

    async def run():
        graph = ice_break.build_ice_break_graph().compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "failed-format-request"}}
        with pytest.raises(ValueError):
            await graph.ainvoke({"user_id": 7, "target_user_id": 8, "request_id": "request-1", "messages": []}, config)
        snapshot = await graph.aget_state(config)
        assert not snapshot.values.get("completed_requests")
        result = await graph.ainvoke(None, config)
        assert len(calls) == 2
        assert result["hints"] == ["聊篮球", "聊音乐", "聊电影"]
        assert result["completed_requests"][0]["request_id"] == "request-1"
    asyncio.run(run())
