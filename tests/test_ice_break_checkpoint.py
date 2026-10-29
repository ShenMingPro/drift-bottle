"""用真实状态图和 SQLite 检查点验证人际聊天 AI 辅助，无外部连接。"""

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

import app.agents.ice_break_agent as ice_break


@pytest.fixture
def fake_dependencies(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    model_inputs = []
    context_queries = []

    def reply(messages):
        model_inputs.append(list(messages))
        round_number = len(model_inputs)
        hints = [f"第{round_number}轮建议{number}" for number in range(1, 4)]
        return AIMessage(
            content=json.dumps({"hints": hints, "scenario": "ice_break"}, ensure_ascii=False),
            id=f"answer-{round_number}",
        )

    async def profile(target_user_id):
        context_queries.append(("profile", target_user_id))
        return json.dumps({"昵称": f"用户{target_user_id}", "爱好": "篮球"}, ensure_ascii=False)

    async def history(user_id, target_user_id):
        context_queries.append(("history", user_id, target_user_id))
        return json.dumps({"场景": "主动破冰", "聊天记录": "暂无聊天记录"}, ensure_ascii=False)

    monkeypatch.setattr(ice_break, "_get_llm", lambda: RunnableLambda(reply))
    monkeypatch.setattr(ice_break, "_query_user_profile", profile)
    monkeypatch.setattr(ice_break, "_query_chat_history", history)
    return model_inputs, context_queries


def turn(request_id, user_id=7, target_user_id=8):
    return {
        "user_id": user_id,
        "target_user_id": target_user_id,
        "request_id": request_id,
        "hints": [],
        "messages": [],
    }


def config(user_id=7, target_user_id=8):
    return {"configurable": {"thread_id": f"ice-break:{user_id}:{target_user_id}"}}


def prompt_text(messages):
    return "\n".join(str(message.content) for message in messages)


def test_same_thread_uses_previous_hints_without_accumulating_old_messages(tmp_path, fake_dependencies):
    model_inputs, context_queries = fake_dependencies

    async def run():
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "rounds.sqlite")) as saver:
            graph = ice_break.build_ice_break_graph().compile(checkpointer=saver)
            first = await graph.ainvoke(turn("request-1"), config())
            second = await graph.ainvoke(turn("request-2"), config())

            assert first["recent_hints"] == first["hints"]
            assert second["hints"] != first["hints"]
            assert second["recent_hints"] == [*first["hints"], *second["hints"]]
            assert len(second["messages"]) == 3
            assert isinstance(second["messages"][0], SystemMessage)
            assert isinstance(second["messages"][1], HumanMessage)
            assert isinstance(second["messages"][2], AIMessage)
            assert second["request_id"] == "request-2"
            assert all(hint in prompt_text(model_inputs[1]) for hint in first["hints"])

    asyncio.run(run())
    assert len(model_inputs) == 2
    assert context_queries.count(("profile", 8)) == 2
    assert context_queries.count(("history", 7, 8)) == 2


def test_sqlite_reopen_preserves_recent_hints_for_new_graph(tmp_path, fake_dependencies):
    model_inputs, _queries = fake_dependencies
    path = str(tmp_path / "restart.sqlite")
    original_hints = []

    async def before_restart():
        async with AsyncSqliteSaver.from_conn_string(path) as saver:
            graph = ice_break.build_ice_break_graph().compile(checkpointer=saver)
            result = await graph.ainvoke(turn("request-before"), config())
            original_hints.extend(result["hints"])

    async def after_restart():
        async with AsyncSqliteSaver.from_conn_string(path) as saver:
            graph = ice_break.build_ice_break_graph().compile(checkpointer=saver)
            saved = await graph.aget_state(config())
            assert saved.next == ()
            assert saved.values["recent_hints"] == original_hints
            result = await graph.ainvoke(turn("request-after"), config())
            assert result["recent_hints"] == [*original_hints, *result["hints"]]
            assert all(hint in prompt_text(model_inputs[1]) for hint in original_hints)

    # 关闭 SQLite 连接及事件循环，重编译图，模拟服务重新启动。
    asyncio.run(before_restart())
    asyncio.run(after_restart())
    assert len(model_inputs) == 2


@pytest.mark.parametrize("user_id,target_user_id", [(7, 9), (9, 8)])
def test_different_users_or_targets_do_not_share_recent_hints(
    tmp_path, fake_dependencies, user_id, target_user_id
):
    model_inputs, _queries = fake_dependencies

    async def run():
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "isolated.sqlite")) as saver:
            graph = ice_break.build_ice_break_graph().compile(checkpointer=saver)
            first = await graph.ainvoke(turn("request-a"), config())
            other = await graph.ainvoke(
                turn("request-b", user_id, target_user_id), config(user_id, target_user_id)
            )
            assert other["recent_hints"] == other["hints"]
            assert all(hint not in prompt_text(model_inputs[1]) for hint in first["hints"])
            saved_first = await graph.aget_state(config())
            assert saved_first.values["recent_hints"] == first["hints"]

    asyncio.run(run())


def test_recent_hints_are_bounded_to_last_nine_suggestions(tmp_path, fake_dependencies):
    _inputs, _queries = fake_dependencies

    async def run():
        generated = []
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "bounded.sqlite")) as saver:
            graph = ice_break.build_ice_break_graph().compile(checkpointer=saver)
            for number in range(1, 6):
                result = await graph.ainvoke(turn(f"request-{number}"), config())
                generated.extend(result["hints"])
                assert result["recent_hints"] == generated[-9:]
            assert len(result["recent_hints"]) == 9

    asyncio.run(run())


def test_repeated_suggestions_do_not_fill_recent_memory_with_duplicates(
    tmp_path, fake_dependencies, monkeypatch
):
    responses = iter([
        ["喜欢篮球吗", "周末去运动吗", "平时看球吗"],
        ["喜欢篮球吗", "最近在看什么", "最近在看什么"],
    ])

    def reply(_messages):
        return AIMessage(content=json.dumps({
            "hints": next(responses), "scenario": "ice_break",
        }, ensure_ascii=False))

    monkeypatch.setattr(ice_break, "_get_llm", lambda: RunnableLambda(reply))

    async def run():
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "unique.sqlite")) as saver:
            graph = ice_break.build_ice_break_graph().compile(checkpointer=saver)
            await graph.ainvoke(turn("request-1"), config())
            result = await graph.ainvoke(turn("request-2"), config())
            assert len(result["recent_hints"]) == 4
            assert set(result["recent_hints"]) == {
                "喜欢篮球吗", "周末去运动吗", "平时看球吗", "最近在看什么",
            }

    asyncio.run(run())


def test_resume_failed_generation_does_not_query_completed_context_again(
    tmp_path, fake_dependencies, monkeypatch
):
    _inputs, context_queries = fake_dependencies
    path = str(tmp_path / "resume.sqlite")
    attempts = []

    def fail_once(messages):
        attempts.append(list(messages))
        if len(attempts) == 1:
            raise RuntimeError("模拟生成节点失败")
        return AIMessage(content=json.dumps({
            "hints": ["恢复后的建议1", "恢复后的建议2", "恢复后的建议3"],
            "scenario": "ice_break",
        }, ensure_ascii=False))

    monkeypatch.setattr(ice_break, "_get_llm", lambda: RunnableLambda(fail_once))

    async def failed_run():
        async with AsyncSqliteSaver.from_conn_string(path) as saver:
            graph = ice_break.build_ice_break_graph().compile(checkpointer=saver)
            with pytest.raises(RuntimeError, match="生成节点失败"):
                await graph.ainvoke(turn("request-resume"), config())
            saved = await graph.aget_state(config())
            assert saved.next == ("generate_hints",)
            assert len(context_queries) == 2
            assert saved.values["target_profile"]
            assert saved.values["chat_context"]

    async def resumed_run():
        async with AsyncSqliteSaver.from_conn_string(path) as saver:
            graph = ice_break.build_ice_break_graph().compile(checkpointer=saver)
            result = await graph.ainvoke(None, config())
            assert result["hints"] == ["恢复后的建议1", "恢复后的建议2", "恢复后的建议3"]
            assert result["recent_hints"] == result["hints"]
            assert (await graph.aget_state(config())).next == ()

    asyncio.run(failed_run())
    asyncio.run(resumed_run())
    assert context_queries.count(("profile", 8)) == 1
    assert context_queries.count(("history", 7, 8)) == 1
    assert len(attempts) == 2
