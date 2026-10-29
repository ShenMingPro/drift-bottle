"""显式拾取工作流的检查点、审批和恢复测试，不访问真实模型或 MySQL。"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from langchain_core.messages import AIMessage

import app.workflows.pick_workflow as workflow
import app.agents.ice_break_agent as ice_break


def state():
    return {
        "run_id": "run-1",
        "user_id": 7,
        "university": "测试大学",
        "city": "测试市",
        "province": "测试省",
        "status": "running",
        "error": None,
    }


def config(thread_id="pick-run:run-1"):
    return {"configurable": {"thread_id": thread_id}}


def install_fakes(monkeypatch):
    async def match(user_id, university, city, province):
        return {
            "agent": "match",
            "status": "success",
            "best_match": {
                "bottle_id": 12,
                "user_id": 23,
                "content": "最近喜欢摄影",
                "nickname": "小林",
                "contact_type": "微信",
                "contact_value": "private-contact",
            },
            "match_reason": "同校且有共同爱好",
            "match_score": 0.86,
            "match_level": "university",
        }

    async def profile(user_id, target_user_id):
        return {"agent": "profile_analysis", "status": "success", "tags": ["摄影"]}

    async def guide(user_id, target_user_id):
        return {"agent": "conversation_guide", "status": "success", "topic_suggestions": ["摄影"]}

    async def ice(user_id, target_user_id, analysis_context=None):
        return {
            "agent": "ice_break",
            "status": "success",
            "hints": ["你也喜欢摄影吗", "最近拍了什么", "违规联系方式"],
            "scenario": "ice_break",
        }

    async def guard(content, user_id):
        return {
            "agent": "guard",
            "status": "success",
            "passed": content != "违规联系方式",
            "reason": "审核通过" if content != "违规联系方式" else "包含联系方式",
            "risk_level": "low" if content != "违规联系方式" else "medium",
        }

    monkeypatch.setattr(workflow, "run_match", match)
    monkeypatch.setattr(workflow, "run_profile_analysis", profile)
    monkeypatch.setattr(workflow, "run_conversation_guide", guide)
    monkeypatch.setattr(workflow, "run_ice_break", ice)
    monkeypatch.setattr(workflow, "run_guard", guard)


def test_workflow_pauses_for_approval_and_strips_contact_info(monkeypatch):
    install_fakes(monkeypatch)

    async def run():
        graph = workflow.pick_workflow_graph.compile(checkpointer=InMemorySaver())
        first = await graph.ainvoke(state(), config())
        assert "__interrupt__" in first

        snapshot = await graph.aget_state(config())
        assert snapshot.next == ("approval",)
        assert snapshot.values["approved_hints"] == ["你也喜欢摄影吗", "最近拍了什么"]
        assert snapshot.values["ice_break"]["hints"] == ["你也喜欢摄影吗", "最近拍了什么"]
        assert "contact_value" not in snapshot.values["match"]["best_match"]
        assert snapshot.values["guard"]["passed"] is True
        approval = snapshot.tasks[0].interrupts[0].value
        assert approval["type"] == "pick_approval"
        assert approval["approvedHints"] == ["你也喜欢摄影吗", "最近拍了什么"]

        result = await graph.ainvoke(
            Command(resume={"approved": True, "hint": "你也喜欢摄影吗"}),
            config(),
        )
        assert result["status"] == "approved"
        assert result["selected_hint"] == "你也喜欢摄影吗"
        assert (await graph.aget_state(config())).next == ()

    asyncio.run(run())


def test_workflow_resume_cancel_is_terminal(monkeypatch):
    install_fakes(monkeypatch)

    async def run():
        graph = workflow.pick_workflow_graph.compile(checkpointer=InMemorySaver())
        await graph.ainvoke(state(), config())
        result = await graph.ainvoke(
            Command(resume={"approved": False}),
            config(),
        )
        assert result["status"] == "canceled"
        assert (await graph.aget_state(config())).next == ()

    asyncio.run(run())


def test_workflow_ends_without_interrupt_when_no_candidate(monkeypatch):
    install_fakes(monkeypatch)

    async def no_match(user_id, university, city, province):
        return {"agent": "match", "status": "no_result", "best_match": None}

    monkeypatch.setattr(workflow, "run_match", no_match)

    async def run():
        graph = workflow.pick_workflow_graph.compile(checkpointer=InMemorySaver())
        result = await graph.ainvoke(state(), config())
        assert result["status"] == "no_result"
        assert "__interrupt__" not in result
        assert (await graph.aget_state(config())).next == ()

    asyncio.run(run())


def test_invalid_match_contract_fails_before_later_agents(monkeypatch):
    install_fakes(monkeypatch)

    async def invalid_match(user_id, university, city, province):
        return {
            "agent": "match",
            "status": "success",
            "best_match": {"content": "缺少纸团 ID 和用户 ID"},
        }

    monkeypatch.setattr(workflow, "run_match", invalid_match)

    async def run():
        graph = workflow.pick_workflow_graph.compile(checkpointer=InMemorySaver())
        result = await graph.ainvoke(state(), config())
        assert result["status"] == "failed"
        assert "契约无效" in result["error"]
        assert "__interrupt__" not in result

    asyncio.run(run())


@pytest.mark.parametrize("failure_status", ["failed", "retryable_error", "fatal_error"])
@pytest.mark.parametrize("failed_step", ["match", "profile_analysis", "conversation_guide", "ice_break"])
def test_failed_artifact_stops_workflow_and_preserves_error(monkeypatch, failed_step, failure_status):
    install_fakes(monkeypatch)
    calls = []
    steps = ["match", "profile_analysis", "conversation_guide", "ice_break", "guard"]
    for step in steps:
        original = getattr(workflow, f"run_{step}")

        async def observed(*args, _step=step, _original=original, **kwargs):
            calls.append(_step)
            if _step == failed_step:
                payload = {
                    "agent": _step, "status": failure_status,
                    "error": "上游服务暂不可用", "error_code": "UPSTREAM_UNAVAILABLE",
                }
                if _step == "ice_break":
                    payload["hints"] = ["失败产物中的建议不能进入审批"]
                return payload
            return await _original(*args, **kwargs)

        monkeypatch.setattr(workflow, f"run_{step}", observed)

    async def run():
        graph = workflow.build_pick_workflow().compile(checkpointer=InMemorySaver())
        result = await graph.ainvoke(state(), config())
        assert result["status"] == "failed"
        assert "上游服务暂不可用" in result["error"]
        assert result[failed_step]["status"] == failure_status
        assert result[failed_step]["error_code"] == "UPSTREAM_UNAVAILABLE"
        assert "__interrupt__" not in result
        assert (await graph.aget_state(config())).next == ()

    asyncio.run(run())
    assert calls == steps[:steps.index(failed_step) + 1]


def test_workflow_ends_without_interrupt_when_guard_rejects_all(monkeypatch):
    install_fakes(monkeypatch)

    async def reject(content, user_id):
        return {"agent": "guard", "status": "success", "passed": False, "reason": "违规"}

    monkeypatch.setattr(workflow, "run_guard", reject)

    async def run():
        graph = workflow.pick_workflow_graph.compile(checkpointer=InMemorySaver())
        result = await graph.ainvoke(state(), config())
        assert result["status"] == "rejected"
        assert "__interrupt__" not in result
        assert (await graph.aget_state(config())).next == ()

    asyncio.run(run())


def test_workflow_can_resume_after_reopening_sqlite_checkpoint(tmp_path, monkeypatch):
    install_fakes(monkeypatch)
    path = str(tmp_path / "pick.sqlite")

    async def before_restart():
        async with AsyncSqliteSaver.from_conn_string(path) as saver:
            graph = workflow.pick_workflow_graph.compile(checkpointer=saver)
            await graph.ainvoke(state(), config())
            saved = await graph.aget_state(config())
            assert saved.next == ("approval",)
            assert saved.values["approved_hints"]

    async def after_restart():
        async with AsyncSqliteSaver.from_conn_string(path) as saver:
            graph = workflow.pick_workflow_graph.compile(checkpointer=saver)
            result = await graph.ainvoke(
                Command(resume={"approved": True, "hint": "最近拍了什么"}),
                config(),
            )
            assert result["status"] == "approved"
            assert result["selected_hint"] == "最近拍了什么"

    asyncio.run(before_restart())
    asyncio.run(after_restart())


def test_upstream_analysis_changes_actual_ice_break_prompt_and_old_call_still_works(monkeypatch):
    real_run_ice_break = workflow.run_ice_break
    install_fakes(monkeypatch)
    monkeypatch.setattr(workflow, "run_ice_break", real_run_ice_break)
    model_inputs = []
    topic = "街头纪实"

    async def profile(user_id, target_user_id):
        return {
            "agent": "profile_analysis", "status": "success", "tags": [topic],
            "compatibility": f"共同兴趣是{topic}",
            "metadata": {"private_notes": "不应传给生成模型"},
        }

    async def guide(user_id, target_user_id):
        return {
            "agent": "conversation_guide", "status": "success",
            "conversation_status": "new", "topic_suggestions": [f"聊聊{topic}"],
            "reply_strategies": [f"先询问{topic}的入门经验"],
        }

    async def query_profile(_user_id):
        return "对方爱好：摄影和运动"

    async def query_history(_user_id, _target_user_id):
        return "暂无对话记录"

    async def reply(messages):
        model_inputs.append(messages[0].content)
        return AIMessage(content=json.dumps({
            "hints": ["你也喜欢摄影吗", "最近拍了什么", "一起交流吗"],
            "scenario": "ice_break",
        }, ensure_ascii=False))

    monkeypatch.setattr(workflow, "run_profile_analysis", profile)
    monkeypatch.setattr(workflow, "run_conversation_guide", guide)
    monkeypatch.setattr(ice_break, "_query_user_profile", query_profile)
    monkeypatch.setattr(ice_break, "_query_chat_history", query_history)
    monkeypatch.setattr(ice_break, "_get_llm", lambda: SimpleNamespace(ainvoke=reply))

    async def run():
        nonlocal topic
        graph = workflow.build_pick_workflow().compile(checkpointer=InMemorySaver())
        for number, topic in enumerate(["街头纪实", "攀岩路线"]):
            result = await graph.ainvoke(state(), config(f"analysis-{number}"))
            assert "__interrupt__" in result
            assert f"共同兴趣是{topic}" in model_inputs[-1]
            assert f"聊聊{topic}" in model_inputs[-1]
            assert f"先询问{topic}的入门经验" in model_inputs[-1]
            assert "不应传给生成模型" not in model_inputs[-1]
        assert "街头纪实" not in model_inputs[1]
        # 原先只有两个用户 ID 的调用方式继续工作，不携带上游分析。
        legacy = await real_run_ice_break(7, 23)
        assert legacy["hints"]
        assert "攀岩路线" not in model_inputs[-1]
        assert "画像分析和沟通策略" not in model_inputs[-1]

    asyncio.run(run())
