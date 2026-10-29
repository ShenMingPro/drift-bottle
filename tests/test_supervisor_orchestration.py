import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

import app.agents.supervisor_agent as supervisor


class FakeTool:
    def __init__(self, callback):
        self.func = callback
        self._callback = callback

    def invoke(self, arguments):
        return self._callback(arguments)


def supervisor_state(
    tool_name,
    *,
    arguments=None,
    action="pick_full",
    artifacts=None,
    step_count=0,
    max_steps=8,
):
    return {
        "user_id": 7,
        "action": action,
        "content": "来自用户的原始内容",
        "target_user_id": 0,
        "target_university": "测试大学",
        "target_city": "测试市",
        "target_province": "测试省",
        "content_type": "bottle",
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool_name,
                        "args": arguments or {},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        "final_result": "",
        "guard_passed": True,
        "guard_reason": "",
        "completed_steps": [],
        "artifacts": artifacts or {},
        "step_count": step_count,
        "retry_count": 0,
        "max_steps": max_steps,
        "status": "running",
        "error": None,
    }


def run_tools(state):
    return asyncio.run(supervisor.execute_tools(state))


def test_execute_tools_records_successful_artifact(monkeypatch):
    payload = {
        "agent": "match",
        "status": "success",
        "best_match": {"user_id": 23, "bottle_id": 12},
        "match_reason": "同校且有共同爱好",
        "match_score": 0.82,
        "match_level": "university",
    }
    monkeypatch.setattr(
        supervisor,
        "supervisor_tool_map",
        {"call_match_agent": FakeTool(lambda _args: json.dumps(payload))},
    )

    result = run_tools(supervisor_state("call_match_agent"))

    assert result["artifacts"] == {"match": payload}
    assert result["completed_steps"] == ["match"]
    assert result["step_count"] == 1
    assert result["status"] == "running"
    assert json.loads(result["messages"][0].content) == payload
    assert result["messages"][0].tool_call_id == "call-1"


def test_execute_tools_overrides_model_identity_and_target_with_trusted_state(monkeypatch):
    captured_arguments = []

    def profile_tool(arguments):
        captured_arguments.append(arguments)
        return json.dumps(
            {
                "agent": "profile_analysis",
                "status": "success",
                "user_id": arguments["user_id"],
                "target_user_id": arguments["target_user_id"],
                "compatibility": "兴趣互补",
            }
        )

    monkeypatch.setattr(
        supervisor,
        "supervisor_tool_map",
        {"call_profile_analysis_agent": FakeTool(profile_tool)},
    )
    state = supervisor_state(
        "call_profile_analysis_agent",
        arguments={"user_id": 999, "target_user_id": 999},
        artifacts={
            "match": {
                "agent": "match",
                "status": "success",
                "best_match": {"user_id": 23, "bottle_id": 12},
            }
        },
    )

    result = run_tools(state)

    assert captured_arguments == [{"user_id": 7, "target_user_id": 23}]
    assert result["artifacts"]["profile_analysis"]["target_user_id"] == 23


def test_no_match_is_recorded_as_terminal_no_result(monkeypatch):
    payload = {
        "agent": "match",
        "status": "no_result",
        "result": "no_candidates",
        "best_match": None,
        "match_reason": "暂时没有符合条件的纸团",
        "match_score": 0.0,
        "match_level": "",
    }
    monkeypatch.setattr(
        supervisor,
        "supervisor_tool_map",
        {"call_match_agent": FakeTool(lambda _args: json.dumps(payload))},
    )

    result = run_tools(supervisor_state("call_match_agent"))

    assert result["artifacts"]["match"] == payload
    assert result["completed_steps"] == ["match"]
    assert result["status"] == "no_result"
    assert result["error"] is None


def test_no_match_skips_dependent_agents_in_same_supervisor_turn(monkeypatch):
    payload = {
        "agent": "match",
        "status": "no_result",
        "result": "no_candidates",
        "best_match": None,
    }
    profile_calls = []

    def profile_tool(arguments):
        profile_calls.append(arguments)
        return json.dumps(
            {
                "agent": "profile_analysis",
                "status": "success",
                "target_user_id": arguments.get("target_user_id"),
            }
        )

    monkeypatch.setattr(
        supervisor,
        "supervisor_tool_map",
        {
            "call_match_agent": FakeTool(lambda _args: json.dumps(payload)),
            "call_profile_analysis_agent": FakeTool(profile_tool),
        },
    )
    state = supervisor_state("call_match_agent", action="pick_full")
    state["messages"] = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "call_match_agent",
                    "args": {},
                    "id": "match-call",
                    "type": "tool_call",
                },
                {
                    "name": "call_profile_analysis_agent",
                    "args": {"target_user_id": 999},
                    "id": "profile-call",
                    "type": "tool_call",
                },
            ],
        )
    ]

    result = run_tools(state)

    assert profile_calls == []
    assert result["status"] == "no_result"
    assert set(result["artifacts"]) == {"match"}


def test_pick_full_rejects_profile_without_match_even_with_forged_target(monkeypatch):
    profile_calls = []

    def profile_tool(arguments):
        profile_calls.append(arguments)
        return json.dumps(
            {
                "agent": "profile_analysis",
                "status": "success",
                "target_user_id": arguments.get("target_user_id"),
            }
        )

    monkeypatch.setattr(
        supervisor,
        "supervisor_tool_map",
        {"call_profile_analysis_agent": FakeTool(profile_tool)},
    )
    state = supervisor_state(
        "call_profile_analysis_agent",
        arguments={"target_user_id": 999},
        action="pick_full",
    )

    result = run_tools(state)

    assert profile_calls == []
    assert result["status"] == "failed"
    assert result["error"]["error_code"] == "DEPENDENCY_NOT_READY"
    assert result["artifacts"]["profile_analysis"]["status"] == "fatal_error"
    assert result["artifacts"]["profile_analysis"]["error_code"] == "DEPENDENCY_NOT_READY"


def test_unknown_worker_exception_becomes_structured_fatal_error(monkeypatch):
    def failing_tool(_arguments):
        raise RuntimeError("下游服务超时")

    monkeypatch.setattr(
        supervisor,
        "supervisor_tool_map",
        {"call_match_agent": FakeTool(failing_tool)},
    )

    result = run_tools(supervisor_state("call_match_agent"))

    failure = result["artifacts"]["match"]
    assert failure["status"] == "fatal_error"
    assert failure["error_code"] == "TOOL_FAILED"
    assert failure["error"] == "服务调用失败，请稍后重试"
    assert result["error"] == failure
    assert result["retry_count"] == 0
    assert result["status"] == "failed"
    assert result["completed_steps"] == []


def test_unknown_worker_status_becomes_fatal_error(monkeypatch):
    monkeypatch.setattr(
        supervisor,
        "supervisor_tool_map",
        {
            "call_match_agent": FakeTool(
                lambda _args: json.dumps({"agent": "match", "status": "maybe"})
            )
        },
    )

    result = run_tools(supervisor_state("call_match_agent"))

    assert result["artifacts"]["match"]["status"] == "fatal_error"
    assert result["artifacts"]["match"]["error_code"] == "INVALID_AGENT_STATUS"
    assert result["status"] == "failed"


def test_research_agent_adapts_mcp_chat_result(monkeypatch):
    import app.agents.ai_chat_agent as ai_chat

    captured = {}

    async def fake_invoke(state):
        captured.update(state)
        return {"final_response": "校园摄影活动将于本周举行"}

    monkeypatch.setattr(ai_chat, "ai_chat_app", SimpleNamespace(ainvoke=fake_invoke))

    raw = asyncio.run(
        supervisor.call_research_agent.ainvoke(
            {"user_id": 7, "query": "查询校园摄影活动热点"}
        )
    )
    payload = json.loads(raw)

    assert payload["agent"] == "research"
    assert payload["status"] == "success"
    assert payload["evidence"] == "校园摄影活动将于本周举行"
    assert captured["user_id"] == 7


def test_step_budget_stops_before_invoking_another_worker(monkeypatch):
    calls = []

    def must_not_run(arguments):
        calls.append(arguments)
        raise AssertionError("达到步数上限后不应继续调用子 Agent")

    monkeypatch.setattr(
        supervisor,
        "supervisor_tool_map",
        {"call_match_agent": FakeTool(must_not_run)},
    )
    state = supervisor_state(
        "call_match_agent",
        step_count=2,
        max_steps=2,
    )

    result = run_tools(state)

    assert calls == []
    assert result["step_count"] == 2
    assert result["status"] == "budget_exceeded"
    assert result["error"]["error_code"] == "STEP_BUDGET_EXCEEDED"


def test_action_allowlist_rejects_unrelated_worker(monkeypatch):
    calls = []

    monkeypatch.setattr(
        supervisor,
        "supervisor_tool_map",
        {"call_match_agent": FakeTool(lambda args: calls.append(args))},
    )

    result = run_tools(
        supervisor_state("call_match_agent", action="write")
    )

    assert calls == []
    assert result["status"] == "failed"
    assert result["error"]["status"] == "fatal_error"
    assert result["error"]["error_code"] == "TOOL_NOT_ALLOWED"
    assert result["completed_steps"] == []


@pytest.mark.parametrize("status", ["failed", "budget_exceeded", "no_result"])
def test_should_continue_after_tools_ends_terminal_statuses(status):
    state = supervisor_state("call_match_agent")
    state["status"] = status

    assert supervisor.should_continue_after_tools(state) == "extract_result"


def test_should_continue_after_tools_returns_to_supervisor_while_running():
    state = supervisor_state("call_match_agent", step_count=1, max_steps=8)
    state["status"] = "running"

    assert supervisor.should_continue_after_tools(state) == "agent"


def test_extract_result_aggregates_pick_full_artifacts_when_llm_output_is_invalid():
    artifacts = {
        "match": {
            "agent": "match",
            "status": "success",
            "best_match": {"user_id": 23, "bottle_id": 12},
        },
        "profile_analysis": {
            "agent": "profile_analysis",
            "status": "success",
            "user_id": 7,
            "target_user_id": 23,
            "compatibility": "有共同兴趣",
        },
        "conversation_guide": {
            "agent": "conversation_guide",
            "status": "success",
            "conversation_status": "new",
        },
        "ice_break": {
            "agent": "ice_break",
            "status": "success",
            "user_id": 7,
            "target_user_id": 23,
            "hints": ["你也喜欢摄影吗？"],
            "scenario": "ice_break",
        },
        "guard": {
            "agent": "guard",
            "status": "success",
            "passed": True,
            "reason": "审核通过",
            "risk_level": "low",
        },
    }
    state = supervisor_state("call_ice_break_agent", artifacts=artifacts)
    state["messages"] = [AIMessage(content="这不是合法 JSON")]
    state["completed_steps"] = ["match", "profile_analysis", "ice_break"]

    result = supervisor.extract_result(state)
    final_result = json.loads(result["final_result"])

    assert final_result["action"] == "pick_full"
    assert final_result["result"] == artifacts
    assert result["guard_passed"] is True
    assert result["guard_reason"] == ""
