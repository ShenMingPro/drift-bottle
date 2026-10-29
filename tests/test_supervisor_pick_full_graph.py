import asyncio
import json

from langchain_core.messages import AIMessage

import app.agents.supervisor_agent as supervisor


class FakeTool:
    def __init__(self, handler):
        self.func = handler
        self.coroutine = None

    def invoke(self, arguments):
        return self.func(arguments)


class PlannedModel:
    def __init__(self):
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, _messages):
        self.calls += 1
        plans = {
            1: ("call_match_agent", {}),
            2: ("call_profile_analysis_agent", {"target_user_id": 999}),
            3: ("call_conversation_guide_agent", {"target_user_id": 999}),
            4: ("call_ice_break_agent", {"target_user_id": 999}),
            5: ("call_guard_agent", {}),
        }
        if self.calls in plans:
            name, args = plans[self.calls]
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": name,
                        "args": args,
                        "id": f"call-{self.calls}",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="这不是合法 JSON")


def test_pick_full_graph_uses_previous_artifacts_for_next_agents(monkeypatch):
    model = PlannedModel()
    calls = []

    def match(_args):
        calls.append(("match", _args))
        return json.dumps(
            {
                "agent": "match",
                "status": "success",
                "best_match": {"bottle_id": 12, "user_id": 23},
                "match_score": 0.82,
            }
        )

    def profile(args):
        calls.append(("profile", args))
        return json.dumps(
            {
                "agent": "profile_analysis",
                "status": "success",
                "target_user_id": args["target_user_id"],
                "compatibility": "有共同兴趣",
            }
        )

    def guide(args):
        calls.append(("guide", args))
        return json.dumps(
            {
                "agent": "conversation_guide",
                "status": "success",
                "target_user_id": args["target_user_id"],
                "conversation_status": "new",
            }
        )

    def ice_break(args):
        calls.append(("ice_break", args))
        return json.dumps(
            {
                "agent": "ice_break",
                "status": "success",
                "target_user_id": args["target_user_id"],
                "hints": ["你也喜欢摄影吗？"],
            }
        )

    def guard(args):
        calls.append(("guard", args))
        return json.dumps(
            {
                "agent": "guard",
                "status": "success",
                "passed": True,
                "reason": "审核通过",
            }
        )

    monkeypatch.setattr(supervisor, "_get_llm", lambda: model)
    monkeypatch.setattr(
        supervisor,
        "supervisor_tool_map",
        {
            "call_match_agent": FakeTool(match),
            "call_profile_analysis_agent": FakeTool(profile),
            "call_conversation_guide_agent": FakeTool(guide),
            "call_ice_break_agent": FakeTool(ice_break),
            "call_guard_agent": FakeTool(guard),
        },
    )

    state = supervisor.create_supervisor_state(
        user_id=7,
        action="pick_full",
        university="测试大学",
        city="测试市",
        province="测试省",
    )
    result = asyncio.run(supervisor.supervisor_app.ainvoke(state))

    assert [name for name, _ in calls] == [
        "match",
        "profile",
        "guide",
        "ice_break",
        "guard",
    ]
    assert calls[1][1]["target_user_id"] == 23
    assert calls[2][1]["target_user_id"] == 23
    assert calls[3][1]["target_user_id"] == 23
    assert calls[3][1]["analysis_context"] == {
        "compatibility": "有共同兴趣", "conversation_status": "new",
    }
    assert calls[4][1]["content"] == "你也喜欢摄影吗？"
    assert result["status"] == "success"
    assert result["completed_steps"] == [
        "match",
        "profile_analysis",
        "conversation_guide",
        "ice_break",
        "guard",
    ]
    final_result = json.loads(result["final_result"])
    assert final_result["result"]["match"]["best_match"]["user_id"] == 23
    assert result["guard_passed"] is True
