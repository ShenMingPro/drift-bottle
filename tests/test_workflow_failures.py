"""审核服务故障与内容不合规是不同的终止状态。"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver

import app.workflows.pick_workflow as workflow
from app.routers.pick_runs import _payload
from test_pick_workflow import config, install_fakes, state


@pytest.mark.parametrize("failure", ["timeout", "failed", "bad_boolean", "invalid_root", "partial_timeout"])
def test_guard_technical_failure_stops_graph_without_approval(monkeypatch, failure):
    install_fakes(monkeypatch)
    calls = []

    async def broken_guard(content, user_id):
        calls.append(content)
        if failure == "partial_timeout" and len(calls) == 1:
            return {"status": "success", "passed": True}
        if failure in {"timeout", "partial_timeout"}:
            raise TimeoutError("internal-secret")
        if failure == "failed":
            return {"status": "retryable_error", "error": "internal-secret", "passed": True}
        if failure == "bad_boolean":
            return {"status": "success", "passed": "false"}
        return []

    monkeypatch.setattr(workflow, "run_guard", broken_guard)

    async def run():
        graph = workflow.build_pick_workflow().compile(checkpointer=InMemorySaver())
        result = await graph.ainvoke(state(), config())
        assert result["status"] == "failed"
        assert result["approved_hints"] == []
        assert result["ice_break"]["hints"] == []
        assert result["guard"]["status"] == "failed"
        assert result["guard"]["passed"] is False
        assert "internal-secret" not in result["error"]
        assert "__interrupt__" not in result
        assert (await graph.aget_state(config())).next == ()

    asyncio.run(run())


def test_workflow_tool_timeout_cancels_call(monkeypatch):
    canceled = []

    async def blocked(_arguments):
        try:
            await asyncio.Event().wait()
        finally:
            canceled.append(True)

    monkeypatch.setattr(workflow.settings, "AGENT_CALL_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(TimeoutError):
        asyncio.run(workflow._invoke_tool(SimpleNamespace(ainvoke=blocked), {}))
    assert canceled == [True]


@pytest.mark.parametrize("step", ["match", "profile_analysis", "conversation_guide", "ice_break"])
def test_workflow_exception_details_do_not_escape_in_public_payload(monkeypatch, step):
    install_fakes(monkeypatch)

    async def broken(*_args, **_kwargs):
        raise RuntimeError("private-provider-sentinel")

    monkeypatch.setattr(workflow, f"run_{step}", broken)

    async def run():
        graph = workflow.build_pick_workflow().compile(checkpointer=InMemorySaver())
        await graph.ainvoke(state(), config())
        snapshot = await graph.aget_state(config())
        payload = _payload(snapshot, dict(snapshot.values), "run-1")
        assert payload["status"] == "failed"
        assert "private-provider-sentinel" not in json.dumps(payload, ensure_ascii=False)
        assert snapshot.next == ()

    asyncio.run(run())
