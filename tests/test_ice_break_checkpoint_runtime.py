"""检查点运行时的真实 SQLite 生命周期与会话锁取消行为。"""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import app.agents.ice_break_agent as ice_break
import app.ice_break_checkpoint as runtime


@pytest.fixture
def checkpoint_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setattr(runtime.settings, "ICE_BREAK_CHECKPOINT_PATH", str(tmp_path / "state.sqlite"))
    monkeypatch.setattr(runtime, "_ice_break_app", None)
    monkeypatch.setattr(runtime, "_checkpoint_resources", None)
    monkeypatch.setattr(runtime, "_lifecycle_lock", asyncio.Lock())
    monkeypatch.setattr(runtime, "_thread_locks", {})
    yield runtime
    assert runtime._ice_break_app is None
    assert runtime._checkpoint_resources is None
    assert runtime._thread_locks == {}


def test_runtime_reopen_retains_recent_hints_and_close_is_idempotent(checkpoint_runtime, monkeypatch):
    model_inputs = []

    def reply(messages):
        model_inputs.append(list(messages))
        round_number = len(model_inputs)
        return AIMessage(content=json.dumps({
            "hints": [f"第{round_number}轮建议{number}" for number in range(1, 4)],
            "scenario": "ice_break",
        }, ensure_ascii=False))

    async def profile(_target_user_id):
        return json.dumps({"昵称": "测试同学", "爱好": "篮球"}, ensure_ascii=False)

    async def history(_user_id, _target_user_id):
        return json.dumps({"场景": "主动破冰", "聊天记录": "暂无聊天"}, ensure_ascii=False)

    monkeypatch.setattr(ice_break, "_get_llm", lambda: RunnableLambda(reply))
    monkeypatch.setattr(ice_break, "_query_user_profile", profile)
    monkeypatch.setattr(ice_break, "_query_chat_history", history)
    thread_config = {"configurable": {"thread_id": "ice-break:7:8"}}

    def turn(request_id):
        return {
            "user_id": 7, "target_user_id": 8, "request_id": request_id,
            "messages": [], "hints": [],
        }

    async def run():
        with pytest.raises(RuntimeError, match="未初始化"):
            checkpoint_runtime.get_ice_break_app()
        await checkpoint_runtime.init_ice_break_checkpoint()
        first_graph = checkpoint_runtime.get_ice_break_app()
        try:
            await checkpoint_runtime.init_ice_break_checkpoint()
            assert checkpoint_runtime.get_ice_break_app() is first_graph
            first = await first_graph.ainvoke(turn("request-1"), thread_config)
        finally:
            await checkpoint_runtime.close_ice_break_checkpoint()

        with pytest.raises(RuntimeError, match="未初始化"):
            checkpoint_runtime.get_ice_break_app()
        await checkpoint_runtime.close_ice_break_checkpoint()

        await checkpoint_runtime.init_ice_break_checkpoint()
        try:
            second_graph = checkpoint_runtime.get_ice_break_app()
            assert second_graph is not first_graph
            saved = await second_graph.aget_state(thread_config)
            assert saved.values["recent_hints"] == first["hints"]
            result = await second_graph.ainvoke(turn("request-2"), thread_config)
            assert result["recent_hints"] == [*first["hints"], *result["hints"]]
            second_prompt = "\n".join(str(message.content) for message in model_inputs[1])
            assert all(hint in second_prompt for hint in first["hints"])
        finally:
            await checkpoint_runtime.close_ice_break_checkpoint()

    asyncio.run(run())


@pytest.mark.parametrize("failure_stage", ["setup", "compile"])
def test_startup_failure_closes_real_sqlite_connection(
    checkpoint_runtime, monkeypatch, failure_stage
):
    savers = []
    original_factory = runtime.AsyncSqliteSaver.from_conn_string

    @asynccontextmanager
    async def capture_saver(path):
        async with original_factory(path) as saver:
            savers.append(saver)
            yield saver

    monkeypatch.setattr(runtime.AsyncSqliteSaver, "from_conn_string", capture_saver)

    async def fail_setup(_saver):
        raise RuntimeError("模拟 setup 失败")

    def fail_compile(**_kwargs):
        raise RuntimeError("模拟 compile 失败")

    if failure_stage == "setup":
        monkeypatch.setattr(runtime.AsyncSqliteSaver, "setup", fail_setup)
    else:
        monkeypatch.setattr(ice_break, "ice_break_graph", SimpleNamespace(compile=fail_compile))

    async def run():
        with pytest.raises(RuntimeError, match=failure_stage):
            await checkpoint_runtime.init_ice_break_checkpoint()
        assert checkpoint_runtime._ice_break_app is None
        assert checkpoint_runtime._checkpoint_resources is None
        assert len(savers) == 1
        # 真连接确实已关闭，不能继续执行本地查询。
        with pytest.raises(ValueError, match="no active connection"):
            await savers[0].conn.execute("SELECT 1")
        await checkpoint_runtime.close_ice_break_checkpoint()

    asyncio.run(run())


def test_cancelled_lock_waiter_keeps_owner_lock_and_reclaims_last_reference(checkpoint_runtime):
    async def run():
        waiter_started = asyncio.Event()
        waiter_entered = asyncio.Event()
        successor_started = asyncio.Event()
        successor_entered = asyncio.Event()

        async def waiter():
            waiter_started.set()
            async with checkpoint_runtime.ice_break_thread_lock("same-chat"):
                waiter_entered.set()

        async def successor():
            successor_started.set()
            async with checkpoint_runtime.ice_break_thread_lock("same-chat"):
                successor_entered.set()

        async with checkpoint_runtime.ice_break_thread_lock("same-chat"):
            entry = checkpoint_runtime._thread_locks["same-chat"]
            blocked = asyncio.create_task(waiter())
            await waiter_started.wait()
            assert entry.users == 2
            assert not waiter_entered.is_set()

            blocked.cancel()
            with pytest.raises(asyncio.CancelledError):
                await blocked
            assert checkpoint_runtime._thread_locks["same-chat"] is entry
            assert entry.users == 1
            assert entry.lock.locked()

            next_request = asyncio.create_task(successor())
            await successor_started.wait()
            assert entry.users == 2
            assert not successor_entered.is_set()

        await asyncio.wait_for(next_request, timeout=1)
        assert successor_entered.is_set()
        assert "same-chat" not in checkpoint_runtime._thread_locks

        # 相同编号再次使用时，分配全新的锁，且退出后仍会回收。
        async with checkpoint_runtime.ice_break_thread_lock("same-chat"):
            assert checkpoint_runtime._thread_locks["same-chat"] is not entry
        assert checkpoint_runtime._thread_locks == {}

    asyncio.run(run())


def test_cancelled_startup_closes_connection_and_can_retry(checkpoint_runtime, monkeypatch):
    savers = []
    original_factory = runtime.AsyncSqliteSaver.from_conn_string
    original_setup = runtime.AsyncSqliteSaver.setup

    @asynccontextmanager
    async def capture_saver(path):
        async with original_factory(path) as saver:
            savers.append(saver)
            yield saver

    monkeypatch.setattr(runtime.AsyncSqliteSaver, "from_conn_string", capture_saver)

    async def run():
        setup_started = asyncio.Event()
        never_complete = asyncio.Event()

        async def blocked_setup(_saver):
            setup_started.set()
            await never_complete.wait()

        monkeypatch.setattr(runtime.AsyncSqliteSaver, "setup", blocked_setup)
        startup = asyncio.create_task(checkpoint_runtime.init_ice_break_checkpoint())
        await setup_started.wait()
        startup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await startup
        assert checkpoint_runtime._ice_break_app is None
        assert checkpoint_runtime._checkpoint_resources is None
        with pytest.raises(ValueError, match="no active connection"):
            await savers[0].conn.execute("SELECT 1")

        monkeypatch.setattr(runtime.AsyncSqliteSaver, "setup", original_setup)
        await checkpoint_runtime.init_ice_break_checkpoint()
        try:
            assert checkpoint_runtime.get_ice_break_app() is not None
            assert len(savers) == 2
        finally:
            await checkpoint_runtime.close_ice_break_checkpoint()

    asyncio.run(run())


def test_different_thread_lock_can_progress_while_another_thread_is_locked(checkpoint_runtime):
    async def run():
        entered = asyncio.Event()

        async def independent_request():
            async with checkpoint_runtime.ice_break_thread_lock("chat-b"):
                entered.set()

        async with checkpoint_runtime.ice_break_thread_lock("chat-a"):
            task = asyncio.create_task(independent_request())
            await asyncio.wait_for(task, timeout=1)
            assert entered.is_set()
            assert "chat-b" not in checkpoint_runtime._thread_locks
            assert checkpoint_runtime._thread_locks["chat-a"].lock.locked()
        assert checkpoint_runtime._thread_locks == {}

    asyncio.run(run())
