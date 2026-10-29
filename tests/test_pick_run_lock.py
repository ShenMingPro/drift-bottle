"""审批锁在存在等待者或取消请求时保持同一把锁。"""

import asyncio

import pytest

import app.pick_checkpoint as runtime


def test_waiting_approval_keeps_same_lock_until_last_request_finishes(monkeypatch):
    monkeypatch.setattr(runtime, "_run_locks", {})

    async def scenario():
        started = asyncio.Event()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def next_approval():
            started.set()
            async with runtime.pick_run_lock("same-run"):
                entered.set()
                await release.wait()

        async with runtime.pick_run_lock("same-run"):
            entry = runtime._run_locks["same-run"]
            waiter = asyncio.create_task(next_approval())
            await started.wait()
            assert not entered.is_set()

        # 前一个请求退出但等待者尚未恢复时，也不能删除它仍在等的锁。
        assert runtime._run_locks["same-run"] is entry
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert runtime._run_locks["same-run"] is entry
        release.set()
        await asyncio.wait_for(waiter, timeout=1)
        assert runtime._run_locks == {}

    asyncio.run(scenario())


def test_cancelled_approval_waiter_does_not_delete_owner_lock(monkeypatch):
    monkeypatch.setattr(runtime, "_run_locks", {})

    async def scenario():
        started = asyncio.Event()

        async def waiting():
            started.set()
            async with runtime.pick_run_lock("same-run"):
                raise AssertionError("等待者已经取消")

        async with runtime.pick_run_lock("same-run"):
            entry = runtime._run_locks["same-run"]
            waiter = asyncio.create_task(waiting())
            await started.wait()
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert runtime._run_locks["same-run"] is entry
            assert entry.users == 1
        assert runtime._run_locks == {}

    asyncio.run(scenario())
