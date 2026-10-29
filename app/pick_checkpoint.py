"""可恢复拾取工作流的 SQLite 检查点生命周期。"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph

from app.config import settings


_pick_workflow_app: CompiledStateGraph | None = None
_checkpoint_resources: AsyncExitStack | None = None
_lifecycle_lock = asyncio.Lock()


@dataclass
class _RunLock:
    lock: asyncio.Lock
    users: int = 0


_run_locks: dict[str, _RunLock] = {}


def _checkpoint_path() -> Path:
    configured_path = settings.PICK_WORKFLOW_CHECKPOINT_PATH.strip()
    if not configured_path:
        raise ValueError("PICK_WORKFLOW_CHECKPOINT_PATH 不能为空")
    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    return path.resolve()


async def init_pick_workflow_checkpoint() -> None:
    """打开检查点并编译工作流；失败时让应用启动失败，避免静默降级。"""
    global _pick_workflow_app, _checkpoint_resources

    async with _lifecycle_lock:
        if _pick_workflow_app is not None:
            return

        path = _checkpoint_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        resources = AsyncExitStack()
        try:
            saver = await resources.enter_async_context(
                AsyncSqliteSaver.from_conn_string(str(path))
            )
            await saver.setup()
            from app.workflows.pick_workflow import pick_workflow_graph

            _pick_workflow_app = pick_workflow_graph.compile(checkpointer=saver)
        except BaseException:
            await resources.aclose()
            raise
        _checkpoint_resources = resources


def get_pick_workflow_app() -> CompiledStateGraph:
    if _pick_workflow_app is None:
        raise RuntimeError("拾取工作流检查点未初始化，请通过应用生命周期启动服务")
    return _pick_workflow_app


async def close_pick_workflow_checkpoint() -> None:
    """关闭 SQLite 连接；重复调用安全。"""
    global _pick_workflow_app, _checkpoint_resources

    async with _lifecycle_lock:
        resources = _checkpoint_resources
        _pick_workflow_app = None
        _checkpoint_resources = None
        if resources is not None:
            await resources.aclose()


@asynccontextmanager
async def pick_run_lock(run_id: str) -> AsyncIterator[None]:
    """同一个 run 的审批和状态读取串行执行，避免重复 resume。"""
    entry = _run_locks.get(run_id)
    if entry is None:
        entry = _RunLock(asyncio.Lock())
        _run_locks[run_id] = entry
    entry.users += 1
    try:
        async with entry.lock:
            yield
    finally:
        entry.users -= 1
        if entry.users == 0 and _run_locks.get(run_id) is entry:
            _run_locks.pop(run_id, None)
