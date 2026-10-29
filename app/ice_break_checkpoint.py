"""人与人聊天 AI 辅助的检查点生命周期，以及会话内串行执行。"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph

from app.config import settings


_ice_break_app: CompiledStateGraph | None = None
_checkpoint_resources: AsyncExitStack | None = None
_lifecycle_lock = asyncio.Lock()


def _checkpoint_path() -> Path:
    configured_path = settings.ICE_BREAK_CHECKPOINT_PATH.strip()
    if not configured_path:
        raise ValueError("ICE_BREAK_CHECKPOINT_PATH 不能为空")
    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    return path.resolve()


async def init_ice_break_checkpoint() -> None:
    """打开持久化存储并编译 AI 辅助图；初始化失败时终止应用启动。"""
    global _ice_break_app, _checkpoint_resources

    async with _lifecycle_lock:
        if _ice_break_app is not None:
            return

        path = _checkpoint_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        resources = AsyncExitStack()
        try:
            saver = await resources.enter_async_context(
                AsyncSqliteSaver.from_conn_string(str(path))
            )
            await saver.setup()
            from app.agents.ice_break_agent import ice_break_graph

            chat_app = ice_break_graph.compile(checkpointer=saver)
        except BaseException:
            await resources.aclose()
            raise

        _checkpoint_resources = resources
        _ice_break_app = chat_app


def get_ice_break_app() -> CompiledStateGraph:
    """返回由 FastAPI 生命周期初始化的持久化 AI 辅助图。"""
    if _ice_break_app is None:
        raise RuntimeError("聊天 AI 辅助检查点未初始化，请通过应用生命周期启动服务")
    return _ice_break_app


async def close_ice_break_checkpoint() -> None:
    """关闭 SQLite 连接；重复关闭安全。"""
    global _ice_break_app, _checkpoint_resources

    async with _lifecycle_lock:
        resources = _checkpoint_resources
        _ice_break_app = None
        _checkpoint_resources = None
        if resources is not None:
            await resources.aclose()


@dataclass
class _ThreadLock:
    lock: asyncio.Lock
    users: int = 0


_thread_locks: dict[str, _ThreadLock] = {}


@asynccontextmanager
async def ice_break_thread_lock(thread_id: str) -> AsyncIterator[None]:
    """串行化同一会话的读写，等待者取消或最后一个请求结束后移除锁。"""
    # 查找与引用计数之间没有 await，在同一事件循环内不会被其他请求打断。
    entry = _thread_locks.get(thread_id)
    if entry is None:
        entry = _ThreadLock(asyncio.Lock())
        _thread_locks[thread_id] = entry
    entry.users += 1
    try:
        async with entry.lock:
            yield
    finally:
        entry.users -= 1
        if entry.users == 0 and _thread_locks.get(thread_id) is entry:
            del _thread_locks[thread_id]
