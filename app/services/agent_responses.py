"""把 Agent 技术失败转换为安全的 HTTP 响应，不重试业务写入。"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException
from httpx import TimeoutException


logger = logging.getLogger(__name__)
FAILURE_STATUSES = {"failed", "fatal_error", "retryable_error", "budget_exceeded"}


def ensure_agent_succeeded(result: Any) -> None:
    """错误信息只能用于判断状态，不能直接展示给客户端。"""
    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="AI 服务返回了无效结果，请稍后重试")
    error = result.get("error")
    status = result.get("status")
    # 旧调用方可省略状态；显式返回的未知/空状态不能被当作成功。
    if "status" in result and (
        not isinstance(status, str)
        or status not in FAILURE_STATUSES | {"success", "no_result", "rejected"}
    ):
        raise HTTPException(status_code=502, detail="AI 服务返回了无效结果，请稍后重试")
    if status not in FAILURE_STATUSES and not error:
        return

    error = error if isinstance(error, dict) else {}
    code = str(error.get("error_code", "")).upper()
    logger.warning("agent_failed status=%s error_code=%s", status, code)
    if "TIMEOUT" in code:
        raise HTTPException(status_code=504, detail="AI 服务响应超时，请稍后重试")
    if error.get("retryable") is True or status in {"retryable_error", "budget_exceeded"}:
        raise HTTPException(status_code=503, detail="AI 服务暂时不可用，请稍后重试")
    raise HTTPException(status_code=502, detail="AI 服务执行失败，请稍后重试")


async def invoke_agent(graph: Any, state: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """保留取消语义，将图执行异常和显式失败都留在业务事务之外。"""
    try:
        result = await graph.ainvoke(state) if config is None else await graph.ainvoke(state, config)
    except (TimeoutError, TimeoutException) as exc:
        logger.exception("agent_timeout action=%s", state.get("action"))
        raise HTTPException(status_code=504, detail="AI 服务响应超时，请稍后重试") from exc
    except Exception as exc:
        logger.exception("agent_invocation_failed action=%s", state.get("action"))
        raise HTTPException(status_code=502, detail="AI 服务执行失败，请稍后重试") from exc
    ensure_agent_succeeded(result)
    return result
