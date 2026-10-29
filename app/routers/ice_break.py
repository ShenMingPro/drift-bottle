"""人与人聊天的 AI 破冰／回复建议：持久化状态和同请求恢复。"""

from __future__ import annotations

import json
import logging
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from langgraph.types import Overwrite

from app.auth_jwt import get_current_user
from app.ice_break_checkpoint import get_ice_break_app, ice_break_thread_lock

router = APIRouter()
logger = logging.getLogger("ice_break")


def _config(user_id: int, target_user_id: int) -> dict:
    # A 给 B 的建议与 B 给 A 的建议属于两个独立状态，不能排序双方 ID。
    return {"configurable": {"thread_id": f"ice-break:{user_id}:{target_user_id}"}}


def _check_identity(user_id: int, current_user_id: int) -> None:
    if current_user_id != user_id:
        raise HTTPException(status_code=403, detail="无权操作：用户身份不匹配")


async def _prepare_request(graph, config, user_id: int, target_user_id: int, request_id: UUID):
    snapshot = await graph.aget_state(config)
    values = snapshot.values or {}
    request_key = str(request_id)
    for completed in values.get("completed_requests", []):
        if completed.get("request_id") == request_key:
            return None, {
                "hints": completed["hints"],
                "scenario": completed["scenario"],
                "requestId": request_key,
            }
    if snapshot.next and values.get("request_id") == request_key:
        # 断流重试原请求：恢复未完成节点，复用已经成功读取的上下文。
        return None, None
    # 新的刷新重新读取最新聊天，保留近期建议，只清空本轮输入输出。
    return {
        "user_id": user_id,
        "target_user_id": target_user_id,
        "request_id": request_key,
        "messages": Overwrite([]),
        "hints": [],
        "scenario": "",
        "analysis_context": None,
    }, None


def _result_payload(result: dict, request_id: UUID) -> dict:
    return {
        "hints": result.get("hints", []),
        "scenario": result.get("scenario", "ice_break"),
        "requestId": str(request_id),
    }


@router.get("/ice-break")
async def get_ice_break_hints(
    userId: Annotated[int, Query()],
    targetUserId: Annotated[int, Query(gt=0)],
    requestId: Annotated[UUID | None, Query()] = None,
    current_user_id: int = Depends(get_current_user),
):
    _check_identity(userId, current_user_id)
    request_id = requestId or uuid4()
    config = _config(current_user_id, targetUserId)
    graph = get_ice_break_app()
    async with ice_break_thread_lock(config["configurable"]["thread_id"]):
        state, cached = await _prepare_request(graph, config, current_user_id, targetUserId, request_id)
        if cached is not None:
            result = cached
        else:
            try:
                result = _result_payload(await graph.ainvoke(state, config), request_id)
            except Exception:
                logger.exception("辅助聊天生成失败，request=%s", request_id)
                raise HTTPException(status_code=502, detail="生成暂时失败，请重试本次建议")
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


def _event(event: str, data: dict) -> dict:
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


@router.get("/ice-break/stream")
async def stream_ice_break_hints(
    userId: Annotated[int, Query()],
    targetUserId: Annotated[int, Query(gt=0)],
    requestId: Annotated[UUID | None, Query()] = None,
    current_user_id: int = Depends(get_current_user),
):
    _check_identity(userId, current_user_id)
    from sse_starlette.sse import EventSourceResponse

    request_id = requestId or uuid4()
    config = _config(current_user_id, targetUserId)
    graph = get_ice_break_app()

    async def event_generator():
        try:
            async with ice_break_thread_lock(config["configurable"]["thread_id"]):
                state, cached = await _prepare_request(graph, config, current_user_id, targetUserId, request_id)
                if cached is not None:
                    yield _event("result", cached)
                else:
                    yield _event("progress", {
                        "stage": "fetching" if state is not None else "generating",
                        "message": "正在分析对方资料和聊天记录..." if state is not None else "正在恢复本次建议生成...",
                    })
                    final_result = None
                    root_run_id = None
                    # 只执行一次图，最终结果读取同一次执行的根图结束事件。
                    async for event in graph.astream_events(state, config, version="v2"):
                        kind = event.get("event")
                        if kind == "on_chain_start" and not event.get("parent_ids"):
                            root_run_id = event.get("run_id")
                        elif kind == "on_chain_end" and event.get("run_id") == root_run_id:
                            final_result = event.get("data", {}).get("output")
                        elif kind == "on_chain_start" and event.get("name") == "generate_hints":
                            yield _event("progress", {"stage": "generating", "message": "正在为你生成破冰建议..."})
                    if not isinstance(final_result, dict):
                        raise RuntimeError("辅助聊天执行未返回最终状态")
                    yield _event("result", _result_payload(final_result, request_id))
                yield _event("done", {"requestId": str(request_id)})
        except Exception:
            logger.exception("辅助聊天流式生成失败，request=%s", request_id)
            yield _event("error", {
                "message": "生成暂时失败，请重试本次建议",
                "requestId": str(request_id),
                "retryable": True,
            })

    return EventSourceResponse(event_generator(), headers={"Cache-Control": "no-store"})
