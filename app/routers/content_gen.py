"""纸团 AI 建议：普通响应与 SSE 共用历史去重和发布凭证处理。"""

import json
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.agents.content_gen_agent import ContentGenerationError, content_gen_app
from app.ai_content_proof import create_ai_content_proof
from app.auth_jwt import get_current_user
from app.content_safety import is_deterministically_safe

router = APIRouter()
logger = logging.getLogger("content_gen")

PreviousSuggestions = Annotated[
    list[Annotated[str, Field(min_length=1, max_length=200)]],
    Query(max_length=30),
]
GENERATION_FAILURE_MESSAGE = "暂时没有生成合适的新内容，请再试一次"


class ContentGenRequest(BaseModel):
    userId: int
    previousSuggestions: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(
        default_factory=list, max_length=30,
    )


def _attach_proofs(suggestions: list[dict], user_id: int) -> list[dict]:
    """为可发布的 AI 建议附加用户和内容绑定的短期签名。"""
    signed_suggestions = []
    for suggestion in suggestions:
        if not isinstance(suggestion, dict):
            continue
        content = str(suggestion.get("content", "")).strip()
        if not content or not is_deterministically_safe(content):
            continue
        signed_suggestions.append({
            **suggestion,
            "content": content,
            "proof": create_ai_content_proof(user_id, content),
        })
    return signed_suggestions


async def _generate_suggestions(user_id: int, previous_suggestions: list[str]) -> list[dict]:
    result = await content_gen_app.ainvoke({
        "user_id": user_id,
        "messages": [],
        "suggestions": [],
        "previous_suggestions": previous_suggestions,
    })
    suggestions = _attach_proofs(result.get("suggestions", []), user_id)
    if len(suggestions) != 3:
        raise ContentGenerationError(GENERATION_FAILURE_MESSAGE)
    logger.info("content_gen_completed user_id=%s count=%s", user_id, len(suggestions))
    return suggestions


@router.get("/content-gen")
async def get_content_suggestions(
    userId: int = Query(...),
    previousSuggestions: PreviousSuggestions = [],
    current_user_id: int = Depends(get_current_user),
):
    """生成三条新建议，previousSuggestions 为本用户最近展示的至多 30 条内容。"""
    if current_user_id != userId:
        raise HTTPException(status_code=403, detail="无权操作：用户身份不匹配")
    try:
        suggestions = await _generate_suggestions(userId, previousSuggestions)
    except Exception as exc:
        logger.warning("content_gen_failed user_id=%s error_type=%s", userId, type(exc).__name__)
        raise HTTPException(status_code=503, detail=GENERATION_FAILURE_MESSAGE) from exc
    return JSONResponse(
        content={"suggestions": suggestions},
        headers={"Cache-Control": "no-store"},
    )


@router.post("/content-gen")
async def post_content_suggestions(
    request: ContentGenRequest,
    current_user_id: int = Depends(get_current_user),
):
    """页面使用 JSON 正文携带历史，避免中文内容导致查询 URL 过长。"""
    return await get_content_suggestions(request.userId, request.previousSuggestions, current_user_id)


@router.get("/content-gen/stream")
async def stream_content_suggestions(
    userId: int = Query(...),
    previousSuggestions: PreviousSuggestions = [],
    current_user_id: int = Depends(get_current_user),
):
    """SSE 依次发送 progress、result、done；生成失败仅发送 error。"""
    if current_user_id != userId:
        raise HTTPException(status_code=403, detail="无权操作：用户身份不匹配")
    from sse_starlette.sse import EventSourceResponse

    async def event_generator():
        try:
            yield {
                "event": "progress",
                "data": json.dumps({"stage": "fetching", "message": "正在分析你的资料..."}, ensure_ascii=False),
            }
            yield {
                "event": "progress",
                "data": json.dumps({"stage": "generating", "message": "正在为你生成新的纸团内容..."}, ensure_ascii=False),
            }
            suggestions = await _generate_suggestions(userId, previousSuggestions)
            yield {
                "event": "result",
                "data": json.dumps({"suggestions": suggestions}, ensure_ascii=False),
            }
            yield {"event": "done", "data": "{}"}
        except Exception as exc:
            logger.warning("content_gen_stream_failed user_id=%s error_type=%s", userId, type(exc).__name__)
            yield {
                "event": "error",
                "data": json.dumps({"message": GENERATION_FAILURE_MESSAGE}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator(), headers={"Cache-Control": "no-store"})


@router.post("/content-gen/stream")
async def post_stream_content_suggestions(
    request: ContentGenRequest,
    current_user_id: int = Depends(get_current_user),
):
    return await stream_content_suggestions(request.userId, request.previousSuggestions, current_user_id)
