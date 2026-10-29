"""可恢复拾取任务 API。"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from langgraph.types import Command

from app.auth_jwt import get_current_user
from app.database import get_database
from app.models.pick_run import PickRunRequest, PickRunResumeRequest
from app.pick_checkpoint import get_pick_workflow_app, pick_run_lock
from app.services.picks import commit_pick, get_pick_commit

router = APIRouter()
logger = logging.getLogger("pick_runs")


def _config(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": f"pick-run:{run_id}"}}


async def _owned_snapshot(graph: Any, config: dict[str, Any], user_id: int):
    snapshot = await graph.aget_state(config)
    values = dict(snapshot.values or {})
    if not values or values.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="拾取任务不存在")
    return snapshot, values


def _interrupt_value(snapshot: Any) -> dict[str, Any] | None:
    for task in getattr(snapshot, "tasks", ()) or ():
        for item in getattr(task, "interrupts", ()) or ():
            value = getattr(item, "value", None)
            if isinstance(value, dict):
                return value
    return None


def _payload(snapshot: Any, values: dict[str, Any], run_id: str) -> dict[str, Any]:
    next_nodes = tuple(getattr(snapshot, "next", ()) or ())
    status = str(values.get("status", "running"))
    if "approval" in next_nodes:
        status = "awaiting_approval"

    approved_hints = set(values.get("approved_hints", []))
    ice_break = values.get("ice_break")
    if isinstance(ice_break, dict):
        ice_break = dict(ice_break)
        ice_break["hints"] = [
            hint for hint in ice_break.get("hints", []) if hint in approved_hints
        ]

    return {
        "runId": run_id,
        "status": status,
        "canResume": "approval" in next_nodes,
        "match": values.get("match"),
        "profileAnalysis": values.get("profile_analysis"),
        "conversationGuide": values.get("conversation_guide"),
        "iceBreak": ice_break,
        "guard": values.get("guard"),
        "approvedHints": values.get("approved_hints", []),
        "selectedHint": values.get("selected_hint", ""),
        "approval": _interrupt_value(snapshot),
        "trace": values.get("trace", []),
        "error": values.get("error"),
    }


async def _record_pick_once(db: Any, user_id: int, bottle_id: int, *, run_id: str) -> bool:
    committed = await commit_pick(db, user_id, bottle_id, request_key=f"run:{run_id}")
    return committed.created


async def _snapshot_response(
    graph: Any, config: dict[str, Any], run_id: str, user_id: int, db=None
):
    snapshot, values = await _owned_snapshot(graph, config, user_id)
    payload = _payload(snapshot, values, run_id)
    payload["commitStatus"] = "not_requested"
    payload["pickRecorded"] = False
    if values.get("status") == "approved":
        bottle_id = ((values.get("match") or {}).get("best_match") or {}).get("bottle_id")
        committed = await get_pick_commit(db, user_id, f"run:{run_id}", bottle_id=bottle_id)
        payload["commitStatus"] = "committed" if committed else "pending"
        payload["pickRecorded"] = committed is not None
    return JSONResponse(
        content=payload,
        headers={"Cache-Control": "no-store"},
    )


@router.post("/pick/runs")
async def start_pick_run(
    req: PickRunRequest,
    current_user_id: int = Depends(get_current_user),
):
    if req.userId != current_user_id:
        raise HTTPException(status_code=403, detail="无权操作：用户身份不匹配")

    run_id = uuid4().hex
    config = _config(run_id)
    graph = get_pick_workflow_app()
    state = {
        "run_id": run_id,
        "user_id": req.userId,
        "university": req.university,
        "city": req.city,
        "province": req.province,
        "status": "running",
        "error": None,
        "trace": [],
    }
    try:
        async with pick_run_lock(run_id):
            await graph.ainvoke(state, config)
    except Exception as exc:
        logger.exception("拾取工作流启动失败 run_id=%s", run_id)
        raise HTTPException(status_code=502, detail="拾取工作流启动失败") from exc
    return await _snapshot_response(graph, config, run_id, req.userId)


@router.get("/pick/runs/{run_id}")
async def get_pick_run(
    run_id: str,
    db=Depends(get_database),
    current_user_id: int = Depends(get_current_user),
):
    graph = get_pick_workflow_app()
    return await _snapshot_response(graph, _config(run_id), run_id, current_user_id, db)


@router.post("/pick/runs/{run_id}/resume")
async def resume_pick_run(
    run_id: str,
    req: PickRunResumeRequest,
    db=Depends(get_database),
    current_user_id: int = Depends(get_current_user),
):
    if req.userId != current_user_id:
        raise HTTPException(status_code=403, detail="无权操作：用户身份不匹配")

    graph = get_pick_workflow_app()
    config = _config(run_id)
    async with pick_run_lock(run_id):
        snapshot, values = await _owned_snapshot(graph, config, current_user_id)
        next_nodes = tuple(getattr(snapshot, "next", ()) or ())

        if "approval" not in next_nodes:
            # 已完成的任务再次提交 resume 是幂等读取，不重新执行图。
            if values.get("status") != "approved":
                return await _snapshot_response(graph, config, run_id, current_user_id, db)
            if not req.approved:
                # 关闭审批窗口不能悄悄重试尚未提交的业务写入。
                return await _snapshot_response(graph, config, run_id, current_user_id, db)
            best_match = (values.get("match") or {}).get("best_match") or {}
            bottle_id = best_match.get("bottle_id")
            if bottle_id:
                await _record_pick_once(db, current_user_id, int(bottle_id), run_id=run_id)
            payload = _payload(snapshot, values, run_id)
            payload["pickRecorded"] = bool(bottle_id)
            payload["commitStatus"] = "committed" if bottle_id else "not_requested"
            return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})

        if req.approved:
            hint = (req.hint or "").strip()
            if not hint:
                raise HTTPException(status_code=400, detail="请选择一条破冰建议")
            if hint not in values.get("approved_hints", []):
                raise HTTPException(status_code=400, detail="只能选择已经通过审核的建议")

        try:
            await graph.ainvoke(
                Command(resume={"approved": req.approved, "hint": req.hint}),
                config,
            )
        except Exception as exc:
            logger.exception("拾取工作流恢复失败 run_id=%s", run_id)
            raise HTTPException(status_code=502, detail="拾取工作流恢复失败") from exc

        snapshot, values = await _owned_snapshot(graph, config, current_user_id)
        payload = _payload(snapshot, values, run_id)
        payload["pickRecorded"] = False
        payload["commitStatus"] = "not_requested"
        if values.get("status") == "approved":
            best_match = (values.get("match") or {}).get("best_match") or {}
            bottle_id = best_match.get("bottle_id")
            if bottle_id:
                await _record_pick_once(db, current_user_id, int(bottle_id), run_id=run_id)
            payload["pickRecorded"] = bool(bottle_id)
            payload["commitStatus"] = "committed" if bottle_id else "not_requested"
        return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})
