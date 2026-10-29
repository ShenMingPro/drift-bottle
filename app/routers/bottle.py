import json
import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.agents.supervisor_agent import create_supervisor_state, supervisor_app
from app.ai_content_proof import verify_ai_content_proof
from app.auth_jwt import get_current_user  # JWT认证依赖项
from app.database import get_database
from app.models.bottle import PickResponse, WriteRequest, WriteResponse
from app.services.agent_responses import ensure_agent_succeeded, invoke_agent
from app.services.chances import consume_chance
from app.services.picks import commit_pick
from app.services.rewards import claim_write_bonus

router = APIRouter()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _candidate_ids(candidate: dict) -> tuple[int, int]:
    """先校验模型返回的业务标识，避免扣次后才发现结果无效。"""
    try:
        values = [candidate["bottle_id"], candidate["user_id"]]
        if any(isinstance(value, bool) or not isinstance(value, (int, str)) for value in values):
            raise ValueError("匹配标识类型无效")
        bottle_id, user_id = (int(value) for value in values)
        if bottle_id <= 0 or user_id <= 0:
            raise ValueError("匹配标识必须为正数")
        return bottle_id, user_id
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="匹配服务返回了无效结果，请稍后重试") from exc


@router.get("/universities")
async def get_universities():
    """公开接口：获取大学列表（无需认证）"""
    data_path = os.path.join(PROJECT_ROOT, "data", "universities.json")
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        raise HTTPException(status_code=500, detail="读取大学数据失败")


@router.post("/write", response_model=WriteResponse)
async def write_bottle(req: WriteRequest, db=Depends(get_database), current_user_id: int = Depends(get_current_user)):
    """写纸团接口（需JWT认证）：校验token中的用户ID与请求中的userId一致

    安全策略：从JWT令牌中提取的current_user_id必须与请求体中的userId一致，
    防止用户冒充他人写纸团。
    """
    # 安全校验：JWT中的用户ID必须与请求中的userId一致
    if current_user_id != req.userId:
        raise HTTPException(status_code=403, detail="无权操作：用户身份不匹配")

    content = req.content.strip()
    is_verified_ai_content = bool(req.aiProof) and verify_ai_content_proof(
        req.aiProof,
        req.userId,
        content,
    )

    if not is_verified_ai_content:
        supervisor_state = {
            "user_id": req.userId,
            "action": "write",
            "content": content,
            "target_user_id": 0,
            "target_university": "",
            "target_city": "",
            "target_province": "",
            "content_type": "bottle",
            "messages": [],
            "final_result": "",
            "guard_passed": False,
            "guard_reason": "",
        }
        supervisor_result = await invoke_agent(supervisor_app, supervisor_state)

        if not supervisor_result.get("guard_passed", False):  # fail-closed：默认拒绝
            raise HTTPException(status_code=400, detail=supervisor_result.get("guard_reason", "内容审核未通过"))

    bonus_used = 0
    bonus_remaining = 3
    # 模型审核已结束，再开启短事务，额度/奖励与纸团写入共同提交。
    async with db.session(transaction=True) as cursor:
        if req.useBonus:
            bonus_used = await claim_write_bonus(cursor, req.userId, require_write=False)
            bonus_remaining = 3 - bonus_used
        else:
            await consume_chance(cursor, req.userId, "write_chances")
        await cursor.execute(
            "INSERT INTO bottles (user_id, content) VALUES (%s, %s)",
            [req.userId, content],
        )
    return WriteResponse(
        success=True,
        message="投放成功",
        writeBonusUsed=bonus_used,
        writeBonusRemaining=bonus_remaining,
    )


@router.get("/pick", response_model=PickResponse)
async def pick_bottle(
    userId: int = Query(...),
    university: str = Query(""),
    city: str = Query(""),
    province: str = Query(""),
    db=Depends(get_database),
    current_user_id: int = Depends(get_current_user),
):
    """旧版单步拾纸团流程：只经由Supervisor调用MatchAgent。"""
    if current_user_id != userId:
        raise HTTPException(status_code=403, detail="无权操作：用户身份不匹配")

    supervisor_state = {
        "user_id": userId,
        "action": "pick",
        "content": "",
        "target_user_id": 0,
        "target_university": university,
        "target_city": city,
        "target_province": province,
        "content_type": "",
        "messages": [],
        "final_result": "",
        "guard_passed": True,
        "guard_reason": "",
    }
    supervisor_result = await invoke_agent(supervisor_app, supervisor_state)

    try:
        final_data = json.loads(supervisor_result.get("final_result", ""))
        if not isinstance(final_data, dict):
            raise ValueError("匹配结果必须是对象")
        match_data = final_data.get("result", final_data)
        if isinstance(match_data, str):
            match_data = json.loads(match_data)
        if not isinstance(match_data, dict) or "best_match" not in match_data:
            raise ValueError("匹配结果缺少候选状态")
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="匹配服务返回了无效结果，请稍后重试") from exc
    ensure_agent_succeeded(match_data)

    best_match = match_data.get("best_match")
    if best_match is None:
        return PickResponse(hasBottle=False)
    if not isinstance(best_match, dict) or not best_match:
        raise HTTPException(status_code=502, detail="匹配服务返回了无效结果，请稍后重试")

    bottle_id, target_user_id = _candidate_ids(best_match)
    try:
        response = PickResponse(
            hasBottle=True,
            bottleUserId=target_user_id,
            content=best_match.get("content", ""),
            createdAt=best_match.get("created_at") or "",
            nickname=best_match.get("nickname") or "",
            gender=best_match.get("gender") or "",
            age=best_match.get("age") or 0,
            hobbies=best_match.get("hobbies") or "",
            contact_type=best_match.get("contact_type") or "",
            contact_value=best_match.get("contact_value") or "",
            province=best_match.get("province") or "",
            city=best_match.get("city") or "",
            university=best_match.get("university") or "",
            matchLevel=match_data.get("match_level") or "",
            matchReason=match_data.get("match_reason") or "",
            matchScore=match_data.get("match_score") or 0.0,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=502, detail="匹配服务返回了无效结果，请稍后重试") from exc
    await commit_pick(
        db, userId, bottle_id, request_key=f"legacy:{uuid.uuid4().hex}"
    )
    return response


@router.get("/pick/full")
async def pick_full_bottle(
    userId: int = Query(...),
    university: str = Query(""),
    city: str = Query(""),
    province: str = Query(""),
    db=Depends(get_database),
    current_user_id: int = Depends(get_current_user),
):
    """完整匹配流程：由Supervisor协调匹配、画像分析和破冰建议。"""
    if current_user_id != userId:
        raise HTTPException(status_code=403, detail="无权操作：用户身份不匹配")

    supervisor_state = create_supervisor_state(
        user_id=userId,
        action="pick_full",
        university=university,
        city=city,
        province=province,
        content_type="message",
    )
    config = {
        "configurable": {
            "thread_id": f"pick-full:{userId}:{uuid.uuid4().hex}",
        }
    }
    supervisor_result = await invoke_agent(supervisor_app, supervisor_state, config)

    try:
        final_data = json.loads(supervisor_result.get("final_result", "{}"))
        if not isinstance(final_data, dict):
            raise ValueError("完整匹配结果必须是对象")
        result_data = final_data.get("result", {})
        if not isinstance(result_data, dict):
            raise ValueError("完整匹配结果必须是对象")
    except (ValueError, TypeError):
        raise HTTPException(status_code=502, detail="完整匹配流程返回了无效结果")

    for artifact in result_data.values():
        if isinstance(artifact, dict):
            ensure_agent_succeeded(artifact)
    match_data = result_data.get("match", {})
    if not isinstance(match_data, dict) or "best_match" not in match_data:
        raise HTTPException(status_code=502, detail="完整匹配流程返回了无效结果")
    best_match = match_data["best_match"]
    if best_match is not None:
        if not isinstance(best_match, dict) or not best_match:
            raise HTTPException(status_code=502, detail="完整匹配流程返回了无效结果")
        bottle_id, _ = _candidate_ids(best_match)

    if result_data.get("ice_break") and not supervisor_result.get("guard_passed", False):
        return JSONResponse(
            status_code=422,
            content={
                "status": "rejected",
                "hasMatch": bool(best_match),
                "reason": supervisor_result.get("guard_reason", "破冰建议未通过审核"),
                "match": match_data,
                "completedSteps": supervisor_result.get("completed_steps", []),
            },
        )

    if best_match and supervisor_result.get("status") == "success":
        await commit_pick(
            db, userId, bottle_id, request_key=f"full:{uuid.uuid4().hex}"
        )

    return JSONResponse(
        content={
            "status": supervisor_result.get("status", "success"),
            "hasMatch": bool(best_match),
            "match": match_data,
            "profileAnalysis": result_data.get("profile_analysis"),
            "conversationGuide": result_data.get("conversation_guide"),
            "iceBreak": result_data.get("ice_break"),
            "completedSteps": supervisor_result.get("completed_steps", []),
            "stepCount": supervisor_result.get("step_count", 0),
        },
        headers={"Cache-Control": "no-store"},
    )
