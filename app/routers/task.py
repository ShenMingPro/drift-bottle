from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth_jwt import get_current_user  # JWT认证依赖项
from app.database import get_database
from app.models.task import (
    CheckinResponse,
    CheckinStatusResponse,
    InviteResponse,
    UpdateChancesRequest,
    UserChancesResponse,
    WriteBonusResponse,
)
from app.services.chances import refresh_cooldown_chances
from app.services.rewards import claim_checkin, claim_invite, claim_write_bonus

router = APIRouter()


@router.get("/user-chances", response_model=UserChancesResponse)
async def get_user_chances(userId: int = Query(...), db=Depends(get_database), current_user_id: int = Depends(get_current_user)):
    """获取用户次数接口（需JWT认证）"""
    if current_user_id != userId:
        raise HTTPException(status_code=403, detail="无权操作：用户身份不匹配")
    async with db.session() as cursor:
        await refresh_cooldown_chances(cursor, userId)
        await cursor.execute("SELECT write_chances, pick_chances FROM users WHERE id = %s", [userId])
        row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=400, detail="用户不存在")
    return UserChancesResponse(writeChances=row["write_chances"], pickChances=row["pick_chances"])


@router.get("/checkin-status", response_model=CheckinStatusResponse)
async def get_checkin_status(userId: int = Query(...), db=Depends(get_database), current_user_id: int = Depends(get_current_user)):
    """签到状态接口（需JWT认证）"""
    if current_user_id != userId:
        raise HTTPException(status_code=403, detail="无权操作：用户身份不匹配")
    async with db.session() as cursor:
        await cursor.execute(
            "SELECT last_checkin_date, last_invite_date, invite_count_today, "
            "write_bonus_used, last_write_bonus_date, CURDATE() AS business_date "
            "FROM users WHERE id = %s",
            [userId],
        )
        row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=400, detail="用户不存在")

    today = row["business_date"]

    last_checkin = row["last_checkin_date"]
    can_checkin = last_checkin != today

    last_invite = row["last_invite_date"]
    is_today_invite = last_invite == today
    invite_used = row["invite_count_today"] if is_today_invite else 0

    last_write_bonus = row["last_write_bonus_date"]
    is_today_bonus = last_write_bonus == today
    write_bonus_used = row["write_bonus_used"] if is_today_bonus else 0

    return CheckinStatusResponse(
        canCheckin=can_checkin,
        inviteUsed=invite_used,
        inviteRemaining=3 - invite_used,
        writeBonusUsed=write_bonus_used,
        writeBonusRemaining=3 - write_bonus_used,
    )


@router.post("/checkin", response_model=CheckinResponse)
async def checkin(body: dict, db=Depends(get_database), current_user_id: int = Depends(get_current_user)):
    """签到接口（需JWT认证）"""
    user_id = body.get("userId")
    if not user_id:
        raise HTTPException(status_code=400, detail="缺少用户ID")
    if current_user_id != user_id:
        raise HTTPException(status_code=403, detail="无权操作：用户身份不匹配")

    async with db.session(transaction=True) as cursor:
        await claim_checkin(cursor, user_id)
    return CheckinResponse(success=True, message="签到成功！写纸团和拾纸团次数各+1")


@router.post("/invite", response_model=InviteResponse)
async def invite(body: dict, db=Depends(get_database), current_user_id: int = Depends(get_current_user)):
    """邀请接口（需JWT认证）"""
    user_id = body.get("userId")
    if not user_id:
        raise HTTPException(status_code=400, detail="缺少用户ID")
    if current_user_id != user_id:
        raise HTTPException(status_code=403, detail="无权操作：用户身份不匹配")

    async with db.session(transaction=True) as cursor:
        used = await claim_invite(cursor, user_id)
    return InviteResponse(
        success=True,
        message="邀请成功！拾纸团次数+1",
        inviteRemaining=3 - used,
    )


@router.post("/write-bonus", response_model=WriteBonusResponse)
async def write_bonus(body: dict, db=Depends(get_database), current_user_id: int = Depends(get_current_user)):
    """写纸团奖励接口（需JWT认证）"""
    user_id = body.get("userId")
    if not user_id:
        raise HTTPException(status_code=400, detail="缺少用户ID")
    if current_user_id != user_id:
        raise HTTPException(status_code=403, detail="无权操作：用户身份不匹配")

    async with db.session(transaction=True) as cursor:
        used = await claim_write_bonus(cursor, user_id)
    return WriteBonusResponse(
        success=True,
        message="写纸团奖励！拾纸团+1",
        writeBonusUsed=used,
        writeBonusRemaining=3 - used,
    )


@router.post("/update-chances")
async def update_chances(req: UpdateChancesRequest, db=Depends(get_database), current_user_id: int = Depends(get_current_user)):
    """兼容旧客户端，但不再允许客户端写入次数。"""
    raise HTTPException(
        status_code=410,
        detail="次数由服务端管理，请使用签到、邀请、写纸团或拾纸团接口",
    )
