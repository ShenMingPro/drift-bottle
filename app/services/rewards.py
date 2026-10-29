"""每日奖励规则；调用方必须在同一连接的事务中调用。"""

from fastapi import HTTPException


async def claim_checkin(cursor, user_id: int) -> None:
    """锁定用户行，按数据库业务日期发放每天一次的签到奖励。"""
    await cursor.execute(
        "SELECT last_checkin_date, CURDATE() AS business_date "
        "FROM users WHERE id = %s FOR UPDATE",
        [user_id],
    )
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=400, detail="用户不存在")
    today = row["business_date"]
    if row["last_checkin_date"] == today:
        raise HTTPException(status_code=400, detail="今日已签到")
    await cursor.execute(
        "UPDATE users SET last_checkin_date = %s, "
        "write_chances = write_chances + 1, pick_chances = pick_chances + 1 "
        "WHERE id = %s",
        [today, user_id],
    )


async def claim_invite(cursor, user_id: int) -> int:
    """锁定用户行后发放邀请奖励，返回数据库当日已使用次数。"""
    await cursor.execute(
        "SELECT last_invite_date, invite_count_today, CURDATE() AS business_date "
        "FROM users WHERE id = %s FOR UPDATE",
        [user_id],
    )
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=400, detail="用户不存在")
    today = row["business_date"]
    count = row["invite_count_today"] if row["last_invite_date"] == today else 0
    if count >= 3:
        raise HTTPException(status_code=400, detail="今日邀请次数已用完")
    used = count + 1
    await cursor.execute(
        "UPDATE users SET last_invite_date = %s, invite_count_today = %s, "
        "pick_chances = pick_chances + 1 WHERE id = %s",
        [today, used, user_id],
    )
    return used


async def claim_write_bonus(cursor, user_id: int, *, require_write: bool = True) -> int:
    """领取写纸团奖励；随发瓶消费时由外层事务保证纸团一起写入。"""
    await cursor.execute(
        "SELECT write_bonus_used, last_write_bonus_date, CURDATE() AS business_date "
        "FROM users WHERE id = %s FOR UPDATE",
        [user_id],
    )
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=400, detail="用户不存在")
    today = row["business_date"]
    if require_write:
        await cursor.execute(
            "SELECT COUNT(*) AS cnt FROM bottles "
            "WHERE user_id = %s AND created_at >= %s AND created_at < %s + INTERVAL 1 DAY",
            [user_id, today, today],
        )
        written_today = await cursor.fetchone()
        if not written_today or not written_today.get("cnt"):
            raise HTTPException(status_code=400, detail="请先完成一次写纸团")
    count = row["write_bonus_used"] if row["last_write_bonus_date"] == today else 0
    if count >= 3:
        raise HTTPException(status_code=400, detail="今日写纸团奖励次数已用完")
    used = count + 1
    await cursor.execute(
        "UPDATE users SET last_write_bonus_date = %s, write_bonus_used = %s, "
        "pick_chances = pick_chances + 1 WHERE id = %s",
        [today, used, user_id],
    )
    return used
