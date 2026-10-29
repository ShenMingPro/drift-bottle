"""服务端次数规则：扣减、奖励和冷却恢复。"""

from __future__ import annotations

from fastapi import HTTPException

CHANCE_FIELDS = {"write_chances", "pick_chances"}
COOLDOWN_HOURS = 3


async def consume_chance(db, user_id: int, field: str) -> None:
    """原子扣减一次机会；没有次数时拒绝操作。"""
    if field not in CHANCE_FIELDS:
        raise ValueError(f"不支持的机会字段: {field}")

    await db.execute(
        f"UPDATE users SET {field} = {field} - 1 "
        f"WHERE id = %s AND {field} > 0",
        [user_id],
    )
    if getattr(db, "rowcount", None) == 0:
        raise HTTPException(status_code=400, detail="使用次数不足，请先完成任务")


async def grant_chances(
    db,
    user_id: int,
    *,
    write_delta: int = 0,
    pick_delta: int = 0,
) -> None:
    """只允许服务端业务动作调用，给用户增加次数。"""
    if write_delta < 0 or pick_delta < 0:
        raise ValueError("奖励次数不能为负数")
    await db.execute(
        "UPDATE users SET write_chances = write_chances + %s, "
        "pick_chances = pick_chances + %s WHERE id = %s",
        [write_delta, pick_delta, user_id],
    )


async def refresh_cooldown_chances(db, user_id: int) -> None:
    """服务端实现 3 小时冷却恢复，避免客户端直接改写次数。

    写纸团使用 bottles.created_at，拾纸团使用 pick_records.picked_at。
    只有次数为 0 且最近一次对应动作超过冷却时间时才恢复 1 次。
    """
    await db.execute(
        "UPDATE users SET write_chances = 1 "
        "WHERE id = %s AND write_chances = 0 AND ("
        "NOT EXISTS (SELECT 1 FROM bottles WHERE user_id = %s) OR "
        "(SELECT MAX(created_at) FROM bottles WHERE user_id = %s) "
        f"<= NOW() - INTERVAL {COOLDOWN_HOURS} HOUR)",
        [user_id, user_id, user_id],
    )
    await db.execute(
        "UPDATE users SET pick_chances = 1 "
        "WHERE id = %s AND pick_chances = 0 AND ("
        "NOT EXISTS (SELECT 1 FROM pick_records WHERE user_id = %s) OR "
        "(SELECT MAX(picked_at) FROM pick_records WHERE user_id = %s) "
        f"<= NOW() - INTERVAL {COOLDOWN_HOURS} HOUR)",
        [user_id, user_id, user_id],
    )
