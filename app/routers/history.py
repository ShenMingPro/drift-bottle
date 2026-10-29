from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth_jwt import get_current_user  # JWT认证依赖项
from app.database import get_db
from app.models.history import HistoryResponse, PickRecord, WriteRecord

router = APIRouter()


@router.get("/history", response_model=HistoryResponse)
async def get_history(userId: int = Query(...), db=Depends(get_db), current_user_id: int = Depends(get_current_user)):
    """历史记录接口（需JWT认证）"""
    if current_user_id != userId:
        raise HTTPException(status_code=403, detail="无权操作：用户身份不匹配")
    await db.execute(
        "SELECT id, content, created_at FROM bottles WHERE user_id = %s ORDER BY created_at DESC",
        [userId],
    )
    write_rows = await db.fetchall()

    await db.execute(
        "SELECT pr.picked_at, b.content, b.created_at AS bottle_created_at, "
        "b.user_id AS bottle_user_id, "
        "u.nickname, u.gender, u.age, u.hobbies, u.contact_type, u.contact_value, "
        "u.province, u.city, u.university "
        "FROM pick_records pr "
        "JOIN bottles b ON pr.bottle_id = b.id "
        "JOIN users u ON b.user_id = u.id "
        "WHERE pr.user_id = %s ORDER BY pr.picked_at DESC",
        [userId],
    )
    pick_rows = await db.fetchall()

    write_records = [
        WriteRecord(id=r["id"], content=r["content"], createdAt=str(r["created_at"]))
        for r in write_rows
    ]

    pick_records = [
        PickRecord(
            content=r["content"],
            pickedAt=str(r["picked_at"]),
            bottleCreatedAt=str(r["bottle_created_at"]),
            bottleUserId=r["bottle_user_id"],
            nickname=r["nickname"],
            gender=r["gender"] or "",
            age=r["age"] or 0,
            hobbies=r["hobbies"] or "",
            contact_type=r["contact_type"] or "",
            contact_value=r["contact_value"] or "",
            province=r["province"] or "",
            city=r["city"] or "",
            university=r["university"] or "",
        )
        for r in pick_rows
    ]

    return HistoryResponse(writeRecords=write_records, pickRecords=pick_records)
