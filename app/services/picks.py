"""匹配结果的短事务提交，统一新旧接口的扣次与幂等规则。"""

from dataclasses import dataclass

from fastapi import HTTPException

from app.database import Database
from app.services.chances import consume_chance


@dataclass(frozen=True)
class PickCommit:
    record_id: int
    created: bool


async def get_pick_commit(
    database: Database, user_id: int, request_key: str, *, bottle_id: int | None = None
) -> dict | None:
    async with database.session() as cursor:
        await cursor.execute(
            "SELECT bottle_id, pick_record_id FROM pick_commits "
            "WHERE user_id = %s AND request_key = %s",
            [user_id, request_key],
        )
        committed = await cursor.fetchone()
        if committed or bottle_id is None:
            return committed
        # 升级前的任务没有提交凭据，以精确的用户/纸团记录兼容旧数据。
        await cursor.execute(
            "SELECT id FROM pick_records WHERE user_id = %s AND bottle_id = %s "
            "ORDER BY id LIMIT 1",
            [user_id, bottle_id],
        )
        existing = await cursor.fetchone()
        return {"bottle_id": bottle_id, "pick_record_id": existing["id"]} if existing else None


async def commit_pick(
    database: Database, user_id: int, bottle_id: int, *, request_key: str
) -> PickCommit:
    """同请求只提交一次；同瓶复用记录，同作者不同瓶拒绝过期匹配。"""
    async with database.session(transaction=True) as cursor:
        # 同一用户的业务提交在数据库内串行，覆盖不同 run 和不同进程。
        await cursor.execute("SELECT id FROM users WHERE id = %s FOR UPDATE", [user_id])
        if not await cursor.fetchone():
            raise HTTPException(status_code=400, detail="用户不存在")

        await cursor.execute(
            "SELECT bottle_id, pick_record_id FROM pick_commits "
            "WHERE user_id = %s AND request_key = %s",
            [user_id, request_key],
        )
        committed = await cursor.fetchone()
        if committed:
            if committed["bottle_id"] != bottle_id:
                raise HTTPException(status_code=409, detail="该请求已提交其他匹配结果")
            return PickCommit(int(committed["pick_record_id"]), created=False)

        await cursor.execute("SELECT user_id FROM bottles WHERE id = %s", [bottle_id])
        bottle = await cursor.fetchone()
        if not bottle:
            raise HTTPException(status_code=404, detail="纸团不存在，请重新匹配")
        if int(bottle["user_id"]) == user_id:
            raise HTTPException(status_code=400, detail="不能拾取自己的纸团")

        await cursor.execute(
            "SELECT id FROM pick_records WHERE user_id = %s AND bottle_id = %s "
            "ORDER BY id LIMIT 1",
            [user_id, bottle_id],
        )
        existing = await cursor.fetchone()
        created = existing is None
        if existing:
            record_id = int(existing["id"])
        else:
            # MatchAgent 同样排除已匹配作者；在提交阶段再次检查并发生成的旧候选。
            await cursor.execute(
                "SELECT pr.id FROM pick_records pr JOIN bottles b ON b.id = pr.bottle_id "
                "WHERE pr.user_id = %s AND b.user_id = %s LIMIT 1",
                [user_id, bottle["user_id"]],
            )
            if await cursor.fetchone():
                raise HTTPException(status_code=409, detail="已匹配过该用户，请重新匹配")
            await consume_chance(cursor, user_id, "pick_chances")
            await cursor.execute(
                "INSERT INTO pick_records (user_id, bottle_id) VALUES (%s, %s)",
                [user_id, bottle_id],
            )
            record_id = cursor.lastrowid

        await cursor.execute(
            "INSERT INTO pick_commits (user_id, request_key, bottle_id, pick_record_id) "
            "VALUES (%s, %s, %s, %s)",
            [user_id, request_key, bottle_id, record_id],
        )
        return PickCommit(record_id, created=created)
