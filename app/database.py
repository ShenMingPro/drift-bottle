import aiomysql
from contextlib import asynccontextmanager
from typing import AsyncIterator

from app.config import settings

pool: aiomysql.Pool = None


class Database:
    """按需借用连接；事务只覆盖数据库操作，不跨模型调用。"""

    @asynccontextmanager
    async def session(self, *, transaction: bool = False) -> AsyncIterator[aiomysql.DictCursor]:
        if pool is None:
            raise RuntimeError("数据库连接池尚未初始化")
        async with pool.acquire() as connection:
            if transaction:
                await connection.begin()
            try:
                async with connection.cursor(aiomysql.DictCursor) as cursor:
                    yield cursor
                if transaction:
                    await connection.commit()
            except BaseException:
                if transaction:
                    await connection.rollback()
                raise


_database = Database()


def get_database() -> Database:
    """FastAPI 依赖：返回访问入口，此时不占用数据库连接。"""
    return _database


async def init_pool():
    global pool
    pool = await aiomysql.create_pool(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        db=settings.DB_NAME,
        charset="utf8mb4",
        autocommit=True,
        minsize=1,
        maxsize=10,
    )


async def close_pool():
    global pool
    if pool:
        pool.close()
        await pool.wait_closed()
        pool = None


async def get_db():
    # 保留无模型调用接口的旧依赖。
    async with _database.session() as cursor:
        yield cursor
