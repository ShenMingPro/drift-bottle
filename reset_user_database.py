"""清空所有用户业务数据，并按当前 init_db.py 重建核心表。"""

from __future__ import annotations

import argparse
import asyncio

import aiomysql

from app.config import settings
from init_db import init_db


CORE_TABLES_DROP_ORDER = (
    "messages",
    "pick_commits",
    "pick_records",
    "conversations",
    "bottles",
    "users",
)
DATA_TABLES_TO_TRUNCATE = (
    "post_comments",
    "post_likes",
    "posts",
    "checkpoint_writes",
    "checkpoint_blobs",
    "checkpoints",
)
SYSTEM_DATABASES = {"information_schema", "mysql", "performance_schema", "sys"}


async def reset_user_database(expected_database: str) -> None:
    database = settings.DB_NAME
    if database != expected_database:
        raise RuntimeError(
            f"拒绝执行：--database={expected_database!r} 与 .env DB_NAME={database!r} 不一致"
        )
    if database.lower() in SYSTEM_DATABASES:
        raise RuntimeError("拒绝操作 MySQL 系统数据库")

    conn = await aiomysql.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        db=database,
        charset="utf8mb4",
        autocommit=False,
    )

    try:
        async with conn.cursor() as cur:
            await cur.execute("SHOW TABLES")
            existing_tables = {row[0] for row in await cur.fetchall()}

            print(f"目标数据库: {database}")
            print("删除前行数:")
            for table in (*DATA_TABLES_TO_TRUNCATE, *CORE_TABLES_DROP_ORDER):
                if table not in existing_tables:
                    continue
                await cur.execute(f"SELECT COUNT(*) FROM `{table}`")
                print(f"  {table}: {(await cur.fetchone())[0]}")

            await cur.execute("SET FOREIGN_KEY_CHECKS = 0")
            try:
                for table in DATA_TABLES_TO_TRUNCATE:
                    if table in existing_tables:
                        await cur.execute(f"TRUNCATE TABLE `{table}`")
                for table in CORE_TABLES_DROP_ORDER:
                    if table in existing_tables:
                        await cur.execute(f"DROP TABLE `{table}`")
            finally:
                await cur.execute("SET FOREIGN_KEY_CHECKS = 1")
            await conn.commit()
    except Exception:
        await conn.rollback()
        raise
    finally:
        conn.close()

    await init_db()
    print("用户业务数据已清空，核心表已按 username + password 结构重建。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, help="必须与 .env 中的 DB_NAME 完全一致")
    parser.add_argument("--yes", action="store_true", help="确认删除所有用户业务数据")
    args = parser.parse_args()
    if not args.yes:
        parser.error("此操作会删除用户业务数据，请显式传入 --yes")
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    asyncio.run(reset_user_database(cli_args.database))
