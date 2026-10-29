import asyncio

import aiomysql
from dotenv import load_dotenv
import os

from app.schema import ensure_backend_schema

load_dotenv()


async def init_db():
    conn = await aiomysql.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        charset="utf8mb4",
    )

    async with conn.cursor() as cur:
        db_name = os.getenv("DB_NAME", "paper_bottle")

        await cur.execute(
            f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        print(f'数据库 "{db_name}" 创建成功')

        await cur.execute(f"USE `{db_name}`")

        await cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(64) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL,
                nickname VARCHAR(50) NOT NULL,
                gender VARCHAR(10),
                age INT,
                hobbies VARCHAR(200),
                contact_type VARCHAR(20),
                contact_value VARCHAR(100),
                province VARCHAR(50),
                city VARCHAR(50),
                university VARCHAR(100),
                last_checkin_date DATE DEFAULT NULL,
                last_invite_date DATE DEFAULT NULL,
                invite_count_today INT DEFAULT 0,
                write_bonus_used INT DEFAULT 0,
                last_write_bonus_date DATE DEFAULT NULL,
                write_chances INT DEFAULT 1,
                pick_chances INT DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print('用户表 "users" 创建成功')

        await cur.execute("""
            CREATE TABLE IF NOT EXISTS bottles (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        print('纸团表 "bottles" 创建成功')

        await cur.execute("""
            CREATE TABLE IF NOT EXISTS pick_records (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                bottle_id INT NOT NULL,
                picked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (bottle_id) REFERENCES bottles(id) ON DELETE CASCADE
            )
        """)
        print('拾取记录表 "pick_records" 创建成功')
        await ensure_backend_schema(cur)

        await cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user1_id INT NOT NULL,
                user2_id INT NOT NULL,
                last_message_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user1_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (user2_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE KEY uk_users (user1_id, user2_id)
            )
        """)
        print('对话表 "conversations" 创建成功')

        await cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INT AUTO_INCREMENT PRIMARY KEY,
                conversation_id INT NOT NULL,
                sender_id INT NOT NULL,
                content TEXT NOT NULL,
                is_read TINYINT DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                FOREIGN KEY (sender_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        print('消息表 "messages" 创建成功')

        await cur.execute(
            "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
            f"WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'users' AND COLUMN_NAME = 'write_bonus_used'",
            [db_name],
        )
        if not await cur.fetchone():
            await cur.execute("ALTER TABLE users ADD COLUMN write_bonus_used INT DEFAULT 0")
            await cur.execute("ALTER TABLE users ADD COLUMN last_write_bonus_date DATE DEFAULT NULL")
            print('用户表新增 "write_bonus_used" 和 "last_write_bonus_date" 字段成功')

        print("\n数据库初始化完成！")

    conn.close()


if __name__ == "__main__":
    asyncio.run(init_db())
