"""可重复执行的增量表定义；保留已有业务记录。"""

PICK_COMMITS_SQL = """
CREATE TABLE IF NOT EXISTS pick_commits (
    user_id INT NOT NULL,
    request_key VARCHAR(128) NOT NULL,
    bottle_id INT NOT NULL,
    pick_record_id INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, request_key),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (bottle_id) REFERENCES bottles(id) ON DELETE CASCADE,
    FOREIGN KEY (pick_record_id) REFERENCES pick_records(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


async def ensure_backend_schema(cursor) -> None:
    # 只新增提交凭据表，不修改、删除或去重历史 pick_records。
    await cursor.execute(PICK_COMMITS_SQL)
