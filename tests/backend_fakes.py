"""路由契约测试使用的数据库替身；事务与并发另由真实 MySQL 验证。"""

from contextlib import asynccontextmanager
from copy import deepcopy


class PickDatabase:
    def __init__(self):
        self.executions = []
        self.records = {}
        self.commits = {}
        self.bottles = {12: 23, 13: 23, 14: 24}
        self.chances = 10
        self.rowcount = 1
        self.lastrowid = 0
        self._next_row = None
        self.fail_next_insert = False
        self.rollbacks = 0

    @asynccontextmanager
    async def session(self, *, transaction=False):
        before = deepcopy((self.records, self.commits, self.chances))
        try:
            yield self
        except BaseException:
            if transaction:
                self.records, self.commits, self.chances = before
                self.rollbacks += 1
            raise

    async def execute(self, query, args):
        self.executions.append((query, args))
        query = " ".join(query.split())
        self._next_row = None
        self.rowcount = 1
        if query.startswith("SELECT id FROM users"):
            self._next_row = {"id": args[0]}
        elif query.startswith("SELECT bottle_id, pick_record_id FROM pick_commits"):
            self._next_row = self.commits.get(tuple(args))
        elif query.startswith("SELECT user_id FROM bottles"):
            owner = self.bottles.get(args[0])
            self._next_row = {"user_id": owner} if owner else None
        elif query.startswith("SELECT id FROM pick_records"):
            record_id = self.records.get(tuple(args))
            self._next_row = {"id": record_id} if record_id else None
        elif query.startswith("SELECT pr.id FROM pick_records"):
            self._next_row = next(
                ({"id": record_id} for (user, bottle), record_id in self.records.items()
                 if user == args[0] and self.bottles[bottle] == args[1]),
                None,
            )
        elif query.startswith("UPDATE users SET pick_chances"):
            self.rowcount = int(self.chances > 0)
            self.chances -= self.rowcount
        elif query.startswith("INSERT INTO pick_records"):
            if self.fail_next_insert:
                self.fail_next_insert = False
                raise RuntimeError("injected record insert failure")
            self.lastrowid = len(self.records) + 1
            self.records[tuple(args)] = self.lastrowid
        elif query.startswith("INSERT INTO pick_commits"):
            user, key, bottle, record_id = args
            self.commits[user, key] = {"bottle_id": bottle, "pick_record_id": record_id}
        else:
            raise AssertionError(f"Unexpected SQL: {query}")

    async def fetchone(self):
        return self._next_row
