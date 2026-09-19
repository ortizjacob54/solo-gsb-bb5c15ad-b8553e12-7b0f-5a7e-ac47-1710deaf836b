"""SQLite 持久化与数据库初始化。

plan_id 由规范化请求 JSON 的 SHA256 派生（与时间无关），相同输入天然幂等：
重复提交会命中已存在的方案记录并原样返回，保证“相同输入得到稳定结果”。
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from typing import Optional

DB_PATH = os.environ.get(
    "NESTING_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data", "nesting.db"),
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS plans (
    plan_id        TEXT PRIMARY KEY,
    request_json   TEXT NOT NULL,
    response_json  TEXT NOT NULL,
    status         TEXT NOT NULL,
    panels_used    INTEGER NOT NULL,
    placed_parts   INTEGER NOT NULL,
    unplaced_parts INTEGER NOT NULL,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_plans_created_at ON plans(created_at DESC);
"""


def init_db(path: Optional[str] = None) -> None:
    """创建数据目录、连接并初始化表结构；可重复执行。"""
    target = path or DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    with sqlite3.connect(target) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.executescript(_SCHEMA)
        conn.commit()


@contextmanager
def get_conn(path: Optional[str] = None):
    target = path or DB_PATH
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
    finally:
        conn.close()


def canonical_request(req_dict: dict) -> str:
    """规范化序列化：键排序、空格固定，使同一语义输入得到同一字符串。"""
    return json.dumps(req_dict, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def plan_id_from_request(req_dict: dict) -> str:
    digest = hashlib.sha256(canonical_request(req_dict).encode("utf-8")).hexdigest()
    # 以 UUID 形式呈现（确定性 UUID，仅借用其格式）
    return str(uuid.UUID(digest[:32]))


def save_plan(plan_id: str, request_json: str, response_json: str,
              created_at: str, conn: sqlite3.Connection) -> None:
    resp = json.loads(response_json)
    summary = resp["summary"]
    conn.execute(
        """
        INSERT OR IGNORE INTO plans
          (plan_id, request_json, response_json, status, panels_used,
           placed_parts, unplaced_parts, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (plan_id, request_json, response_json, resp["status"],
         summary["panels_used"], summary["placed_parts"],
         summary["unplaced_parts"], created_at),
    )
    conn.commit()


def get_plan(plan_id: str, conn: sqlite3.Connection) -> Optional[dict]:
    row = conn.execute(
        "SELECT response_json FROM plans WHERE plan_id = ?", (plan_id,)
    ).fetchone()
    return json.loads(row["response_json"]) if row else None


def list_plans(limit: int, offset: int, conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT plan_id, status, created_at, panels_used, placed_parts,
               unplaced_parts
          FROM plans
         ORDER BY created_at DESC, plan_id
         LIMIT ? OFFSET ?
        """,
        (limit, offset),
    ).fetchall()
    return [dict(r) for r in rows]


def count_plans(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) AS c FROM plans").fetchone()["c"]
