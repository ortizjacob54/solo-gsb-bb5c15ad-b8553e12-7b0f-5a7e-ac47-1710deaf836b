"""SQLite 持久化与数据库初始化。

plan_id 由规范化请求 JSON 的 SHA256 派生（与时间无关），相同输入天然幂等：
重复提交会命中已存在的方案记录并原样返回，保证“相同输入得到稳定结果”。

切割执行会话（cut_sessions / cut_events）：
- cut_sessions 记录会话进度与乐观并发版本 version（每完成一刀 +1）；
- cut_events 以 (session_id, idempotency_key) 为主键记录已受理事件及其
  响应快照，同一键重试直接返回首次结果，状态不再推进。
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

CREATE TABLE IF NOT EXISTS cut_sessions (
    session_id     TEXT PRIMARY KEY,
    plan_id        TEXT NOT NULL REFERENCES plans(plan_id),
    status         TEXT NOT NULL,            -- active / completed
    version        INTEGER NOT NULL,         -- 乐观并发版本：每完成一刀 +1
    completed_cuts INTEGER NOT NULL,         -- 已完成刀数
    total_cuts     INTEGER NOT NULL,         -- 方案总刀数（方案不可变，启动时固化）
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    completed_at   TEXT                      -- 全部完成时间；未完成为 NULL
);
CREATE INDEX IF NOT EXISTS idx_cut_sessions_plan ON cut_sessions(plan_id);

CREATE TABLE IF NOT EXISTS cut_events (
    session_id      TEXT NOT NULL REFERENCES cut_sessions(session_id),
    idempotency_key TEXT NOT NULL,
    panel_no        INTEGER NOT NULL,
    step            INTEGER NOT NULL,
    from_version    INTEGER NOT NULL,
    to_version      INTEGER NOT NULL,
    result_json     TEXT NOT NULL,           -- 首次受理时返回的会话快照，重试原样返回
    created_at      TEXT NOT NULL,
    PRIMARY KEY (session_id, idempotency_key)
);
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
    # 并发写者等待而非立即报 SQLITE_BUSY（配合 WAL 与 BEGIN IMMEDIATE 串行化）
    conn.execute("PRAGMA busy_timeout=5000;")
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def immediate_tx(conn: sqlite3.Connection):
    """显式 BEGIN IMMEDIATE 事务：进入即取写锁，把并发写请求串行化。

    事务内的校验（版本、当前刀、幂等键）与推进（更新会话、记录事件）
    构成一个原子单元；异常时整体回滚，绝不留下半推进状态。
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()


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


# ---------------------------------------------------------------------------
# 切割执行会话
# ---------------------------------------------------------------------------

def create_session(session_id: str, plan_id: str, status: str, total_cuts: int,
                   now: str, conn: sqlite3.Connection) -> None:
    """新建会话；无切割线的方案（total_cuts=0）创建即 completed。"""
    conn.execute(
        """
        INSERT INTO cut_sessions
          (session_id, plan_id, status, version, completed_cuts, total_cuts,
           created_at, updated_at, completed_at)
        VALUES (?, ?, ?, 0, 0, ?, ?, ?, ?)
        """,
        (session_id, plan_id, status, total_cuts, now, now,
         now if status == "completed" else None),
    )
    conn.commit()


def get_session(session_id: str, conn: sqlite3.Connection) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM cut_sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    return dict(row) if row else None


def get_session_event(session_id: str, idempotency_key: str,
                      conn: sqlite3.Connection) -> Optional[dict]:
    """按幂等键取首次受理时的响应快照；未受理过返回 None。"""
    row = conn.execute(
        """
        SELECT result_json FROM cut_events
         WHERE session_id = ? AND idempotency_key = ?
        """,
        (session_id, idempotency_key),
    ).fetchone()
    return json.loads(row["result_json"]) if row else None


def advance_session(session_id: str, from_version: int, completed_cuts: int,
                    status: str, now: str, conn: sqlite3.Connection) -> bool:
    """乐观并发推进（CAS）：仅当当前版本仍为 from_version 时生效。"""
    cur = conn.execute(
        """
        UPDATE cut_sessions
           SET version = version + 1,
               completed_cuts = ?,
               status = ?,
               updated_at = ?,
               completed_at = CASE WHEN ? = 'completed' THEN ? ELSE completed_at END
         WHERE session_id = ? AND version = ?
        """,
        (completed_cuts, status, now, status, now, session_id, from_version),
    )
    return cur.rowcount == 1


def record_session_event(session_id: str, idempotency_key: str, panel_no: int,
                         step: int, from_version: int, to_version: int,
                         result_json: str, now: str,
                         conn: sqlite3.Connection) -> None:
    """记录已受理事件及其响应快照（主键即幂等键，重复插入会报错）。"""
    conn.execute(
        """
        INSERT INTO cut_events
          (session_id, idempotency_key, panel_no, step, from_version,
           to_version, result_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (session_id, idempotency_key, panel_no, step, from_version,
         to_version, result_json, now),
    )


def count_session_events(session_id: str, conn: sqlite3.Connection) -> int:
    return conn.execute(
        "SELECT COUNT(*) AS c FROM cut_events WHERE session_id = ?",
        (session_id,),
    ).fetchone()["c"]
