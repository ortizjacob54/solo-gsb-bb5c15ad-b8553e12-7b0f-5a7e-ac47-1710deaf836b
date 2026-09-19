"""排样业务编排：参数模型 -> 核心算法 -> 响应装配 -> 落库。"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from . import database as db
from .gcode import generate_gcode, validate_plan_cuts
from .models import CutEventRequest, GcodeRequest, PlanRequest
from .nesting import build_payload, solve


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_plan(req: PlanRequest) -> dict:
    """纯计算（不碰数据库），便于单元测试。"""
    parts = [
        {
            "id": p.id,
            "width": p.width,
            "height": p.height,
            "quantity": p.quantity,
            "rotatable": p.rotatable,
        }
        for p in req.parts
    ]
    internal = solve(
        board_w=req.board_width,
        board_h=req.board_height,
        kerf=req.kerf_width,
        max_boards=req.max_boards,
        parts=parts,
    )
    return build_payload(internal)


def create_plan(req_dict: dict) -> dict:
    """幂等创建：相同请求返回同一方案。"""
    request_json = db.canonical_request(req_dict)
    plan_id = db.plan_id_from_request(req_dict)

    with db.get_conn() as conn:
        existing = db.get_plan(plan_id, conn)
        if existing is not None:
            return existing

        payload = compute_plan(PlanRequest.model_validate(req_dict))
        payload["plan_id"] = plan_id
        payload["created_at"] = _utcnow()
        db.save_plan(plan_id, request_json, json.dumps(payload, ensure_ascii=False),
                     payload["created_at"], conn)
        return payload


class PlanGeometryError(Exception):
    """已存方案未通过导出前几何校验；violations 为结构化错误列表。"""

    def __init__(self, violations: list[dict]):
        super().__init__("方案切割线未通过导出前几何校验")
        self.violations = violations


def export_gcode(plan_id: str, req: GcodeRequest) -> Optional[dict]:
    """导出已存方案的 G-code。

    方案不存在返回 None；几何校验失败抛 PlanGeometryError（不产出任何
    部分 G-code）。
    """
    with db.get_conn() as conn:
        plan = db.get_plan(plan_id, conn)
    if plan is None:
        return None
    violations = validate_plan_cuts(plan)
    if violations:
        raise PlanGeometryError(violations)
    return generate_gcode(
        plan,
        plan_id=plan_id,
        origin=req.origin,
        unit=req.unit,
        safe_z=req.safe_z,
    )


# ---------------------------------------------------------------------------
# 切割执行会话
# ---------------------------------------------------------------------------

class SessionConflictError(Exception):
    """会话事件冲突（HTTP 409）；detail 为结构化错误体。"""

    def __init__(self, detail: dict):
        super().__init__(detail.get("message", "会话事件冲突"))
        self.detail = detail


def _flatten_cuts(plan: dict) -> list[dict]:
    """把方案按 (panel_no, step) 展平为全局切割序列（与 G-code 导出同序）。"""
    seq: list[dict] = []
    for panel in sorted(plan.get("panels", []), key=lambda p: p["panel_no"]):
        for cut in sorted(panel.get("cuts", []), key=lambda c: c["step"]):
            seq.append({
                "panel_no": panel["panel_no"],
                "step": cut["step"],
                "axis": cut["axis"],
                "x": cut["x"],
                "y": cut["y"],
                "length": cut["length"],
                "region": cut["region"],
            })
    return seq


def _session_snapshot(session: dict, cut_seq: list[dict]) -> dict:
    """由会话行 + 方案切割序列装配对外快照。

    当前母板/下一刀由 completed_cuts 在切割序列中的位置唯一推导，
    不在库里冗余存储，避免状态漂移。
    """
    idx = session["completed_cuts"]
    nxt = cut_seq[idx] if idx < len(cut_seq) else None
    return {
        "session_id": session["session_id"],
        "plan_id": session["plan_id"],
        "status": session["status"],
        "completed": session["status"] == "completed",
        "version": session["version"],
        "current_panel": nxt["panel_no"] if nxt else None,
        "next_cut": nxt,
        "completed_cuts": session["completed_cuts"],
        "total_cuts": session["total_cuts"],
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
        "completed_at": session["completed_at"],
    }


def start_session(plan_id: str) -> Optional[dict]:
    """从已保存方案启动切割会话；方案不存在返回 None。

    方案没有任何切割线时（如零件恰好铺满母板），会话创建即 completed。
    """
    with db.get_conn() as conn:
        plan = db.get_plan(plan_id, conn)
        if plan is None:
            return None
        cut_seq = _flatten_cuts(plan)
        now = _utcnow()
        session_id = str(uuid.uuid4())
        status = "completed" if not cut_seq else "active"
        db.create_session(session_id, plan_id, status, len(cut_seq), now, conn)
        session = db.get_session(session_id, conn)
    return _session_snapshot(session, cut_seq)


def get_session_snapshot(session_id: str) -> Optional[dict]:
    """查询会话：当前母板、下一刀、已完成数与 completed 状态。"""
    with db.get_conn() as conn:
        session = db.get_session(session_id, conn)
        if session is None:
            return None
        plan = db.get_plan(session["plan_id"], conn)
    return _session_snapshot(session, _flatten_cuts(plan))


def submit_cut_event(session_id: str, req: CutEventRequest) -> Optional[dict]:
    """提交“完成当前刀”事件：在单个 SQLite 事务中校验并推进会话。

    事务内依次校验：幂等键（重试直接返回首次结果）-> 会话未完成 ->
    expected_version 等于当前版本 -> (panel_no, step) 正是当前刀；
    全部通过才以 CAS 方式推进版本并记录事件。会话不存在返回 None；
    校验失败抛 SessionConflictError（事务整体回滚，状态不推进）。
    """
    with db.get_conn() as conn:
        with db.immediate_tx(conn):
            session = db.get_session(session_id, conn)
            if session is None:
                return None

            # 同一幂等键重试：原样返回首次受理时的结果，不再推进状态
            replay = db.get_session_event(session_id, req.idempotency_key, conn)
            if replay is not None:
                return replay

            if session["status"] == "completed":
                raise SessionConflictError({
                    "error": "session_completed",
                    "message": "会话已完成，不再接受新的切割事件",
                })

            if req.expected_version != session["version"]:
                raise SessionConflictError({
                    "error": "version_conflict",
                    "message": f"期望版本 {req.expected_version} 与当前版本 "
                               f"{session['version']} 不一致",
                    "current_version": session["version"],
                })

            plan = db.get_plan(session["plan_id"], conn)
            cut_seq = _flatten_cuts(plan)
            current = cut_seq[session["completed_cuts"]]
            if (req.panel_no, req.step) != (current["panel_no"], current["step"]):
                raise SessionConflictError({
                    "error": "cut_mismatch",
                    "message": "提交的刀序与当前应执行的刀不一致（跳步或错刀）",
                    "expected": {"panel_no": current["panel_no"],
                                 "step": current["step"]},
                    "got": {"panel_no": req.panel_no, "step": req.step},
                })

            now = _utcnow()
            completed_cuts = session["completed_cuts"] + 1
            status = ("completed" if completed_cuts >= session["total_cuts"]
                      else "active")
            if not db.advance_session(session_id, session["version"],
                                      completed_cuts, status, now, conn):
                # BEGIN IMMEDIATE 串行化下理论上不可达；防御性兜底
                fresh = db.get_session(session_id, conn)
                raise SessionConflictError({
                    "error": "version_conflict",
                    "message": "会话版本已变化，请刷新后重试",
                    "current_version": fresh["version"],
                })

            session.update(
                version=session["version"] + 1,
                completed_cuts=completed_cuts,
                status=status,
                updated_at=now,
                completed_at=now if status == "completed" else None,
            )
            snapshot = _session_snapshot(session, cut_seq)
            db.record_session_event(
                session_id, req.idempotency_key, req.panel_no, req.step,
                req.expected_version, session["version"],
                json.dumps(snapshot, ensure_ascii=False), now, conn,
            )
            return snapshot
