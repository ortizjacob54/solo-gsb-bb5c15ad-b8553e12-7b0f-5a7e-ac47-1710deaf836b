"""排样业务编排：参数模型 -> 核心算法 -> 响应装配 -> 落库。"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from . import database as db
from .gcode import generate_gcode, validate_plan_cuts
from .models import GcodeRequest, PlanRequest
from .nesting import build_payload, solve


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
        payload["created_at"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
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
