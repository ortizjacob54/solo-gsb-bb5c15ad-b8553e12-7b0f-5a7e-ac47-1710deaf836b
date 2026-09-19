"""FastAPI 入口。

启动即初始化 SQLite（见 lifespan）；接口：
  GET  /health
  POST /api/v1/plans                  创建/获取排样方案（幂等）
  GET  /api/v1/plans/{plan_id}        查询单个方案
  GET  /api/v1/plans                  查询方案列表（分页）
  POST /api/v1/plans/{plan_id}/gcode  导出方案的 G-code
  POST /api/v1/plans/{plan_id}/sessions       从已保存方案启动切割执行会话
  GET  /api/v1/sessions/{session_id}          查询切割会话（当前母板/下一刀/进度）
  POST /api/v1/sessions/{session_id}/events   提交“完成当前刀”事件（版本校验 + 幂等）
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from . import database as db
from .models import (
    CutEventRequest,
    CutSessionOut,
    GcodeRequest,
    GcodeResponse,
    HealthOut,
    PlanListResponse,
    PlanRequest,
    PlanResponse,
    PlanSummary,
)
from .services import (
    PlanGeometryError,
    SessionConflictError,
    create_plan,
    export_gcode,
    get_session_snapshot,
    start_session,
    submit_cut_event,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化数据库（表结构幂等）
    db.init_db()
    yield


app = FastAPI(
    title="板材直切排样 API",
    description=(
        "面向小型加工坊的矩形零件母板直切（guillotine）排样服务。"
        "仅生成贯穿当前余料区域的直切方案，锯缝占位，"
        "相同输入返回稳定且幂等的结果。"
        "已保存方案可导出确定性 G-code，并可启动持久化的切割执行会话："
        "按 panel_no/step 逐刀回报完成，版本与刀序在 SQLite 事务内校验，"
        "幂等键重试返回原结果。"
    ),
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthOut, tags=["meta"])
def health() -> HealthOut:
    return HealthOut(status="ok")


@app.post(
    "/api/v1/plans",
    response_model=PlanResponse,
    tags=["plans"],
    summary="创建排样方案",
)
def post_plan(req: PlanRequest) -> PlanResponse:
    payload = create_plan(req.model_dump())
    return JSONResponse(payload)


@app.get(
    "/api/v1/plans/{plan_id}",
    response_model=PlanResponse,
    tags=["plans"],
    summary="按编号查询方案",
)
def get_plan(plan_id: str) -> PlanResponse:
    with db.get_conn() as conn:
        payload = db.get_plan(plan_id, conn)
    if payload is None:
        raise HTTPException(status_code=404, detail="方案不存在")
    return JSONResponse(payload)


@app.post(
    "/api/v1/plans/{plan_id}/gcode",
    response_model=GcodeResponse,
    tags=["plans"],
    summary="导出方案的 G-code",
)
def post_plan_gcode(plan_id: str, req: GcodeRequest) -> GcodeResponse:
    try:
        payload = export_gcode(plan_id, req)
    except PlanGeometryError as exc:
        # 校验失败：结构化错误，不输出任何部分 G-code
        raise HTTPException(
            status_code=409,
            detail={
                "error": "invalid_plan_geometry",
                "message": "方案切割线未通过导出前几何校验，未生成任何 G-code",
                "violations": exc.violations,
            },
        ) from exc
    if payload is None:
        raise HTTPException(status_code=404, detail="方案不存在")
    return JSONResponse(payload)


@app.get(
    "/api/v1/plans",
    response_model=PlanListResponse,
    tags=["plans"],
    summary="查询方案列表",
)
def list_plans(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> PlanListResponse:
    with db.get_conn() as conn:
        total = db.count_plans(conn)
        rows = db.list_plans(limit, offset, conn)
    return PlanListResponse(
        total=total,
        plans=[PlanSummary(**row) for row in rows],
    )


@app.post(
    "/api/v1/plans/{plan_id}/sessions",
    response_model=CutSessionOut,
    status_code=201,
    tags=["sessions"],
    summary="从已保存方案启动切割执行会话",
)
def post_plan_session(plan_id: str) -> CutSessionOut:
    """启动会话：方案的全部切割线按 (panel_no, step) 展平为全局刀序。

    方案没有任何切割线时（如零件恰好铺满母板），会话创建即 completed。
    """
    payload = start_session(plan_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="方案不存在")
    return JSONResponse(payload, status_code=201)


@app.get(
    "/api/v1/sessions/{session_id}",
    response_model=CutSessionOut,
    tags=["sessions"],
    summary="查询切割会话",
)
def get_session(session_id: str) -> CutSessionOut:
    """返回当前母板、下一刀、已完成数与 completed 状态。"""
    payload = get_session_snapshot(session_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return JSONResponse(payload)


@app.post(
    "/api/v1/sessions/{session_id}/events",
    response_model=CutSessionOut,
    tags=["sessions"],
    summary="提交“完成当前刀”事件（幂等）",
)
def post_session_event(session_id: str, req: CutEventRequest) -> CutSessionOut:
    """在单个 SQLite 事务内校验版本与当前刀并推进会话。

    同一 idempotency_key 重试返回首次受理时的结果；旧版本或跳步/错刀
    请求返回 409，状态不推进。
    """
    try:
        payload = submit_cut_event(session_id, req)
    except SessionConflictError as exc:
        # 冲突：结构化错误体，状态未推进
        raise HTTPException(status_code=409, detail=exc.detail) from exc
    if payload is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return JSONResponse(payload)
