"""切割执行会话测试：持久化、幂等重试、版本/刀序校验与并发安全。

服务层并发测试用多线程直接打 SQLite（每线程独立连接），
验证 BEGIN IMMEDIATE 事务 + 乐观版本 CAS 的串行化语义；
API 层用 TestClient 做串行端到端（未安装 fastapi 时跳过）。
"""
import json
import os
import tempfile
import threading
import unittest

from app import database as db
from app.models import CutEventRequest
from app.services import (
    SessionConflictError,
    create_plan,
    get_session_snapshot,
    start_session,
    submit_cut_event,
)

try:
    from app.main import app
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except Exception:
    HAS_FASTAPI = False


# 有多条切割线的常规方案
REQ = {
    "board_width": 244.0, "board_height": 122.0, "kerf_width": 3.0,
    "max_boards": 4,
    "parts": [
        {"id": "door", "width": 120, "height": 60, "quantity": 2,
         "rotatable": True},
        {"id": "shelf", "width": 60, "height": 40, "quantity": 3,
         "rotatable": True},
    ],
}

# 零件恰好铺满母板：方案没有任何切割线
EXACT_REQ = {
    "board_width": 100.0, "board_height": 100.0, "kerf_width": 3.0,
    "max_boards": 2,
    "parts": [
        {"id": "A", "width": 100, "height": 100, "quantity": 1,
         "rotatable": False},
    ],
}


def ev(panel_no, step, version, key):
    return CutEventRequest(panel_no=panel_no, step=step,
                           expected_version=version, idempotency_key=key)


class SessionTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._orig_path = db.DB_PATH
        db.DB_PATH = os.path.join(self.tmp.name, "sessions.db")
        db.init_db(db.DB_PATH)

    def tearDown(self):
        db.DB_PATH = self._orig_path
        self.tmp.cleanup()

    def _plan(self, req=REQ):
        return create_plan(json.loads(json.dumps(req)))

    def _event_count(self, session_id):
        with db.get_conn() as conn:
            return db.count_session_events(session_id, conn)


class TestSessionService(SessionTestBase):
    def test_start_session_snapshot(self):
        plan = self._plan()
        total = sum(len(p["cuts"]) for p in plan["panels"])
        self.assertGreater(total, 0)

        s = start_session(plan["plan_id"])
        self.assertEqual(s["plan_id"], plan["plan_id"])
        self.assertEqual(s["status"], "active")
        self.assertFalse(s["completed"])
        self.assertEqual(s["version"], 0)
        self.assertEqual(s["completed_cuts"], 0)
        self.assertEqual(s["total_cuts"], total)
        first = plan["panels"][0]["cuts"][0]
        self.assertEqual(s["current_panel"], plan["panels"][0]["panel_no"])
        self.assertEqual(s["next_cut"]["panel_no"], plan["panels"][0]["panel_no"])
        self.assertEqual(s["next_cut"]["step"], first["step"])
        self.assertEqual(s["next_cut"]["axis"], first["axis"])
        self.assertIsNone(s["completed_at"])
        # 查询接口返回同一快照
        self.assertEqual(get_session_snapshot(s["session_id"]), s)

    def test_start_session_plan_missing(self):
        self.assertIsNone(start_session("no-such-plan"))

    def test_empty_plan_completes_immediately(self):
        plan = self._plan(EXACT_REQ)
        self.assertEqual(sum(len(p["cuts"]) for p in plan["panels"]), 0)
        s = start_session(plan["plan_id"])
        self.assertEqual(s["status"], "completed")
        self.assertTrue(s["completed"])
        self.assertIsNone(s["current_panel"])
        self.assertIsNone(s["next_cut"])
        self.assertEqual(s["completed_cuts"], 0)
        self.assertEqual(s["total_cuts"], 0)
        self.assertIsNotNone(s["completed_at"])
        # 已完成会话拒绝任何事件
        with self.assertRaises(SessionConflictError) as ctx:
            submit_cut_event(s["session_id"], ev(1, 1, 0, "k"))
        self.assertEqual(ctx.exception.detail["error"], "session_completed")

    def test_walk_to_completion(self):
        plan = self._plan()
        s = start_session(plan["plan_id"])
        seq = [(p["panel_no"], c["step"])
               for p in plan["panels"] for c in p["cuts"]]
        self.assertEqual(len(seq), s["total_cuts"])

        for i, (pno, step) in enumerate(seq):
            s = submit_cut_event(s["session_id"], ev(pno, step, i, f"cut-{i}"))
            self.assertEqual(s["version"], i + 1)
            self.assertEqual(s["completed_cuts"], i + 1)
            if i < len(seq) - 1:
                self.assertFalse(s["completed"])
                self.assertEqual(s["next_cut"]["panel_no"], seq[i + 1][0])
                self.assertEqual(s["next_cut"]["step"], seq[i + 1][1])
                self.assertEqual(s["current_panel"], seq[i + 1][0])
        self.assertTrue(s["completed"])
        self.assertEqual(s["status"], "completed")
        self.assertIsNone(s["next_cut"])
        self.assertIsNone(s["current_panel"])
        self.assertIsNotNone(s["completed_at"])
        self.assertEqual(self._event_count(s["session_id"]), len(seq))

        # 完成后新键事件被拒绝，状态不再变化
        with self.assertRaises(SessionConflictError) as ctx:
            submit_cut_event(s["session_id"], ev(*seq[-1], s["version"], "late"))
        self.assertEqual(ctx.exception.detail["error"], "session_completed")
        self.assertEqual(
            get_session_snapshot(s["session_id"])["completed_cuts"], len(seq))

    def test_idempotent_retry_returns_original(self):
        plan = self._plan()
        s = start_session(plan["plan_id"])
        self.assertGreaterEqual(s["total_cuts"], 2)
        sid = s["session_id"]

        nxt = s["next_cut"]
        r1 = submit_cut_event(sid, ev(nxt["panel_no"], nxt["step"], 0, "K"))
        self.assertEqual(r1["version"], 1)
        # 再推进一刀，让会话状态远离首次结果
        nxt2 = r1["next_cut"]
        submit_cut_event(sid, ev(nxt2["panel_no"], nxt2["step"], 1, "K2"))

        # 同键重试：返回首次受理时的快照（version=1），而非当前状态
        r2 = submit_cut_event(sid, ev(nxt["panel_no"], nxt["step"], 0, "K"))
        self.assertEqual(r2, r1)
        # 状态只按真实受理次数推进，事件表也只有两条
        final = get_session_snapshot(sid)
        self.assertEqual(final["completed_cuts"], 2)
        self.assertEqual(final["version"], 2)
        self.assertEqual(self._event_count(sid), 2)

    def test_stale_version_rejected(self):
        plan = self._plan()
        s = start_session(plan["plan_id"])
        sid = s["session_id"]
        nxt = s["next_cut"]
        submit_cut_event(sid, ev(nxt["panel_no"], nxt["step"], 0, "k1"))

        cur = get_session_snapshot(sid)
        with self.assertRaises(SessionConflictError) as ctx:
            submit_cut_event(sid, ev(cur["next_cut"]["panel_no"],
                                     cur["next_cut"]["step"], 0, "k2"))
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], "version_conflict")
        self.assertEqual(detail["current_version"], 1)
        # 状态未推进，事件未记录
        self.assertEqual(get_session_snapshot(sid)["completed_cuts"], 1)
        self.assertEqual(self._event_count(sid), 1)

    def test_skip_step_and_wrong_panel_rejected(self):
        plan = self._plan()
        s = start_session(plan["plan_id"])
        sid = s["session_id"]
        nxt = s["next_cut"]

        with self.assertRaises(SessionConflictError) as ctx:
            submit_cut_event(sid, ev(nxt["panel_no"], nxt["step"] + 1, 0, "k"))
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], "cut_mismatch")
        self.assertEqual(detail["expected"],
                         {"panel_no": nxt["panel_no"], "step": nxt["step"]})
        self.assertEqual(detail["got"],
                         {"panel_no": nxt["panel_no"], "step": nxt["step"] + 1})

        with self.assertRaises(SessionConflictError) as ctx:
            submit_cut_event(sid, ev(nxt["panel_no"] + 1, nxt["step"], 0, "k"))
        self.assertEqual(ctx.exception.detail["error"], "cut_mismatch")

        # 跳步/错刀均未推进状态
        cur = get_session_snapshot(sid)
        self.assertEqual(cur["completed_cuts"], 0)
        self.assertEqual(cur["version"], 0)
        self.assertEqual(self._event_count(sid), 0)

    def test_concurrent_same_version_single_advance(self):
        """N 个线程以同一 expected_version 抢同一刀：恰好一个成功。"""
        plan = self._plan()
        s = start_session(plan["plan_id"])
        sid = s["session_id"]
        nxt = s["next_cut"]
        n_threads = 8
        results = []
        lock = threading.Lock()

        def worker(i):
            try:
                submit_cut_event(sid, ev(nxt["panel_no"], nxt["step"], 0,
                                         f"key-{i}"))
                outcome = ("ok", None)
            except SessionConflictError as exc:
                outcome = ("conflict", exc.detail["error"])
            with lock:
                results.append(outcome)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        oks = [r for r in results if r[0] == "ok"]
        conflicts = [r for r in results if r[0] == "conflict"]
        self.assertEqual(len(oks), 1)
        self.assertEqual(len(conflicts), n_threads - 1)
        self.assertTrue(all(c[1] == "version_conflict" for c in conflicts))
        # 状态恰好推进一刀
        final = get_session_snapshot(sid)
        self.assertEqual(final["completed_cuts"], 1)
        self.assertEqual(final["version"], 1)
        self.assertEqual(self._event_count(sid), 1)

    def test_concurrent_same_key_replays_single_result(self):
        """N 个线程用同一幂等键提交同一刀：全部拿到同一份首次结果。"""
        plan = self._plan()
        s = start_session(plan["plan_id"])
        sid = s["session_id"]
        nxt = s["next_cut"]
        n_threads = 8
        results = []
        lock = threading.Lock()

        def worker():
            r = submit_cut_event(
                sid, ev(nxt["panel_no"], nxt["step"], 0, "same-key"))
            with lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), n_threads)
        for r in results[1:]:
            self.assertEqual(r, results[0])
        # 状态只推进一刀，事件表只有一条
        final = get_session_snapshot(sid)
        self.assertEqual(final["completed_cuts"], 1)
        self.assertEqual(self._event_count(sid), 1)


@unittest.skipUnless(HAS_FASTAPI, "未安装 fastapi，跳过会话 API 测试")
class TestSessionApi(SessionTestBase):
    def setUp(self):
        super().setUp()
        self.client = TestClient(app)

    def _create_plan(self, req=REQ):
        r = self.client.post("/api/v1/plans", json=req)
        self.assertEqual(r.status_code, 200)
        return r.json()

    def test_full_session_flow(self):
        plan = self._create_plan()
        plan_id = plan["plan_id"]
        total = sum(len(p["cuts"]) for p in plan["panels"])
        self.assertGreater(total, 0)

        r = self.client.post(f"/api/v1/plans/{plan_id}/sessions")
        self.assertEqual(r.status_code, 201)
        s = r.json()
        self.assertEqual(s["status"], "active")
        self.assertFalse(s["completed"])
        self.assertEqual(s["version"], 0)
        self.assertEqual(s["completed_cuts"], 0)
        self.assertEqual(s["total_cuts"], total)
        self.assertEqual(s["current_panel"], plan["panels"][0]["panel_no"])
        self.assertEqual(s["next_cut"]["step"], 1)
        sid = s["session_id"]

        # 查询返回相同快照
        r = self.client.get(f"/api/v1/sessions/{sid}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), s)

        # 依次提交每一刀，直到完成
        first_event = None
        for i in range(total):
            cur = self.client.get(f"/api/v1/sessions/{sid}").json()
            nxt = cur["next_cut"]
            r = self.client.post(f"/api/v1/sessions/{sid}/events", json={
                "panel_no": nxt["panel_no"], "step": nxt["step"],
                "expected_version": cur["version"],
                "idempotency_key": f"cut-{i}",
            })
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertEqual(body["version"], i + 1)
            self.assertEqual(body["completed_cuts"], i + 1)
            if i == 0:
                first_event = (nxt, body)
        self.assertTrue(body["completed"])
        self.assertEqual(body["status"], "completed")
        self.assertIsNone(body["next_cut"])
        self.assertIsNone(body["current_panel"])
        self.assertIsNotNone(body["completed_at"])

        # 同键重试第一刀：返回首次受理时的快照（version=1），而非已完成状态
        nxt, first_body = first_event
        r = self.client.post(f"/api/v1/sessions/{sid}/events", json={
            "panel_no": nxt["panel_no"], "step": nxt["step"],
            "expected_version": 0, "idempotency_key": "cut-0",
        })
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), first_body)

        # 完成后新键事件：409 session_completed
        r = self.client.post(f"/api/v1/sessions/{sid}/events", json={
            "panel_no": 1, "step": 1, "expected_version": total,
            "idempotency_key": "extra",
        })
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.json()["detail"]["error"], "session_completed")

    def test_conflicts_do_not_advance(self):
        plan = self._create_plan()
        sid = self.client.post(
            f"/api/v1/plans/{plan['plan_id']}/sessions").json()["session_id"]
        cur = self.client.get(f"/api/v1/sessions/{sid}").json()
        self.assertGreaterEqual(cur["total_cuts"], 2)
        nxt = cur["next_cut"]

        # 跳步：409 cut_mismatch，带期望/实收刀序
        r = self.client.post(f"/api/v1/sessions/{sid}/events", json={
            "panel_no": nxt["panel_no"], "step": nxt["step"] + 1,
            "expected_version": 0, "idempotency_key": "skip",
        })
        self.assertEqual(r.status_code, 409)
        detail = r.json()["detail"]
        self.assertEqual(detail["error"], "cut_mismatch")
        self.assertEqual(detail["expected"],
                         {"panel_no": nxt["panel_no"], "step": nxt["step"]})
        self.assertEqual(detail["got"],
                         {"panel_no": nxt["panel_no"], "step": nxt["step"] + 1})

        # 正确提交一刀
        r = self.client.post(f"/api/v1/sessions/{sid}/events", json={
            "panel_no": nxt["panel_no"], "step": nxt["step"],
            "expected_version": 0, "idempotency_key": "ok-1",
        })
        self.assertEqual(r.status_code, 200)

        # 旧版本：409 version_conflict，带当前版本
        cur = self.client.get(f"/api/v1/sessions/{sid}").json()
        r = self.client.post(f"/api/v1/sessions/{sid}/events", json={
            "panel_no": cur["next_cut"]["panel_no"],
            "step": cur["next_cut"]["step"],
            "expected_version": 0, "idempotency_key": "stale",
        })
        self.assertEqual(r.status_code, 409)
        detail = r.json()["detail"]
        self.assertEqual(detail["error"], "version_conflict")
        self.assertEqual(detail["current_version"], 1)

        # 冲突请求均未推进状态
        cur = self.client.get(f"/api/v1/sessions/{sid}").json()
        self.assertEqual(cur["completed_cuts"], 1)
        self.assertEqual(cur["version"], 1)

    def test_not_found_and_validation(self):
        self.assertEqual(
            self.client.post("/api/v1/plans/nope/sessions").status_code, 404)
        self.assertEqual(self.client.get("/api/v1/sessions/nope").status_code,
                         404)
        r = self.client.post("/api/v1/sessions/nope/events", json={
            "panel_no": 1, "step": 1, "expected_version": 0,
            "idempotency_key": "k",
        })
        self.assertEqual(r.status_code, 404)

        plan = self._create_plan()
        sid = self.client.post(
            f"/api/v1/plans/{plan['plan_id']}/sessions").json()["session_id"]
        bad_payloads = [
            {"panel_no": 1, "step": 1, "expected_version": 0},          # 缺幂等键
            {"panel_no": 1, "step": 0, "expected_version": 0,
             "idempotency_key": "k"},                                    # step 非法
            {"panel_no": 1, "step": 1, "expected_version": -1,
             "idempotency_key": "k"},                                    # 版本非法
            {"panel_no": 1, "step": 1, "expected_version": 0,
             "idempotency_key": ""},                                     # 空幂等键
            {"panel_no": 1, "step": 1, "expected_version": 0,
             "idempotency_key": "k", "extra": 1},                        # 多余字段
        ]
        for bad in bad_payloads:
            self.assertEqual(
                self.client.post(f"/api/v1/sessions/{sid}/events",
                                 json=bad).status_code,
                422, bad)

    def test_empty_plan_session_completed_on_start(self):
        plan = self._create_plan(EXACT_REQ)
        self.assertEqual(sum(len(p["cuts"]) for p in plan["panels"]), 0)
        r = self.client.post(f"/api/v1/plans/{plan['plan_id']}/sessions")
        self.assertEqual(r.status_code, 201)
        s = r.json()
        self.assertEqual(s["status"], "completed")
        self.assertTrue(s["completed"])
        self.assertIsNone(s["current_panel"])
        self.assertIsNone(s["next_cut"])
        self.assertEqual(s["total_cuts"], 0)
        self.assertIsNotNone(s["completed_at"])

    def test_multiple_sessions_per_plan_are_independent(self):
        plan = self._create_plan()
        plan_id = plan["plan_id"]
        s1 = self.client.post(f"/api/v1/plans/{plan_id}/sessions").json()
        s2 = self.client.post(f"/api/v1/plans/{plan_id}/sessions").json()
        self.assertNotEqual(s1["session_id"], s2["session_id"])

        nxt = s1["next_cut"]
        r = self.client.post(f"/api/v1/sessions/{s1['session_id']}/events",
                             json={
                                 "panel_no": nxt["panel_no"],
                                 "step": nxt["step"],
                                 "expected_version": 0,
                                 "idempotency_key": "k",
                             })
        self.assertEqual(r.status_code, 200)
        # 另一会话不受影响
        cur2 = self.client.get(f"/api/v1/sessions/{s2['session_id']}").json()
        self.assertEqual(cur2["completed_cuts"], 0)
        self.assertEqual(cur2["version"], 0)


if __name__ == "__main__":
    unittest.main()
