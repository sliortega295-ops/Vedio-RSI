"""SQLite queue, atomic transitions, and a replayable hash-chained event stream.

Execution is at-least-once after a crash. Terminal result commits are idempotent;
this module deliberately does NOT claim exactly-once external side effects.
"""
from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from pathlib import Path

from .contracts import ContractError, TERMINAL, canonical, digest, validate_plan, verify_result


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "queue.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS batches(id TEXT PRIMARY KEY, hash TEXT, plan TEXT);
                CREATE TABLE IF NOT EXISTS tasks(
                    id TEXT PRIMARY KEY, batch TEXT, spec TEXT, state TEXT,
                    attempts INTEGER DEFAULT 0, ready_at REAL, submitted REAL,
                    result TEXT, error TEXT);
                CREATE TABLE IF NOT EXISTS events(
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, time REAL, kind TEXT,
                    task TEXT, data TEXT, previous TEXT, hash TEXT);
                CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, result TEXT, producer TEXT);
            """)

    @contextlib.contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def event(self, db, kind: str, task: str, data: dict):
        last = db.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        previous = last[0] if last else "0" * 64
        now = time.time()
        value = {"time": now, "kind": kind, "task": task, "data": data, "previous": previous}
        db.execute("INSERT INTO events(time,kind,task,data,previous,hash) VALUES(?,?,?,?,?,?)",
                   (now, kind, task, canonical(data).decode(), previous, digest(value)))

    def submit(self, raw: dict) -> str:
        p = validate_plan(raw)
        key = p["batch_id"]
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT hash FROM batches WHERE id=?", (key,)).fetchone()
            if old:
                if old[0] != digest(p):
                    raise ContractError("batch_id already exists with different content")
                return key
            db.execute("INSERT INTO batches VALUES(?,?,?)", (key, digest(p), canonical(p).decode()))
            for t in p["tasks"]:
                tid = key + "/" + t["id"]
                now = time.time()
                db.execute("INSERT INTO tasks(id,batch,spec,state,ready_at,submitted) VALUES(?,?,?,?,?,?)",
                           (tid, key, canonical(t).decode(), "queued", now, now))
                self.event(db, "submitted", tid, {"spec_hash": digest(t), "state": "queued"})
        return key

    def rows(self) -> list[dict]:
        with self.connect() as db:
            rows = [dict(r) for r in db.execute("SELECT * FROM tasks ORDER BY submitted,id")]
        for r in rows:
            r["spec"] = json.loads(r["spec"])
            r["result"] = json.loads(r["result"]) if r["result"] else None
        return rows

    def transition(self, tid: str, expected: set[str], state: str, data: dict | None = None) -> bool:
        data = data or {}
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone()
            if not row or row["state"] not in expected:
                return False
            attempts = row["attempts"] + (state == "running")
            result = canonical(data["result"]).decode() if "result" in data else row["result"]
            error = None if state == "succeeded" else data.get("error", row["error"])
            db.execute("UPDATE tasks SET state=?,attempts=?,ready_at=?,result=?,error=? WHERE id=?",
                       (state, attempts, data.get("ready_at", time.time()), result, error, tid))
            self.event(db, "transition", tid, {**data, "state": state, "attempts": attempts})
            return True

    def annotate(self, kind: str, tid: str, data: dict):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self.event(db, kind, tid, data)

    def cancel(self, batch: str | None = None):
        for row in self.rows():
            if batch is None or row["batch"] == batch:
                self.transition(row["id"], {"queued", "running"}, "cancelled", {"error": "operator cancellation"})

    def cache_get(self, key: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT result,producer FROM cache WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        result = json.loads(row[0])
        with self.connect() as db:
            producer = db.execute("SELECT state,result FROM tasks WHERE id=?", (row[1],)).fetchone()
        if not producer or producer["state"] != "succeeded" or json.loads(producer["result"]) != result:
            raise ContractError("cache does not bind a successful producer receipt")
        verify_result(result, self.root)
        return {"result": result, "producer": row[1]}

    def cache_put(self, key: str, result: dict, producer: str):
        verify_result(result, self.root)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR IGNORE INTO cache VALUES(?,?,?)", (key, canonical(result).decode(), producer))

    def recover(self):
        # Caller must hold the workspace controller lock and clean up old children.
        for row in self.rows():
            if row["state"] == "running":
                self.transition(row["id"], {"running"}, "queued", {"error": "interrupted attempt; executing again, not claiming exactly-once"})

    def report(self) -> dict:
        rows = self.rows()
        with self.connect() as db:
            events = [dict(r) for r in db.execute("SELECT * FROM events ORDER BY seq")]
        attempts = [json.loads(e["data"]) for e in events if e["kind"] == "attempt_finished"]
        counts = {state: sum(r["state"] == state for r in rows) for state in ("queued", "running", *sorted(TERMINAL))}
        gpu = sum(a.get("wall_s", 0) for a in attempts if a.get("resource") == "gpu")
        mode = [json.loads(e["data"]).get("simulation") for e in events if e["kind"] == "controller_started"]
        return {"schema_version": 1, "execution_mode": ("simulation" if mode[-1] else "real") if mode else "not_run", "performance_claim": False, "tasks": counts,
                "attempt_gpu_hours": gpu / 3600, "measurement": "task wall occupancy, not SM utilization",
                "cache_hits": sum(e["kind"] == "cache_hit" for e in events),
                "task_results": {r["id"]: {"state": r["state"], "attempts": r["attempts"], "error": r["error"], "result": r["result"]} for r in rows}}

    def verify(self) -> dict:
        with self.connect() as db:
            db.execute("BEGIN")
            batches = list(db.execute("SELECT * FROM batches"))
            events = list(db.execute("SELECT * FROM events ORDER BY seq"))
            task_rows = [dict(r) for r in db.execute("SELECT * FROM tasks ORDER BY submitted,id")]
        specs = {}
        for batch in batches:
            plan = validate_plan(json.loads(batch["plan"]))
            if digest(plan) != batch["hash"] or plan["batch_id"] != batch["id"]:
                raise ContractError("stored plan changed")
            specs.update({batch["id"] + "/" + t["id"]: t for t in plan["tasks"]})
        previous = "0" * 64
        states, results, attempts = {}, {}, {}
        for e in events:
            data = json.loads(e["data"])
            obj = {"time": e["time"], "kind": e["kind"], "task": e["task"], "data": data, "previous": previous}
            if previous != e["previous"] or digest(obj) != e["hash"]:
                raise ContractError("event chain corrupted")
            if e["kind"] == "submitted":
                if e["task"] not in specs or data["spec_hash"] != digest(specs[e["task"]]) or e["task"] in states:
                    raise ContractError("submission does not bind plan")
                states[e["task"]] = "queued"
                attempts[e["task"]] = 0
            if e["kind"] == "transition":
                if e["task"] not in states or states[e["task"]] in TERMINAL:
                    raise ContractError("illegal transition after terminal result")
                states[e["task"]] = data["state"]
                attempts[e["task"]] = data["attempts"]
                if "result" in data:
                    verify_result(data["result"], self.root)
                    results[e["task"]] = data["result"]
            previous = e["hash"]
        rows = task_rows
        for row in rows:
            row["spec"] = json.loads(row["spec"])
            row["result"] = json.loads(row["result"]) if row["result"] else None
        if set(states) != {r["id"] for r in rows} or set(states) != set(specs):
            raise ContractError("task inventory changed")
        for r in rows:
            if r["spec"] != specs[r["id"]] or r["state"] != states[r["id"]] or r["attempts"] != attempts[r["id"]] or r["result"] != results.get(r["id"]):
                raise ContractError("queue state does not replay from events")
        return {"status": "PASS", "tasks": len(rows), "events": len(events), "chain_head": previous,
                "boundary": "integrity/replay check, not a trusted-code sandbox or proof of model quality"}
