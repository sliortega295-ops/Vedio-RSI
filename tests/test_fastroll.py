from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from fastroll.cli import expand_candidate_matrix
from fastroll.contracts import ContractError, atomic_json, digest, pin_source, pin_assets, read_json, sha256, validate_plan, verify_source, verify_assets
from fastroll.resources import Authority, FileLock
from fastroll.runtime import Controller, _residency_check, _session_env
from fastroll.store import Store


def task(name="a", **kw):
    return {"id": name, "role": "generate", "resource": "gpu", "session": {"backend": "demo"}, "payload": {"seed": 42}, **kw}


def plan(*tasks, name="batch"):
    return {"schema_version": 1, "batch_id": name, "tasks": list(tasks)}


class Contracts(unittest.TestCase):
    def test_defaults_and_stable_hash(self):
        p = validate_plan(plan(task()))
        self.assertEqual(p, validate_plan(p))
        self.assertEqual(p["tasks"][0]["mode"], "oneshot")

    def test_invalid_plans(self):
        invalid = [plan(task("../x")), plan(task(), task()), plan(task(depends_on=["missing"])),
                   plan(task("a", depends_on=["b"]), task("b", depends_on=["a"])),
                   plan(task(timeout_s=0)), plan(task(priority=True)), plan(task(retries=4)),
                   plan(task(reuse=True)), plan(task(role="primary", mode="resident")),
                   plan(task(session={"backend": "command", "config": {"argv": ["true"]}})),
                   plan(task(inputs={"x": {"task": "a", "artifact": "video"}})),
                   plan(task(unknown=True)), plan(task(resource="tpu")), plan(task(mode="magic"))]
        for p in invalid:
            with self.subTest(p=p), self.assertRaises((ContractError, ValueError)):
                validate_plan(p)

    def test_matrix_parallel_samples_reference_dedup(self):
        spec = {"batch_id": "matrix", "samples": [{"prompt": "a", "seed": 1}, {"prompt": "b", "seed": 2}],
                "candidates": [{"id": "a", "session": {"backend": "demo"}}, {"id": "b", "session": {"backend": "demo"}}],
                "reference_session": {"backend": "demo"}}
        p = expand_candidate_matrix(spec)
        self.assertEqual(len(p["tasks"]), 8)
        self.assertEqual(sum(t["role"] == "reference_generate" for t in p["tasks"]), 2)
        a = next(t for t in p["tasks"] if t["id"] == "a-s1")
        self.assertEqual(a["depends_on"], ["a-primary"])

    def test_environment_strips_credentials_and_hides_cpu_cuda(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ["FASTROLL_TEST_SECRET"] = "not-for-workers"
            env = _session_env({"backend": "demo", "config": {}}, {"uuid": "CPU-0", "real": False}, Path(d))
            self.assertNotIn("FASTROLL_TEST_SECRET", env)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")
            del os.environ["FASTROLL_TEST_SECRET"]

    def test_environment_cannot_override_device(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ContractError):
            _session_env({"config": {"env": {"CUDA_VISIBLE_DEVICES": "0,1"}}}, {"uuid": "CPU-0"}, Path(d))

    def test_asset_mutation_and_inventory_drift(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "w").write_text("weights")
            manifest = pin_assets(root)
            verify_assets([manifest])
            (root / "w").write_text("changed")
            with self.assertRaises(ContractError):
                verify_assets([manifest])
            (root / "w").write_text("weights")
            (root / "extra").write_text("new")
            with self.assertRaises(ContractError):
                verify_assets([manifest])

    def test_qualification_creates_bound_certificate(self):
        from fastroll.qualification import qualify, verify_certificate
        with tempfile.TemporaryDirectory() as d:
            result = qualify({"backend": "demo"}, [{"prompt":"A","seed":1}, {"prompt":"B","seed":2}], Path(d), authorization=None, simulate_gpus=1)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(len(result["evidence"]), 6)
            self.assertTrue(result["simulation"])
            session = {"backend": "demo", "config": {}, "assets": []}
            verify_certificate(result, session)
            with self.assertRaises(ContractError):
                verify_certificate({**result, "evidence": result["evidence"][:-1]}, session)

    def test_real_residency_requires_certificate(self):
        with self.assertRaises(ContractError):
            _residency_check({"mode": "resident", "session": {"backend": "sana", "config": {}}})


class Runtime(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / "workspace")

    def tearDown(self):
        self.tmp.cleanup()

    def run_plan(self, p, gpus=2, **kw):
        self.store.submit(p)
        return Controller(self.store, simulate_gpus=gpus, cpu_workers=1, idle_exit_s=0.1, **kw).drain()

    def test_real_process_parallelism_and_replay(self):
        report = self.run_plan(plan(task("a", payload={"sleep_s": 0.2}), task("b", payload={"sleep_s": 0.2})))
        self.assertEqual(report["tasks"]["succeeded"], 2)
        devices = {r["result"]["provenance"]["device"] for r in report["task_results"].values()}
        self.assertEqual(devices, {"SIM-0", "SIM-1"})
        self.assertEqual(self.store.verify()["status"], "PASS")
        self.assertEqual(report["execution_mode"], "simulation")

    def test_native_stdout_does_not_corrupt_protocol(self):
        report = self.run_plan(plan(task(payload={"native_stdout": True})))
        self.assertEqual(report["tasks"]["succeeded"], 1)

    def test_resident_reset_aba(self):
        rows = [task("a", mode="resident", payload={"prompt": "A", "seed": 1}),
                task("b", mode="resident", payload={"prompt": "B", "seed": 2}, depends_on=["a"]),
                task("c", mode="resident", payload={"prompt": "A", "seed": 1}, depends_on=["b"])]
        result = self.run_plan(plan(*rows), gpus=1)["task_results"]
        self.assertEqual(result["batch/c"]["result"]["metrics"]["session_requests"], 3)
        self.assertEqual(result["batch/a"]["result"]["artifacts"]["video"]["sha256"], result["batch/c"]["result"]["artifacts"]["video"]["sha256"])

    def test_reference_cache_reuses_producer_not_candidate(self):
        a = task("a", role="reference_generate", reuse=True)
        b = task("b", role="reference_generate", reuse=True, depends_on=["a"])
        report = self.run_plan(plan(a, b), gpus=1)
        self.assertEqual(report["cache_hits"], 1)
        self.assertEqual(report["task_results"]["batch/b"]["attempts"], 0)
        self.assertEqual(report["task_results"]["batch/b"]["result"]["provenance"]["reused_from"], "batch/a")
        self.store.verify()

    def test_reference_singleflight_deduplicates_ready_tasks(self):
        report = self.run_plan(plan(task("a", role="reference_generate", reuse=True), task("b", role="reference_generate", reuse=True)), gpus=1)
        self.assertEqual(report["cache_hits"], 1)

    def test_reference_cache_changed_payload_misses(self):
        report = self.run_plan(plan(task("a", role="reference_generate", reuse=True), task("b", role="reference_generate", reuse=True, payload={"seed": 43}, depends_on=["a"])), gpus=1)
        self.assertEqual(report["cache_hits"], 0)

    def test_cache_artifact_tamper_fails_closed(self):
        self.run_plan(plan(task("a", role="reference_generate", reuse=True)), gpus=1)
        output = self.store.rows()[0]["result"]["artifacts"]["video"]["path"]
        Path(output).write_text("tampered")
        with self.assertRaises(ContractError):
            self.store.verify()

    def test_failed_dependency_blocked(self):
        report = self.run_plan(plan(task("a", payload={"fail": True}), task("b", depends_on=["a"])))
        self.assertEqual(report["tasks"]["failed"], 1)
        self.assertEqual(report["tasks"]["blocked"], 1)
        self.store.verify()

    def test_timeout_terminates_worker(self):
        report = self.run_plan(plan(task(timeout_s=0.1, payload={"sleep_s": 1})))
        self.assertEqual(report["tasks"]["failed"], 1)
        self.assertIn("deadline", report["task_results"]["batch/a"]["error"])

    def test_retry_is_bounded_and_recorded(self):
        report = self.run_plan(plan(task(payload={"fail": True}, retries=1)))
        self.assertEqual(report["task_results"]["batch/a"]["attempts"], 2)
        self.assertEqual(report["tasks"]["failed"], 1)
        self.store.verify()

    def test_cancellation_propagates_to_active_task(self):
        self.store.submit(plan(task(payload={"sleep_s": 1.5})))
        controller = Controller(self.store, simulate_gpus=1, idle_exit_s=0.1)
        thread = threading.Thread(target=controller.drain)
        thread.start()
        for _ in range(100):
            if self.store.rows()[0]["state"] == "running":
                break
            time.sleep(0.02)
        self.store.cancel()
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.store.rows()[0]["state"], "cancelled")
        self.store.verify()

    def test_queue_accepts_submission_during_drain(self):
        self.store.submit(plan(task(payload={"sleep_s": 0.3})))
        controller = Controller(self.store, simulate_gpus=2, idle_exit_s=0.5)
        thread = threading.Thread(target=controller.drain)
        thread.start()
        time.sleep(0.05)
        self.store.submit(plan(task("late"), name="second"))
        thread.join(5)
        self.assertEqual(self.store.report()["tasks"]["succeeded"], 2)

    def test_recovery_does_not_redo_completed_task(self):
        self.run_plan(plan(task()), gpus=1)
        self.store.submit(plan(task("b"), name="second"))
        self.store.transition("second/b", {"queued"}, "running")
        Controller(self.store, simulate_gpus=1, idle_exit_s=0.1).drain()
        rows = {r["id"]: r for r in self.store.rows()}
        self.assertEqual(rows["batch/a"]["attempts"], 1)
        self.assertEqual(rows["second/b"]["attempts"], 2)
        self.store.verify()

    def test_submit_idempotent_and_conflict_rejected(self):
        self.store.submit(plan(task()))
        self.store.submit(plan(task()))
        with self.assertRaises(ContractError):
            self.store.submit(plan(task(payload={"seed": 0})))
        self.assertEqual(len(self.store.rows()), 1)

    def test_database_mutation_not_accepted_as_result(self):
        self.store.submit(plan(task()))
        with self.store.connect() as db:
            db.execute("UPDATE tasks SET state='succeeded'")
        with self.assertRaises(ContractError):
            self.store.verify()

    def test_event_mutation_fails_hash_chain(self):
        self.store.submit(plan(task()))
        with self.store.connect() as db:
            db.execute("UPDATE events SET data='{}'")
        with self.assertRaises(ContractError):
            self.store.verify()

    def test_controller_lock_excludes_second_controller(self):
        lock = FileLock(self.store.root / "controller.lock")
        try:
            with self.assertRaises(BlockingIOError):
                Controller(self.store, simulate_gpus=1).drain()
        finally:
            lock.close()

    def test_missing_authority_never_launches(self):
        with self.assertRaises(PermissionError):
            Controller(self.store)

    def test_authority_revocation_and_expiry(self):
        path = self.root / "auth.json"
        value = {"authorized": True, "issuer": "test-only", "workspace": str(self.store.root), "gpu_uuids": [],
                 "approved_sessions": ["x"], "source_roots": [], "not_before": time.time()-1,
                 "expires_at": time.time()+60, "lock_root": str(self.root / "locks")}
        atomic_json(path, value)
        a = Authority(path, self.store.root)
        atomic_json(path, {**value, "authorized": False})
        with self.assertRaises(PermissionError):
            a.check()
        atomic_json(path, {**value, "expires_at": time.time()-1})
        with self.assertRaises(PermissionError):
            Authority(path, self.store.root)

    def test_task_affinity_does_not_block_other_ready_work(self):
        report = self.run_plan(plan(task("a", device="SIM-9"), task("b", device="SIM-1")))
        self.assertEqual(report["tasks"]["succeeded"], 1)
        self.assertEqual(report["tasks"]["queued"], 1)

    def test_primary_is_always_cold(self):
        report = self.run_plan(plan(task("a", role="primary"), task("b", role="primary", depends_on=["a"])), gpus=1)
        for row in report["task_results"].values():
            self.assertEqual(row["result"]["metrics"]["session_requests"], 1)
            self.assertGreater(row["result"]["metrics"]["primary_wall_s"], 0)

    def test_compare_uses_real_predecessor_artifact(self):
        compare = task("check", role="validate", resource="cpu", session={"backend": "compare"},
                       payload={"checks": [{"input": "v", "key": "value", "op": "le", "limit": 1}]},
                       depends_on=["a"], inputs={"v": {"task": "a", "artifact": "video"}})
        report = self.run_plan(plan(task(), compare))
        self.assertTrue(report["task_results"]["batch/check"]["result"]["metrics"]["quality_pass"])


class PinnedCommand(unittest.TestCase):
    def test_actual_command_adapter_and_source_drift(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "source"
            source.mkdir()
            script = source / "adapter.py"
            script.write_text('import json,sys,pathlib\nr=json.load(open(sys.argv[1]))\np=pathlib.Path(r["output_dir"])/"evidence.json"\np.write_text(json.dumps({"value":.2}))\njson.dump({"request_id":r["request_id"],"metrics":{"ok":True},"artifacts":{"evidence":str(p)}},open(sys.argv[2],"w"))\n')
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "add", "."], check=True)
            subprocess.run(["git", "-C", str(source), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
            pinned = pin_source(source)
            verify_source(pinned)
            session = {"backend": "command", "assets": [], "source": pinned, "config": {"argv": [sys.executable, str(script), "{request}", "{result}"]}}
            store = Store(root / "workspace")
            store.submit(plan(task(role="score", resource="cpu", session=session)))
            auth = root / "auth.json"
            atomic_json(auth, {"authorized": True, "issuer": "CPU unit test only", "workspace": str(store.root), "gpu_uuids": [],
                               "approved_sessions": [digest(session)], "source_roots": [str(source)],
                               "not_before": time.time()-1, "expires_at": time.time()+60, "lock_root": str(root / "locks")})
            report = Controller(store, authorization=auth, idle_exit_s=0.1).drain()
            self.assertEqual(report["tasks"]["succeeded"], 1)
            store.verify()
            script.write_text("changed")
            with self.assertRaises(ContractError):
                verify_source(pinned)


if __name__ == "__main__":
    unittest.main()
