"""CPU-only negative tests for the site Phase 2 GPU safety boundary."""

from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from orchestration import site_job, site_job_broker, site_observer, site_remote
from models.sana_video_2b_a100.baseline import gpu_infer


class Phase2WindowTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch_root = Path("/tmp/lyy-experiments/SolAgent")
        scratch_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="phase2-window-test-", dir=scratch_root)
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.auth_path = self.project / "state/phase2_authorization.json"
        self.state_path = self.project / "state/phase2_first_gpu_job.json"
        self.lock_path = self.project / "state/phase2_first_gpu_job.lock"
        self.run_parent = self.project / "repo/runs"
        self.run_parent.mkdir(parents=True)
        for name, value in (
            ("PROJECT", self.project),
            ("PHASE2_AUTH", self.auth_path),
            ("PHASE2_STATE", self.state_path),
            ("PHASE2_LOCK", self.lock_path),
        ):
            patcher = mock.patch.object(site_job, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def authorization(self, *, expires_s: int = 600) -> dict:
        auth = {
            "schema_version": 1,
            "phase": "phase2",
            "status": "armed",
            "remote_project": str(self.project),
            "remote_host": socket.gethostname(),
            "max_window_s": 21600,
            "submit_reserve_s": 1260,
            "gpu_uuids": list(site_job.PHASE2_GPU_UUIDS),
            "excluded_run_names": ["phase1-old-run"],
            "arm_expires_unix_ns": time.time_ns() + expires_s * 10**9,
        }
        self.auth_path.parent.mkdir(parents=True, exist_ok=True)
        self.auth_path.write_text(json.dumps(auth, sort_keys=True) + "\n")
        return auth

    def run_dir(self, name: str = "phase2-new-run", *, tag: str = "phase2") -> Path:
        run = self.run_parent / name
        run.mkdir()
        (run / "manifest.resolved.toml").write_text(
            f'[resolved_profile.env]\nSOL_A100_BUDGET_PHASE = "{tag}"\n'
        )
        return run

    def test_unarmed_never_creates_new_budget_or_changes_phase1(self) -> None:
        old = self.project / "state/first_gpu_job.json"
        old.parent.mkdir(parents=True)
        old.write_text('{"phase":"phase1","immutable":true}\n')
        original = old.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "authorization is absent"):
            site_job.phase2_budget_status()
        self.assertFalse(self.state_path.exists())
        self.assertEqual(old.read_bytes(), original)

    def test_first_job_creates_exact_six_hour_budget_once(self) -> None:
        self.authorization()
        run = self.run_dir()
        first = site_job.phase2_start_budget(run)
        self.assertTrue(first["new_submit_open"])
        state = json.loads(self.state_path.read_text())
        self.assertEqual(state["deadline_unix_ns"] - state["first_gpu_unix_ns"], 21600 * 10**9)
        self.assertEqual(state["first_run_dir"], str(run))
        original = self.state_path.read_bytes()
        second_run = self.run_dir("phase2-second-run")
        site_job.phase2_start_budget(second_run)
        self.assertEqual(self.state_path.read_bytes(), original)
        self.assertFalse((self.project / "state/first_gpu_job.json").exists())

    def test_old_or_untagged_run_is_rejected_before_budget_creation(self) -> None:
        auth = self.authorization()
        old = self.run_dir("phase1-old-run")
        with self.assertRaisesRegex(RuntimeError, "predates authorization"):
            site_job.phase2_run_allowed(old, auth)
        untagged = self.run_dir("phase2-untagged", tag="phase1")
        with self.assertRaisesRegex(RuntimeError, "lacks its exact budget tag"):
            site_job.phase2_start_budget(untagged)
        self.assertFalse(self.state_path.exists())

    def test_expired_arm_and_submit_cutoff_are_fail_closed(self) -> None:
        self.authorization(expires_s=-1)
        run = self.run_dir()
        with self.assertRaises(TimeoutError):
            site_job.phase2_start_budget(run)
        self.assertFalse(self.state_path.exists())
        self.authorization()
        site_job.phase2_start_budget(run)
        state = json.loads(self.state_path.read_text())
        with mock.patch.object(site_job.time, "time_ns", return_value=state["deadline_unix_ns"] - 1259 * 10**9):
            with self.assertRaisesRegex(TimeoutError, "new-submit cutoff"):
                site_job.phase2_start_budget(self.run_dir("phase2-after-cutoff"))

    def test_budget_symlink_and_authorization_mutation_rejected(self) -> None:
        self.authorization()
        run = self.run_dir()
        site_job.phase2_start_budget(run)
        original = self.state_path.read_bytes()
        self.state_path.unlink()
        self.state_path.symlink_to(self.auth_path)
        with self.assertRaisesRegex(RuntimeError, "cannot be a symlink"):
            site_job.phase2_budget_status()
        self.state_path.unlink()
        self.state_path.write_bytes(original)
        self.authorization()
        with self.assertRaisesRegex(RuntimeError, "malformed or mismatched"):
            site_job.phase2_budget_status()

    def test_remote_worker_hard_deadline_kills_only_its_cpu_group(self) -> None:
        self.authorization()
        run = self.run_dir()
        (run / "launch.sh").write_text("#!/bin/bash\nsleep 60\n")
        site_job.phase2_start_budget(run)
        state = json.loads(self.state_path.read_text())
        state["first_gpu_unix_ns"] = time.time_ns() - 21600 * 10**9 + 1_200_000_000
        state["deadline_unix_ns"] = state["first_gpu_unix_ns"] + 21600 * 10**9
        self.state_path.write_text(json.dumps(state, sort_keys=True) + "\n")
        result = site_job.worker(run, run / "launch.sh", "phase2")
        self.assertTrue(result["timed_out_budget"])
        self.assertNotEqual(result["exit_code"], 0)
        self.assertLess(result["duration_s"], 10)

    def test_runner_phase2_cannot_fall_back_to_old_budget(self) -> None:
        old = self.project / "state/first_gpu_job.json"
        old.parent.mkdir(parents=True)
        old.write_text('{"immutable":true}\n')
        with mock.patch.object(gpu_infer, "PROJECT", self.project), mock.patch.dict(
            os.environ, {"SOL_A100_BUDGET_PHASE": "phase2"}
        ):
            with self.assertRaisesRegex(RuntimeError, "receipt is absent"):
                gpu_infer.first_gpu_budget()
        self.assertEqual(old.read_text(), '{"immutable":true}\n')

    def test_host_broker_rejects_phase2_without_arm_before_ssh(self) -> None:
        run = self.run_dir()
        with mock.patch.object(site_job_broker, "PROJECT", self.project), mock.patch.object(
            site_remote, "phase2_budget_status_direct", side_effect=AssertionError("unexpected SSH")
        ):
            with self.assertRaisesRegex(RuntimeError, "authorization is absent on host"):
                site_job_broker.assert_gpu_submit_budget(run)

    def test_host_broker_rejects_remote_authorization_mismatch(self) -> None:
        run = self.run_dir()
        auth = self.authorization()
        auth["remote_project"] = str(site_remote.REMOTE_PROJECT)
        auth["remote_host"] = "job-c6cfcde5-db4b-4527-8eb8-7518319a0277-master-0"
        self.auth_path.write_text(json.dumps(auth, sort_keys=True) + "\n")
        with mock.patch.object(site_job_broker, "PROJECT", self.project), mock.patch.object(
            site_remote,
            "phase2_budget_status_direct",
            return_value={"phase": "phase2", "authorized": True,
                          "authorization_sha256": "0" * 64, "new_submit_open": True},
        ):
            with self.assertRaisesRegex(TimeoutError, "mismatch"):
                site_job_broker.assert_gpu_submit_budget(run)

    def test_host_transport_rejects_stale_phase2_observer(self) -> None:
        reported = {"phase": "phase2", "status": "running", "sample_count": 2,
                    "last_sample_uuid_complete": True, "last_sample_error": None,
                    "last_sample_age_s": 5.0}
        with mock.patch.object(
            site_remote, "trusted_remote_observer_script", return_value=(Path("/fake/observer.py"), "a" * 64)
        ), mock.patch.object(
            site_remote, "ssh", return_value=mock.Mock(stdout=json.dumps(reported) + "\n")
        ):
            with self.assertRaisesRegex(RuntimeError, "stale"):
                site_remote.phase2_observer_audit_direct()

    def test_observer_phase2_status_does_not_read_phase1_state(self) -> None:
        phase1 = self.project / "telemetry/observer"
        phase1.mkdir(parents=True)
        (phase1 / "state.json").write_text('{"pid":99999,"phase":"phase1"}\n')
        stream = io.StringIO()
        with mock.patch.object(site_observer, "PROJECT", self.project), mock.patch.object(
            sys, "argv", ["site_observer.py", "--phase", "phase2", "status"]
        ), contextlib.redirect_stdout(stream):
            site_observer.main()
        self.assertEqual(json.loads(stream.getvalue())["status"], "not_started")
        self.assertTrue((phase1 / "state.json").is_file())


if __name__ == "__main__":
    unittest.main()
