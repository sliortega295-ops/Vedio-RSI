"""Scratch-only stale-verdict protection tests; no real Cache run is touched."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestration import site_reverify_output_guard as guard


class ReverifyOutputGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path("/tmp/lyy-experiments/SolAgent/phase3-reverify-guard-tests")
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="stale-verdict-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.runs = Path(self.temp.name) / "runs"
        self.runs.mkdir()
        self.run = self.runs / "exact-real-run"
        self.run.mkdir()
        self.output = self.run / "reverify_verdict.json"

    def test_stale_numeric_file_cannot_survive_failed_shim(self) -> None:
        stale = {"lpips_max": 0.01, "speedup": 1.2, "status": "old_success"}
        self.output.write_text(json.dumps(stale) + "\n")
        predecessor = guard.sha(self.output)
        with mock.patch.object(guard, "RUNS", self.runs):
            receipt = guard.prepare(self.run, expected_predecessor_sha256=predecessor)
            simulated_shim_exit_code = 2
        self.assertEqual(simulated_shim_exit_code, 2)
        self.assertFalse(self.output.exists(), "archived verifier must see no stale numeric file")
        archived = Path(receipt["archive"])
        self.assertEqual(guard.sha(archived), predecessor)
        self.assertEqual(json.loads(archived.read_text()), stale)
        self.assertEqual(receipt["status"], "predecessor_quarantined")
        self.assertEqual(guard.sha(Path(receipt["receipt_path"])), receipt["receipt_sha256"])

    def test_absent_predecessor_stays_absent(self) -> None:
        with mock.patch.object(guard, "RUNS", self.runs):
            receipt = guard.prepare(self.run, expected_predecessor_sha256=None)
        self.assertEqual(receipt["status"], "predecessor_absent")
        self.assertFalse(self.output.exists())
        self.assertIsNone(receipt["archive"])

    def test_wrong_hash_fails_before_move(self) -> None:
        self.output.write_text('{"lpips_max": 0.01}\n')
        with mock.patch.object(guard, "RUNS", self.runs):
            with self.assertRaisesRegex(RuntimeError, "SHA mismatch"):
                guard.prepare(self.run, expected_predecessor_sha256="0" * 64)
        self.assertTrue(self.output.is_file())

    def test_unexpected_predecessor_fails_before_move(self) -> None:
        self.output.write_text('{"lpips_max": 0.01}\n')
        with mock.patch.object(guard, "RUNS", self.runs):
            with self.assertRaisesRegex(RuntimeError, "presence mismatch"):
                guard.prepare(self.run, expected_predecessor_sha256=None)
        self.assertTrue(self.output.is_file())

    def test_symlinked_predecessor_rejected(self) -> None:
        source = Path(self.temp.name) / "elsewhere.json"
        source.write_text('{"lpips_max": 0.01}\n')
        self.output.symlink_to(source)
        with mock.patch.object(guard, "RUNS", self.runs):
            with self.assertRaisesRegex(RuntimeError, "symlinked verdict"):
                guard.prepare(self.run, expected_predecessor_sha256=guard.sha(source))
        self.assertTrue(self.output.is_symlink())
        self.assertTrue(source.is_file())

    def test_other_run_parent_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact real Cache run"):
            guard.prepare(self.run, expected_predecessor_sha256=None)


if __name__ == "__main__":
    unittest.main()
