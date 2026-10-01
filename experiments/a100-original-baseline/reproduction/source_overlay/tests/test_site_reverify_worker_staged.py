"""No-network, no-GPU denial tests for the undeployed remote reverify worker."""

from __future__ import annotations

import unittest
from unittest import mock

from orchestration import site_reverify_worker as worker


class StagedReverifyWorkerTests(unittest.TestCase):
    def test_host_path_is_not_a_remote_worker(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "outside the trusted coordinator repo"):
            worker.preflight("R08")

    def test_remote_identity_has_only_two_staged_points(self) -> None:
        self.assertEqual(set(worker.APPROVED_POINTS), {"R05", "R08"})
        with mock.patch.object(worker, "EXPECTED_REPO", worker.REPO):
            with self.assertRaisesRegex(RuntimeError, "not explicitly approved"):
                worker.preflight("R04")

    def test_quality_helper_source_change_rejected_before_run_key(self) -> None:
        with mock.patch.object(worker, "EXPECTED_REPO", worker.REPO), \
             mock.patch.object(worker, "QUALITY_WORKER_SHA256", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "helper source SHA changed"):
                worker.preflight("R08")

    def test_r08_reverify_pins_post_quality_metadata_not_old_bundle(self) -> None:
        approved = worker.APPROVED_POINTS["R08"]
        old = worker.quality.RUNS["R08"]["remote_bundle_sha256"]["metadata.json"]
        current = worker.quality_preflight_spec("R08", approved)
        self.assertNotEqual(old, approved["metadata.json"])
        self.assertEqual(current["remote_bundle_sha256"], {
            name: approved[name]
            for name in ("launch.sh", "manifest.resolved.toml", "metadata.json")})
        self.assertEqual(current["benchmark_before_sha256"],
                         approved["outputs/benchmark.json"])


if __name__ == "__main__":
    unittest.main()
