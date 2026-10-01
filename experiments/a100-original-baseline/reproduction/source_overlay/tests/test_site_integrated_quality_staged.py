"""No-network, no-GPU denial tests for a future composed-run CPU judge."""

from __future__ import annotations

import unittest
from unittest import mock

from orchestration import site_integrated_quality_worker as worker


class IntegratedQualityStagedTests(unittest.TestCase):
    def test_host_path_cannot_run_remote_judge(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "trusted remote coordinator"):
            worker.preflight()

    def test_no_future_run_is_preapproved(self) -> None:
        self.assertIsNone(worker.APPROVED_RUN)
        with mock.patch.object(worker, "REMOTE_REPO", worker.REPO):
            with self.assertRaisesRegex(RuntimeError, "no exact composed run"):
                worker.preflight()

    def test_changed_quality_source_fails_before_run_identity(self) -> None:
        with mock.patch.object(worker, "REMOTE_REPO", worker.REPO), \
             mock.patch.object(worker, "QUALITY_WORKER_SHA256", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "helper source SHA changed"):
                worker.preflight()


if __name__ == "__main__":
    unittest.main()
