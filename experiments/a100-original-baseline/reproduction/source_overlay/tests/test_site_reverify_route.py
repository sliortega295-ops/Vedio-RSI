"""CPU-only fail-closed tests for the future exact Cache verifier transport."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestration import site_job_broker as broker
from orchestration import site_reverify_python as shim
from orchestration import site_reverify_transport as route


R01 = route.RUNS / "20260913-095257-cache_ca_r01_easycache_t010"


def exact_argv(run: Path = R01) -> list[str]:
    return ["search/plan_eval.py", "--model", "sana_video_2b_a100",
            "--no-gemini", "--assess", str(run), "--out",
            str(run / "reverify_verdict.json"), "--baseline-frames",
            str(shim.DENSE_FRAMES)]


def request(run: Path = R01) -> dict:
    script = run / "launch.sh"
    return {"command": "reassess_quality", "run_dir": str(run),
            "launch_script": str(script), **broker.stamps(run, script)}


class SiteReverifyRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path("/tmp/lyy-experiments/SolAgent")
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="phase3-reverify-test-", dir=scratch)
        self.addCleanup(self.temp.cleanup)

    def test_exact_archived_cli_shape(self) -> None:
        self.assertEqual(shim.parse_exact(exact_argv()), R01)

    def test_extra_flag_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-archived"):
            shim.parse_exact(exact_argv() + ["--no-refresh-collection"])

    def test_wrong_model_rejected(self) -> None:
        argv = exact_argv()
        argv[2] = "different_model"
        with self.assertRaisesRegex(ValueError, "non-archived"):
            shim.parse_exact(argv)

    def test_wrong_out_rejected(self) -> None:
        argv = exact_argv()
        argv[7] = str(R01 / "outputs/quality.json")
        with self.assertRaisesRegex(ValueError, "exact Cache run/output"):
            shim.parse_exact(argv)

    def test_missing_delivery_denied_before_remote(self) -> None:
        missing = Path(self.temp.name) / "missing_delivery.json"
        with mock.patch.object(route, "DELIVERY", missing):
            with self.assertRaisesRegex(RuntimeError, "DELIVERY absent"):
                route.assess(R01)

    def test_unapproved_delivery_denied_before_remote(self) -> None:
        delivery = Path(self.temp.name) / "DELIVERY.json"
        delivery.write_text(json.dumps({"schema_version": 2, "status": "complete",
                                        "component": "cache", "model_id": route.MODEL_ID,
                                        "frontier_points": [{"run_dir": f"runs/{R01.name}"}]}))
        with mock.patch.object(route, "DELIVERY", delivery):
            with self.assertRaisesRegex(RuntimeError, "not approved"):
                route.assess(R01)

    def test_coordinator_queue_fails_closed_without_delivery(self) -> None:
        missing = Path(self.temp.name) / "missing_delivery.json"
        with mock.patch.object(route, "DELIVERY", missing):
            with self.assertRaisesRegex(RuntimeError, "DELIVERY absent"):
                broker.dispatch(request(), broker.QUEUE)

    def test_executor_queue_cannot_independently_reverify(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot originate in an Executor"):
            broker.dispatch(request(), broker.EXPERIMENT_QUEUES["sana2ba100-cache_ca-0002"])

    def test_pulled_post_must_match_remote_receipt_before_any_sync(self) -> None:
        root = Path(self.temp.name)
        run, attempt = root / "run", root / "attempt"
        prior = {}
        remote = {}
        for relative in route.SYNC_FILES:
            local, pulled = run / relative, attempt / "post" / relative
            local.parent.mkdir(parents=True, exist_ok=True)
            pulled.parent.mkdir(parents=True, exist_ok=True)
            local.write_text("old\n")
            pulled.write_text("new\n")
            prior[relative] = route.sha(local)
            remote[relative] = {"sha256": route.sha(pulled)}
        # The apparent valid remote receipt lies about exactly one small file.
        remote["outputs/quality.json"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "differs from trusted receipt"):
            route.sync_small_post(run, attempt, prior, remote, 1.0)
        for relative in route.SYNC_FILES:
            self.assertEqual(route.sha(run / relative), prior[relative])

    def test_approved_predecessor_mismatch_stops_before_remote_run(self) -> None:
        spec = {"predecessor_sha256": {"outputs/quality.json": "0" * 64},
                "delivery_sha256": "1" * 64, "run_key": "R01", "total_s": 375.18}
        with mock.patch.object(route, "assert_approved", return_value={}), \
             mock.patch.object(route, "APPROVED_POINTS", {R01.name: spec}), \
             mock.patch.object(route, "trusted_remote_worker", return_value={}), \
             mock.patch.object(route, "snapshot_local", return_value=(Path(self.temp.name),
                                                                       {"outputs/quality.json": "2" * 64})), \
             mock.patch.object(route, "remote_run") as remote:
            with self.assertRaisesRegex(RuntimeError, "pre-snapshot differs"):
                route.assess(R01)
            remote.assert_not_called()


if __name__ == "__main__":
    unittest.main()
