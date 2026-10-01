"""CPU-only negative tests for the exact Phase 3 Cache quality transport."""

from __future__ import annotations

import importlib.util
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestration import site_job_broker, site_quality_transport, site_quality_worker


CACHE_RUNS = site_quality_transport.CACHE / "runs"


class SiteQualityRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path("/tmp/lyy-experiments/SolAgent")
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="phase3-quality-test-", dir=scratch)
        self.addCleanup(self.temp.cleanup)

    def run_path(self, key: str) -> Path:
        return CACHE_RUNS / site_quality_transport.EXPECTED_BUNDLES[key]["name"]

    def request(self, key: str) -> dict:
        run = self.run_path(key)
        return {"command": "assess_quality", "run_dir": str(run),
                "launch_script": str(run / "launch.sh"),
                **site_job_broker.stamps(run, run / "launch.sh")}

    def test_wrong_run_id_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "real seq0002 Cache run parent"):
            site_quality_transport.key_for_run(CACHE_RUNS / "new-candidate")

    def test_r02_second_assessment_requires_trusted_post_chain(self) -> None:
        with self.assertRaisesRegex(ValueError, "benchmark.json"):
            site_quality_transport.validate_local(self.run_path("R02"))

    def test_r03_exact_bundle_matches_after_review(self) -> None:
        with self.assertRaisesRegex(ValueError, "benchmark.json"):
            site_quality_transport.validate_local(self.run_path("R03"))

    def test_r04_second_assessment_requires_trusted_post_chain(self) -> None:
        self.assertEqual(site_quality_transport.EXPECTED_BUNDLES["R04"]["total_s"], 382.27)
        with self.assertRaisesRegex(ValueError, "benchmark.json"):
            site_quality_transport.validate_local(self.run_path("R04"))

    def test_r04_bad_benchmark_sha_rejected(self) -> None:
        altered = dict(site_quality_transport.EXPECTED_BUNDLES)
        altered["R04"] = {**altered["R04"], "benchmark_sha256": "0" * 64}
        with mock.patch.object(site_quality_transport, "EXPECTED_BUNDLES", altered):
            with self.assertRaisesRegex(ValueError, "benchmark.json"):
                site_quality_transport.validate_local(self.run_path("R04"))

    def test_r05_second_assessment_requires_trusted_post_chain(self) -> None:
        self.assertEqual(site_quality_transport.EXPECTED_BUNDLES["R05"]["total_s"], 371.36)
        with self.assertRaisesRegex(ValueError, "benchmark.json"):
            site_quality_transport.validate_local(self.run_path("R05"))

    def test_r05_bad_benchmark_sha_rejected(self) -> None:
        altered = dict(site_quality_transport.EXPECTED_BUNDLES)
        altered["R05"] = {**altered["R05"], "benchmark_sha256": "0" * 64}
        with mock.patch.object(site_quality_transport, "EXPECTED_BUNDLES", altered):
            with self.assertRaisesRegex(ValueError, "benchmark.json"):
                site_quality_transport.validate_local(self.run_path("R05"))

    def test_r06_second_assessment_requires_trusted_post_chain(self) -> None:
        self.assertEqual(site_quality_transport.EXPECTED_BUNDLES["R06"]["total_s"], 480.37)
        with self.assertRaisesRegex(ValueError, "benchmark.json"):
            site_quality_transport.validate_local(self.run_path("R06"))
        self.assertEqual(site_quality_worker.RUNS["R06"]["total_s"], 480.37)
        digest, frames = site_quality_worker.frames_digest(self.run_path("R06") / "outputs/frames")
        self.assertEqual(digest, site_quality_worker.RUNS["R06"]["frames_digest"])
        self.assertEqual(len(frames), 80)

    def test_r06_bad_benchmark_sha_rejected(self) -> None:
        altered = dict(site_quality_transport.EXPECTED_BUNDLES)
        altered["R06"] = {**altered["R06"], "benchmark_sha256": "0" * 64}
        with mock.patch.object(site_quality_transport, "EXPECTED_BUNDLES", altered):
            with self.assertRaisesRegex(ValueError, "benchmark.json"):
                site_quality_transport.validate_local(self.run_path("R06"))

    def test_r07_second_assessment_requires_trusted_post_chain(self) -> None:
        with self.assertRaisesRegex(ValueError, "benchmark.json"):
            site_quality_transport.validate_local(self.run_path("R07"))
        self.assertEqual(site_quality_worker.RUNS["R07"]["total_s"], 480.18)
        digest, frames = site_quality_worker.frames_digest(self.run_path("R07") / "outputs/frames")
        self.assertEqual(digest, site_quality_worker.RUNS["R07"]["frames_digest"])
        self.assertEqual(len(frames), 80)

    def test_r07_bad_benchmark_sha_rejected_before_remote(self) -> None:
        altered = dict(site_quality_transport.EXPECTED_BUNDLES)
        altered["R07"] = {**altered["R07"], "benchmark_sha256": "0" * 64}
        with mock.patch.object(site_quality_transport, "EXPECTED_BUNDLES", altered):
            with mock.patch.object(site_quality_transport.site_remote, "ssh",
                                   side_effect=AssertionError("unexpected SSH")):
                with self.assertRaisesRegex(ValueError, "benchmark.json"):
                    site_quality_transport.validate_local(self.run_path("R07"))

    def test_r08_staged_local_bundle_and_80_frames_match(self) -> None:
        key, spec = site_quality_transport.validate_local(self.run_path("R08"))
        self.assertEqual(key, "R08")
        self.assertEqual(spec["total_s"], 480.96)
        self.assertEqual(site_quality_worker.RUNS["R08"]["total_s"], 480.96)
        digest, frames = site_quality_worker.frames_digest(self.run_path("R08") / "outputs/frames")
        self.assertEqual(digest, site_quality_worker.RUNS["R08"]["frames_digest"])
        self.assertEqual(len(frames), 80)
        self.assertFalse((self.run_path("R08") / "assess_verdict.json").exists())

    def test_r08_bad_benchmark_sha_rejected_before_remote(self) -> None:
        altered = dict(site_quality_transport.EXPECTED_BUNDLES)
        altered["R08"] = {**altered["R08"], "benchmark_sha256": "0" * 64}
        with mock.patch.object(site_quality_transport, "EXPECTED_BUNDLES", altered):
            with mock.patch.object(site_quality_transport.site_remote, "ssh",
                                   side_effect=AssertionError("unexpected SSH")):
                with self.assertRaisesRegex(ValueError, "benchmark.json"):
                    site_quality_transport.validate_local(self.run_path("R08"))

    def test_r08_unexpected_assess_predecessor_rejected(self) -> None:
        with mock.patch.object(Path, "exists", autospec=True,
                               side_effect=lambda path: path.name == "assess_verdict.json"):
            with self.assertRaisesRegex(ValueError, "expected absent predecessor: assess_verdict.json"):
                site_quality_transport.validate_local(self.run_path("R08"))

    def test_r03_bad_benchmark_sha_rejected(self) -> None:
        altered = dict(site_quality_transport.EXPECTED_BUNDLES)
        altered["R03"] = {**altered["R03"], "benchmark_sha256": "0" * 64}
        with mock.patch.object(site_quality_transport, "EXPECTED_BUNDLES", altered):
            with self.assertRaisesRegex(ValueError, "benchmark.json"):
                site_quality_transport.validate_local(self.run_path("R03"))

    def test_r01_second_assessment_requires_trusted_post_chain(self) -> None:
        with self.assertRaisesRegex(ValueError, "benchmark.json"):
            site_quality_transport.validate_local(self.run_path("R01"))

    def test_r01_bad_benchmark_sha_rejected(self) -> None:
        altered = dict(site_quality_transport.EXPECTED_BUNDLES)
        altered["R01"] = {**altered["R01"], "benchmark_sha256": "0" * 64}
        with mock.patch.object(site_quality_transport, "EXPECTED_BUNDLES", altered):
            with self.assertRaisesRegex(ValueError, "benchmark.json"):
                site_quality_transport.validate_local(self.run_path("R01"))

    def test_bad_trusted_worker_sha_rejected_before_ssh(self) -> None:
        with mock.patch.object(site_quality_transport, "TRUSTED_WORKER_SHA256", "0" * 64):
            with mock.patch.object(site_quality_transport.site_remote, "ssh",
                                   side_effect=AssertionError("unexpected SSH")):
                with self.assertRaisesRegex(RuntimeError, "host trusted quality worker SHA"):
                    site_quality_transport.trusted_remote_worker()

    def test_broker_coordinator_queue_cannot_assess(self) -> None:
        with mock.patch.object(site_quality_transport, "assess",
                               side_effect=AssertionError("assess must not run")):
            with self.assertRaisesRegex(ValueError, "coordinator queue may only submit coordinator runs"):
                site_job_broker.dispatch(self.request("R01"), site_job_broker.QUEUE)

    def test_broker_cache_queue_routes_only_exact_r03(self) -> None:
        sentinel = {"status": "mock_cpu_only"}
        with mock.patch.object(site_quality_transport, "assess", return_value=sentinel) as called:
            result = site_job_broker.dispatch(
                self.request("R03"),
                site_job_broker.EXPERIMENT_QUEUES["sana2ba100-cache_ca-0002"],
            )
        self.assertEqual(result, sentinel)
        called.assert_called_once_with(self.run_path("R03"))

    def test_worker_missing_dependency_fails_before_snapshot(self) -> None:
        missing = Path(self.temp.name) / "missing-wheel.whl"
        with mock.patch.object(site_quality_worker, "CACHE", site_quality_transport.CACHE), \
             mock.patch.object(site_quality_worker, "FIXED_FILES", {missing: "0" * 64}):
            with self.assertRaisesRegex(RuntimeError, "trusted source/dependency SHA"):
                site_quality_worker.preflight(self.run_path("R01"), site_quality_worker.RUNS["R01"])
        self.assertFalse(missing.exists())

    def test_worker_wrong_source_sha_fails_before_snapshot(self) -> None:
        source = Path(self.temp.name) / "source.py"
        source.write_text("trusted source\n")
        with mock.patch.object(site_quality_worker, "CACHE", site_quality_transport.CACHE), \
             mock.patch.object(site_quality_worker, "FIXED_FILES", {source: "0" * 64}):
            with self.assertRaisesRegex(RuntimeError, "trusted source/dependency SHA"):
                site_quality_worker.preflight(self.run_path("R01"), site_quality_worker.RUNS["R01"])

    def test_network_guard_blocks_connect_without_egress(self) -> None:
        guard = Path(__file__).resolve().parents[1] / "orchestration/site_quality_guard/sitecustomize.py"
        spec = importlib.util.spec_from_file_location("phase3_test_sitecustomize", guard)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        original_connect, original_connect_ex = socket.socket.connect, socket.socket.connect_ex
        try:
            with mock.patch.dict(os.environ, {"SOL_PHASE3_NETWORK_AUDIT": ""}):
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    with self.assertRaisesRegex(OSError, "NETWORK_DISABLED_PHASE3"):
                        sock.connect(("127.0.0.1", 9))
        finally:
            socket.socket.connect, socket.socket.connect_ex = original_connect, original_connect_ex


if __name__ == "__main__":
    unittest.main()
