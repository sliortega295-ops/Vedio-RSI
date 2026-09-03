from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "models" / "sana_video_2b_h100" / "baseline"


def _runner_module():
    sys.path.insert(0, str(BASELINE_DIR))
    try:
        return importlib.import_module("gpu_infer")
    finally:
        sys.path.pop(0)


class SanaWorkloadSeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = _runner_module()

    def test_seed_defaults_to_42_without_mutating_global_workload(self) -> None:
        original = dict(self.runner.WORKLOAD)
        with mock.patch.dict(os.environ, {}, clear=True):
            workload = self.runner._workload_from_env()
        self.assertEqual(42, workload["seed"])
        self.assertEqual(original, self.runner.WORKLOAD)
        self.assertIsNot(workload, self.runner.WORKLOAD)

    def test_only_the_two_formal_decimal_seed_values_are_accepted(self) -> None:
        for seed in ("42", "12345"):
            with self.subTest(seed=seed), mock.patch.dict(
                os.environ, {"SANA_WORKLOAD_SEED": seed}, clear=True
            ):
                self.assertEqual(int(seed), self.runner._workload_from_env()["seed"])
        for seed in ("", "41", "12346", "42.0", "+42", " 42", "0042", "abc"):
            with self.subTest(seed=seed), mock.patch.dict(
                os.environ, {"SANA_WORKLOAD_SEED": seed}, clear=True
            ):
                with self.assertRaisesRegex(RuntimeError, "SANA_WORKLOAD_SEED"):
                    self.runner._workload_from_env()

    def test_seed_routes_to_command_receipt_fingerprint_and_explicit_video_validation(self) -> None:
        workload = {**self.runner.WORKLOAD, "seed": 12345}
        knobs = {
            "torch_compile": False,
            "max_autotune": False,
            "linear_attention_bf16": False,
            "qkv_merge": False,
            "easycache_threshold": 0.0,
            "cache_family": "off",
            "warmup_disabled": False,
        }
        assignment = {
            "slot": 1,
            "reserved_port_block": {"start": 26000, "end": 29999},
            "port": 29500,
            "master_port": 28000,
            "scheduler_port": 26000,
            "nccl_port": 27000,
            "strict_ports": True,
        }
        command = self.runner._command(
            Path("/python"),
            Path("/runtime"),
            Path("/model"),
            Path("/prompt"),
            "out",
            knobs,
            workload,
            assignment,
        )
        self.assertEqual("12345", command[command.index("--seed") + 1])
        self.assertTrue(command[1].endswith("port_isolated_exec.py"))
        self.assertEqual(
            "/runtime/scripts/sana/sana_video_sglang_run.py",
            command[command.index("--target") + 1],
        )
        self.assertEqual("29500", command[command.index("--port") + 1])
        self.assertEqual("28000", command[command.index("--master-port") + 1])
        self.assertEqual(
            "26000", command[command.index("--scheduler-port") + 1]
        )
        self.assertEqual("27000", command[command.index("--nccl-port") + 1])
        self.assertIn("--strict-ports", command)
        receipt = self.runner._workload_receipt(workload, "prompt", "prompt-hash")
        self.assertEqual(12345, receipt["seed"])
        self.assertNotEqual(
            self.runner._fingerprint(receipt),
            self.runner._fingerprint(self.runner._workload_receipt(self.runner.WORKLOAD, "prompt", "prompt-hash")),
        )

        explicit = {**workload, "width": 2, "height": 2, "frames": 3, "fps": 1}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "out.mp4"
            video.write_bytes(b"video")

            def fake_ffmpeg(*_args, **_kwargs) -> None:
                frames = root / "frames"
                for index in range(3):
                    (frames / f"frame_{index:02d}.png").write_bytes(b"frame")

            with mock.patch.object(
                self.runner,
                "_ffprobe",
                return_value={"width": 2, "height": 2, "frames": 3, "fps": 1.0, "duration_s": 3.0, "raw": {}},
            ), mock.patch.object(self.runner.subprocess, "run", side_effect=fake_ffmpeg):
                validity = self.runner._validate_video(video, root / "frames", explicit)
        self.assertEqual("VALIDATED", validity["status"])

    def test_port_assignment_is_exact_and_fails_closed_for_unknown_slots(self) -> None:
        expected = {
            "0": {
                "slot": 0,
                "reserved_port_block": {"start": 16000, "end": 19999},
                "port": 19500,
                "master_port": 18000,
                "scheduler_port": 16000,
                "nccl_port": 17000,
                "strict_ports": True,
            },
            "1": {
                "slot": 1,
                "reserved_port_block": {"start": 26000, "end": 29999},
                "port": 29500,
                "master_port": 28000,
                "scheduler_port": 26000,
                "nccl_port": 27000,
                "strict_ports": True,
            },
        }
        for raw, assignment in expected.items():
            with self.subTest(raw=raw), mock.patch.dict(
                os.environ, {"ROLLOUTBENCH_PORT_SLOT": raw}, clear=True
            ):
                self.assertEqual(assignment, self.runner._port_assignment_from_env())
        for raw in ("", "2", "01", "+1", " 1", "worker-1"):
            with self.subTest(raw=raw), mock.patch.dict(
                os.environ, {"ROLLOUTBENCH_PORT_SLOT": raw}, clear=True
            ):
                with self.assertRaisesRegex(RuntimeError, "ROLLOUTBENCH_PORT_SLOT"):
                    self.runner._port_assignment_from_env()
        first, second = expected["0"], expected["1"]
        self.assertLess(
            first["reserved_port_block"]["end"],
            second["reserved_port_block"]["start"],
        )
        for assignment in expected.values():
            block = assignment["reserved_port_block"]
            self.assertNotEqual(5555, assignment["scheduler_port"])
            for name in ("port", "master_port", "scheduler_port", "nccl_port"):
                self.assertLessEqual(block["start"], assignment[name])
                self.assertLessEqual(assignment[name], block["end"])

    def test_port_receipt_revalidation_checks_marker_and_effective_scheduler(self) -> None:
        assignment = {
            "slot": 1,
            "reserved_port_block": {"start": 26000, "end": 29999},
            "port": 29500,
            "master_port": 28000,
            "scheduler_port": 26000,
            "nccl_port": 27000,
            "strict_ports": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "port_isolated_exec.py"
            target = root / "sana_video_sglang_run.py"
            adapter.write_bytes(b"adapter")
            target.write_bytes(b"target")
            contract = self.runner._port_isolation_contract(
                adapter, target, assignment
            )
            marker = json.dumps(
                contract["expected_adapter_receipt"],
                sort_keys=True,
                separators=(",", ":"),
            )
            effective = json.dumps(
                contract["expected_effective_receipt"],
                sort_keys=True,
                separators=(",", ":"),
            )
            transcript = (
                f"ROLLOUTBENCH_PORT_ISOLATION {marker}\n"
                f"ROLLOUTBENCH_EFFECTIVE_PORTS {effective}\n"
                "[runtime] Scheduler bind at endpoint: tcp://127.0.0.1:26000\n"
            )
            verified = self.runner._verify_port_isolation_transcript(
                transcript, contract
            )
            self.assertEqual("VERIFIED", verified["status"])
            self.assertEqual(26000, verified["effective_scheduler_port"])
            for bad in (
                transcript + f"ROLLOUTBENCH_PORT_ISOLATION {marker}\n",
                transcript.replace(":26000", ":26042"),
                transcript.replace('"nccl_port":27000', '"nccl_port":27001'),
                "not-json\n",
            ):
                with self.subTest(bad=bad[:80]):
                    with self.assertRaisesRegex(RuntimeError, "port-isolation"):
                        self.runner._verify_port_isolation_transcript(bad, contract)

    def test_runtime_receipt_paths_are_candidate_aware_and_fail_closed(self) -> None:
        early = list(self.runner.COMMON_RUNTIME_FILES)
        with mock.patch.dict(
            os.environ,
            {"ROLLOUTBENCH_REQUIRED_RUNTIME_PATHS_JSON": json.dumps(early)},
            clear=True,
        ):
            self.assertEqual(tuple(early), self.runner._runtime_receipt_paths())
        invalid = [*early, "python/not-authorized.py"]
        with mock.patch.dict(
            os.environ,
            {"ROLLOUTBENCH_REQUIRED_RUNTIME_PATHS_JSON": json.dumps(invalid)},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "runtime manifest"):
                self.runner._runtime_receipt_paths()

    def test_k22_cleanup_accepts_exact_python_warning_source_variants(self) -> None:
        common = [
            "[09-02 17:59:23] Generator was garbage collected without being "
            "shut down. Attempting to shut down the local server and client.",
            "/runtime/python3.12/multiprocessing/resource_tracker.py:279: "
            "UserWarning: resource_tracker: There appear to be 1 leaked semaphore "
            "objects to clean up at shutdown",
        ]
        for warning_source_line in (
            "warnings.warn('resource_tracker: There appear to be %d '",
            "warnings.warn('resource_tracker: There appear to be %d ')",
        ):
            with self.subTest(warning_source_line=warning_source_line):
                self.assertTrue(
                    self.runner._post_sentinel_cleanup_only(
                        [*common, warning_source_line]
                    )
                )
        self.assertFalse(
            self.runner._post_sentinel_cleanup_only(
                [
                    *common,
                    "warnings.warn('resource_tracker: There appear to be %d ') extra",
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()
