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
        command = self.runner._command(
            Path("/python"), Path("/runtime"), Path("/model"), Path("/prompt"), "out", knobs, workload
        )
        self.assertEqual("12345", command[command.index("--seed") + 1])
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


if __name__ == "__main__":
    unittest.main()
