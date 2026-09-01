from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rolloutbench.cli import main
from rolloutbench.h100_preflight import (
    CommandResult,
    build_preflight_spec,
    run_h100_preflight,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE = REPO_ROOT / "benchmarks" / "sana_video_2b_h100_v0" / "h100_profile.json"
DINO_MAIN_SHA = "7c446df5b9f45747937fb0d72314eb9f7b66930a"


def _remote_success(spec: dict) -> dict:
    paths = {
        item["id"]: {
            "path": item["path"],
            "expected_type": item["type"],
            "exists": True,
            "type_matches": True,
        }
        for item in spec["runtime_paths"]
    }
    return {
        "schema_version": 1,
        "observed_at_utc": "2026-09-01T12:00:00+00:00",
        "hostname": "h100-node",
        "persistent_storage": {
            "path": spec["persistent_path"],
            "exists": True,
            "mount": {
                "ok": True,
                "target": spec["persistent_path"],
                "source": "Ustor_file_posix",
                "fstype": "fuse",
            },
            "disk": {
                "ok": True,
                "total_kib": 10_000_000,
                "available_kib": 5_000_000,
                "capacity": "50%",
            },
        },
        "gpus": [
            {
                "index": item["index"],
                "uuid": item["uuid"],
                "name": "NVIDIA H100 80GB HBM3",
                "memory_total_mib": 81559,
            }
            for item in spec["target_gpus"]
        ],
        "compute_apps": [],
        "nvidia_smi_ok": True,
        "runtime": {
            "ok": True,
            "python": spec["expected_environment"]["python"],
            "torch": spec["expected_environment"]["torch"],
            "cuda": spec["expected_environment"]["cuda"],
            "triton": spec["expected_environment"]["triton"],
        },
        "runtime_paths": paths,
        "model": {
            "path": spec["model_path"],
            "model_index_path": f'{spec["model_path"]}/model_index.json',
            "model_index_exists": True,
            "model_index_valid": True,
            "class_name": spec["model_class_name"],
        },
        "remote_benchmark_root": {
            "path": spec["remote_benchmark_root"],
            "exists": True,
            "is_dir": True,
        },
        "vbench": {
            "source_path": spec["vbench_source_path"],
            "source_exists": True,
            "source_is_dir": True,
            "git_ok": True,
            "git_ref": spec["vbench_git_ref"],
            "cache_path": spec["vbench_cache_path"],
            "weights": [
                {
                    "id": item["id"],
                    "path": item["path"],
                    "exists": True,
                    "is_file": True,
                    "readable": True,
                    "size_bytes": 1024,
                }
                for item in spec["vbench_weights"]
            ],
            "dino_source": {
                "path": spec["dino_source"]["path"],
                "exists": True,
                "is_dir": True,
                "git_ok": True,
                "head": spec["dino_source"]["git_ref"],
                "detached": True,
                "clean": True,
                "hubconf": {
                    "path": f'{spec["dino_source"]["path"]}/hubconf.py',
                    "exists": True,
                    "is_file": True,
                    "readable": True,
                    "size_bytes": 1024,
                },
            },
        },
    }


class RecordingRunner:
    def __init__(self, payload: dict | None = None, *, stdout: str | None = None):
        self.payload = payload
        self.stdout = stdout
        self.calls: list[tuple[list[str], str]] = []

    def __call__(self, argv: list[str], stdin: str) -> CommandResult:
        self.calls.append((argv, stdin))
        output = self.stdout if self.stdout is not None else json.dumps(self.payload)
        return CommandResult(returncode=0, stdout=output, stderr="")


class H100PreflightTests(unittest.TestCase):
    def test_spec_is_bound_to_model_and_frozen_suite(self) -> None:
        spec = build_preflight_spec(PROFILE, repo_root=REPO_ROOT)
        self.assertEqual("BAAI", spec["ssh_host"])
        self.assertEqual(
            {
                "python": "3.12.14",
                "torch": "2.11.0+cu128",
                "cuda": "12.8",
                "triton": "3.6.0",
            },
            spec["expected_environment"],
        )
        self.assertEqual(
            "db5f398b13ca086d09a50ce156c20527773841b1", spec["model_revision"]
        )
        self.assertEqual([6, 7], [item["index"] for item in spec["target_gpus"]])
        self.assertEqual(8, len(spec["vbench_weights"]))
        self.assertEqual(
            {
                "repository_url": "https://github.com/facebookresearch/dino.git",
                "path": (
                    "/home/jiangzhikun/yongyan_liu/Experiments/"
                    "SolRolloutBench/20260901-v0/sources/dino"
                ),
                "git_ref": DINO_MAIN_SHA,
                "required_file": "hubconf.py",
            },
            spec["dino_source"],
        )

    def test_dino_profile_rejects_unpinned_unofficial_or_external_source(self) -> None:
        profile = json.loads(PROFILE.read_text(encoding="utf-8"))
        mutations = {
            "unpinned": {"git_ref": "7c446df"},
            "unofficial": {"repository_url": "https://example.invalid/dino.git"},
            "external": {"path": "/tmp/dino"},
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, mutation in mutations.items():
                with self.subTest(name=name):
                    changed = copy.deepcopy(profile)
                    changed["dino_source"].update(mutation)
                    path = Path(directory) / f"{name}.json"
                    path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "DINO|dino_source"):
                        build_preflight_spec(path, repo_root=REPO_ROOT)

    def test_single_read_only_ssh_query_can_mark_all_readiness_dimensions(self) -> None:
        spec = build_preflight_spec(PROFILE, repo_root=REPO_ROOT)
        runner = RecordingRunner(_remote_success(spec))
        receipt = run_h100_preflight(PROFILE, repo_root=REPO_ROOT, runner=runner)
        self.assertEqual(1, len(runner.calls))
        argv, remote_script = runner.calls[0]
        self.assertEqual("ssh", argv[0])
        self.assertIn("BAAI", argv)
        self.assertEqual("PASS", receipt["query_status"])
        self.assertTrue(receipt["runtime_ready"])
        self.assertTrue(receipt["two_gpu_idle_point_in_time"])
        self.assertTrue(receipt["quality_ready"])
        self.assertTrue(receipt["pilot_ready"])
        self.assertFalse(receipt["gpu_idle_scope"]["ownership_verified"])
        self.assertIn("symbolic-ref", remote_script)
        self.assertIn("--no-optional-locks", remote_script)
        self.assertIn("--porcelain", remote_script)
        self.assertIn('dino_spec["required_file"]', remote_script)
        self.assertNotIn("os.environ", remote_script)
        self.assertNotIn("printenv", remote_script)
        self.assertNotIn("mkdir", remote_script)
        self.assertNotIn("wget", remote_script)
        self.assertNotIn('"clone"', remote_script)

    def test_busy_gpu_is_point_in_time_not_an_ownership_claim(self) -> None:
        spec = build_preflight_spec(PROFILE, repo_root=REPO_ROOT)
        payload = _remote_success(spec)
        payload["compute_apps"] = [
            {
                "gpu_uuid": spec["target_gpus"][0]["uuid"],
                "pid": 123,
                "process_name": "python",
                "used_memory_mib": 2000,
            }
        ]
        receipt = run_h100_preflight(
            PROFILE, repo_root=REPO_ROOT, runner=RecordingRunner(payload)
        )
        self.assertTrue(receipt["runtime_ready"])
        self.assertFalse(receipt["two_gpu_idle_point_in_time"])
        self.assertFalse(receipt["pilot_ready"])
        self.assertFalse(receipt["gpu_idle_scope"]["ownership_verified"])

    def test_missing_weight_and_version_mismatch_fail_closed(self) -> None:
        spec = build_preflight_spec(PROFILE, repo_root=REPO_ROOT)
        payload = _remote_success(spec)
        payload["vbench"]["weights"][0]["exists"] = False
        payload["vbench"]["weights"][0]["is_file"] = False
        payload["runtime"]["torch"] = "unexpected"
        receipt = run_h100_preflight(
            PROFILE, repo_root=REPO_ROOT, runner=RecordingRunner(payload)
        )
        self.assertFalse(receipt["runtime_ready"])
        self.assertFalse(receipt["quality_ready"])
        self.assertFalse(receipt["pilot_ready"])
        self.assertIn("torch", " ".join(receipt["checks"]["runtime"]["errors"]))

    def test_missing_attached_or_dirty_dino_source_fails_quality_closed(self) -> None:
        spec = build_preflight_spec(PROFILE, repo_root=REPO_ROOT)
        for mutation in ("missing", "attached", "dirty"):
            with self.subTest(mutation=mutation):
                payload = _remote_success(spec)
                dino = payload["vbench"]["dino_source"]
                if mutation == "missing":
                    dino["exists"] = False
                    dino["is_dir"] = False
                    dino["git_ok"] = False
                    dino["hubconf"]["exists"] = False
                    dino["hubconf"]["is_file"] = False
                elif mutation == "attached":
                    dino["detached"] = False
                else:
                    dino["clean"] = False
                receipt = run_h100_preflight(
                    PROFILE, repo_root=REPO_ROOT, runner=RecordingRunner(payload)
                )
                self.assertTrue(receipt["runtime_ready"])
                self.assertTrue(receipt["two_gpu_idle_point_in_time"])
                self.assertFalse(receipt["quality_ready"])
                self.assertFalse(receipt["pilot_ready"])
                self.assertIn("DINO", " ".join(receipt["checks"]["quality"]["errors"]))

    def test_wrong_dino_head_or_missing_hubconf_fails_quality_closed(self) -> None:
        spec = build_preflight_spec(PROFILE, repo_root=REPO_ROOT)
        for mutation in ("wrong_head", "missing_hubconf"):
            with self.subTest(mutation=mutation):
                payload = _remote_success(spec)
                dino = payload["vbench"]["dino_source"]
                if mutation == "wrong_head":
                    dino["head"] = "0" * 40
                else:
                    dino["hubconf"]["exists"] = False
                    dino["hubconf"]["is_file"] = False
                receipt = run_h100_preflight(
                    PROFILE, repo_root=REPO_ROOT, runner=RecordingRunner(payload)
                )
                self.assertFalse(receipt["quality_ready"])
                self.assertFalse(receipt["pilot_ready"])
                self.assertIn("DINO", " ".join(receipt["checks"]["quality"]["errors"]))

    def test_malformed_remote_output_is_a_false_readiness_receipt(self) -> None:
        runner = RecordingRunner(stdout="not-json")
        receipt = run_h100_preflight(PROFILE, repo_root=REPO_ROOT, runner=runner)
        self.assertEqual("ERROR", receipt["query_status"])
        self.assertFalse(receipt["runtime_ready"])
        self.assertFalse(receipt["two_gpu_idle_point_in_time"])
        self.assertFalse(receipt["quality_ready"])
        self.assertFalse(receipt["pilot_ready"])
        self.assertEqual(1, len(runner.calls))

    def test_cli_writes_the_receipt_without_a_second_ssh_call(self) -> None:
        spec = build_preflight_spec(PROFILE, repo_root=REPO_ROOT)
        runner = RecordingRunner(_remote_success(spec))
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "rolloutbench.h100_preflight._default_command_runner", runner
        ):
            output = Path(directory) / "receipt.json"
            self.assertEqual(
                0,
                main(
                    [
                        "h100-preflight",
                        "--profile",
                        str(PROFILE),
                        "--repo-root",
                        str(REPO_ROOT),
                        "--output",
                        str(output),
                    ]
                ),
            )
            self.assertEqual(1, len(runner.calls))
            self.assertEqual("PASS", json.loads(output.read_text())["query_status"])
            output.write_text("conflict\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_h100_preflight(
                    PROFILE,
                    repo_root=REPO_ROOT,
                    runner=RecordingRunner(_remote_success(spec)),
                    output_path=output,
                )


if __name__ == "__main__":
    unittest.main()
