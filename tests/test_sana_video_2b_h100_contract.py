from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "models/sana_video_2b_h100/baseline"


def _load_module(name: str, path: Path):
    sys.path.insert(0, str(BASELINE))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(BASELINE))


def _fake_nvidia_smi(tmp_path: Path) -> Path:
    script = tmp_path / "nvidia-smi"
    script.write_text(
        """#!/usr/bin/env python3
import os
import sys
if any(arg.startswith('--query-gpu=') for arg in sys.argv):
    print('GPU-test-uuid, 7, NVIDIA H100 80GB HBM3, 81559, 0, 0')
elif any(arg.startswith('--query-compute-apps=') for arg in sys.argv):
    if os.environ.get('FAKE_GPU_APP') == '1':
        print('GPU-test-uuid, 1234, foreign.py, 4096')
else:
    raise SystemExit(2)
"""
    )
    script.chmod(0o755)
    return script


class SanaVideo2BH100ContractTest(unittest.TestCase):
    def test_gpu_guard_is_uuid_scoped_and_fails_closed(self) -> None:
        guard = _load_module("sana_gpu_guard", BASELINE / "gpu_guard.py")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake = _fake_nvidia_smi(tmp_path)
            lease_file = tmp_path / "GPU_LEASE.json"
            lock_path = tmp_path / "gpu.lock"
            lease_file.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "active",
                        "gpu_uuid": "GPU-test-uuid",
                        "lock_path": str(lock_path),
                        "host": "test-host",
                        "owner": "test-owner",
                        "leased_at_utc": "2026-08-31T00:00:00+00:00",
                    }
                )
            )
            with mock.patch.dict(os.environ, {"SANA_NVIDIA_SMI": str(fake)}, clear=False):
                with guard.locked_idle_lease(lease_file) as (lease, gpu):
                    self.assertEqual(lease.gpu_uuid, "GPU-test-uuid")
                    self.assertEqual(gpu["index"], 7)
                    self.assertEqual(gpu["memory_used_mib"], 0)

                with mock.patch.dict(os.environ, {"FAKE_GPU_APP": "1"}, clear=False):
                    with self.assertRaisesRegex(RuntimeError, "refusing to disturb"):
                        with guard.locked_idle_lease(lease_file):
                            pass

    def test_dense_control_rejects_every_exposed_optimization(self) -> None:
        infer = _load_module("sana_gpu_infer", BASELINE / "gpu_infer.py")
        dense_env = {
            "SANA_ENABLE_COMPILE": "0",
            "SANA_ENABLE_MAX_AUTOTUNE": "0",
            "SANA_ENABLE_LINATTN_BF16": "0",
            "SANA_ENABLE_QKV_MERGE": "0",
            "SANA_DISABLE_WARMUP": "0",
            "SANA_EASYCACHE_THRESH": "0",
        }
        with mock.patch.dict(os.environ, dense_env, clear=False):
            knobs = infer._optimization_knobs()
            infer._assert_dense_control("sana_video_2b_h100_dense_baseline", knobs)
            with mock.patch.dict(os.environ, {"SANA_ENABLE_QKV_MERGE": "1"}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "optimization knobs enabled"):
                    infer._assert_dense_control(
                        "sana_video_2b_h100_dense_baseline",
                        infer._optimization_knobs(),
                    )

    def test_model_contract_and_workload_are_pinned(self) -> None:
        profile = tomllib.loads((ROOT / "models/sana_video_2b_h100.toml").read_text())
        contract = tomllib.loads(
            (ROOT / "models/sana_video_2b_h100/model.toml").read_text()
        )
        config = tomllib.loads(
            (ROOT / "config/sana_video_2b_h100/baseline.toml").read_text()
        )
        official = profile["official_config"]
        self.assertEqual(config["id"], "sana_video_2b_h100_dense_baseline")
        self.assertEqual(contract["id"], "sana_video_2b_h100")
        self.assertEqual(profile["orchestration"]["default_techniques"], ["kernel", "cache"])
        self.assertEqual(profile["orchestration"]["inference_world_size"], 1)
        self.assertEqual(
            official,
            {
                "model": "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
                "revision": "db5f398b13ca086d09a50ce156c20527773841b1",
                "pipeline_class_name": "SanaVideoPipeline",
                "width": 832,
                "height": 480,
                "frames": 81,
                "fps": 16,
                "steps": 50,
                "guidance_scale": 6.0,
                "seed": 42,
                "motion_score": 30,
                "flow_shift": 8.0,
                "vae_precision": "fp32",
                "transformer_precision": "bf16",
                "text_encoder_precision": "bf16",
                "num_gpus": 1,
            },
        )
        prompt = (
            ROOT
            / "models/sana_video_2b_h100/prompts/model_card_long_motion30.txt"
        ).read_text().strip()
        self.assertTrue(prompt.endswith("motion score: 30."))
        for name in (
            "SANA_ENABLE_COMPILE",
            "SANA_ENABLE_MAX_AUTOTUNE",
            "SANA_ENABLE_LINATTN_BF16",
            "SANA_ENABLE_QKV_MERGE",
            "SANA_EASYCACHE_THRESH",
            "SANA_DISABLE_WARMUP",
        ):
            self.assertEqual(profile["env"][name], "0")

    def test_contract_materializer_and_launch_config_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            create = subprocess.run(
                [
                    sys.executable,
                    "scripts/create_model_experiment.py",
                    "--model",
                    "sana_video_2b_h100",
                    "--workflow-uid",
                    "kernel_aw",
                    "--experiment-uid",
                    "sana2b-kernel_aw-9999",
                    "--experiments-root",
                    str(tmp_path / "not-created-by-dry-run"),
                    "--dry-run",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            materialization = json.loads(create.stdout)
            self.assertGreater(materialization["external_copy_count"], 0)
            self.assertTrue(
                any(
                    path.endswith(
                        "external/sol_runtime/scripts/sana/sana_video_sglang_run.py"
                    )
                    for path in materialization["external_sources"][0]["copied_paths"]
                )
            )

            launch = subprocess.run(
                [
                    sys.executable,
                    "scripts/launch_config.py",
                    "config/sana_video_2b_h100/baseline.toml",
                    "--mode",
                    "dry-run",
                    "--run-root",
                    str(tmp_path / "dry-runs"),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            self.assertIn("sana_video_2b_h100_dense_baseline", launch.stdout)

    def test_pisa_disposition_matches_pinned_sources(self) -> None:
        disposition = (
            ROOT / "models/sana_video_2b_h100/PISA_NOT_APPLICABLE.md"
        ).read_text()
        self.assertIn("Status: `NOT_APPLICABLE`", disposition)
        self.assertIn("head dimension 112", disposition)
        self.assertIn("SanaVideoLinearAttention", disposition)


if __name__ == "__main__":
    unittest.main(verbosity=2)
