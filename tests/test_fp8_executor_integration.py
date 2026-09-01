from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _toml(relative: str):
    with (ROOT / relative).open("rb") as handle:
        return tomllib.load(handle)


def _verifier_module():
    path = ROOT / "orchestration/bin/verify_delivery.py"
    spec = importlib.util.spec_from_file_location("verify_delivery", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load verifier: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FP8ExecutorIntegrationTest(unittest.TestCase):
    def test_fp8_executor_is_registered_for_sana_after_kernel_and_cache(self):
        registry = _toml("orchestration/techniques.toml")
        profile = _toml("models/sana_video_2b_h100.toml")

        self.assertEqual(
            registry["techniques"]["fp8"],
            {
                "workflow_uid": "quant_qe",
                "scope": "workflow/quant_qe/nodes/codex_executor/fp8_scope.md",
                "correctness": "quality_gated",
            },
        )
        self.assertEqual(
            profile["orchestration"]["default_techniques"],
            ["kernel", "cache", "fp8"],
        )
        self.assertEqual(profile["env"]["SANA_ENABLE_FP8"], "0")

    def test_fp8_manifests_have_isolated_and_integrated_variants(self):
        integrated = _toml("config/sana_video_2b_h100/fp8/fp8_ffn_all_blocks.toml")
        cache013 = _toml(
            "config/sana_video_2b_h100/fp8/fp8_ffn_all_blocks_cache013.toml"
        )
        isolated = _toml(
            "config/sana_video_2b_h100/fp8/fp8_ffn_only_all_blocks.toml"
        )

        for manifest in (integrated, cache013, isolated):
            self.assertEqual(manifest["env"]["SANA_ENABLE_FP8"], "1")
            self.assertEqual(manifest["env"]["SANA_FP8_SCOPE"], "ffn_1x1")
            self.assertEqual(manifest["env"]["SANA_FP8_BLOCK_START"], "0")
            self.assertEqual(manifest["env"]["SANA_FP8_BLOCK_END"], "-1")
            self.assertEqual(manifest["env"]["SANA_FP8_STRICT"], "1")
        self.assertEqual(integrated["kind"], "integrated")
        self.assertEqual(integrated["env"]["SANA_CACHE_FAMILY"], "easycache")
        self.assertEqual(cache013["env"]["SANA_CACHE_THRESHOLD"], "0.13")
        self.assertEqual(isolated["kind"], "patch")
        self.assertEqual(isolated["env"]["SANA_CACHE_FAMILY"], "off")

    def test_model_contract_installs_fp8_overlays_after_external_runtime_copy(self):
        contract = _toml("models/sana_video_2b_h100/model.toml")
        overlays = contract["baseline"]["overlay_copy"]
        sources = {item["source"] for item in overlays}
        destinations = {item["dest"] for item in overlays}

        expected = {
            "external/sol_runtime/python/sglang/multimodal_gen/runtime/models/dits/sana_fp8.py",
            "external/sol_runtime/python/sglang/multimodal_gen/runtime/models/dits/sana_video.py",
            "external/sol_runtime/scripts/sana/sana_video_sglang_run.py",
            "external/sol_runtime/python/sglang/multimodal_gen/runtime/efficiency/transforms/fp8_ffn.py",
            "external/sol_runtime/python/sglang/multimodal_gen/runtime/efficiency/transforms/__init__.py",
        }
        self.assertEqual(sources, expected)
        self.assertEqual(destinations, expected)
        self.assertTrue(all((ROOT / source).is_file() for source in sources))

    def test_fp8_entrypoints_compile_and_transform_owns_precision_seam(self):
        paths = [
            ROOT
            / "external/sol_runtime/python/sglang/multimodal_gen/runtime/models/dits/sana_fp8.py",
            ROOT
            / "external/sol_runtime/python/sglang/multimodal_gen/runtime/efficiency/transforms/fp8_ffn.py",
            ROOT / "external/sol_runtime/scripts/sana/sana_video_sglang_run.py",
            ROOT / "models/sana_video_2b_h100/baseline/gpu_infer.py",
            ROOT / "scripts/sana/fp8_component_smoke.py",
        ]
        trees = [ast.parse(path.read_text(), filename=str(path)) for path in paths]
        transform_tree = trees[1]
        class_node = next(
            node
            for node in transform_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "FP8FFN"
        )
        assignments = {
            target.id: node.value
            for node in class_node.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertIn("writes", assignments)

        harness_source = paths[3].read_text()
        self.assertIn('"warmup_request_s": warmup_request_s', harness_source)
        self.assertIn('"warm_steady_state_s": warm_steady_state_s', harness_source)
        self.assertIn('"decode_s": decode_s', harness_source)
        self.assertIn("first_gen.generate_call_including_one_step_runtime_warmup", harness_source)

    def test_executor_contract_requires_machine_readable_delivery(self):
        interface = _toml("workflow/quant_qe/nodes/codex_executor/interface.toml")
        self.assertEqual(
            interface["outputs"],
            ["TRAJECTORY.jsonl", "DELIVERY.json", "FP8-SEARCH-STATE.json"],
        )
        self.assertIn("structured_negative", interface["outcomes"])

    def test_fp8_verifier_accepts_native_smoke_and_video_receipts(self):
        verifier = _verifier_module()
        with tempfile.TemporaryDirectory() as tmp:
            worktree = Path(tmp).resolve()
            run_dir = worktree / "runs/test"
            outputs = run_dir / "outputs"
            outputs.mkdir(parents=True)
            benchmark = {
                "status": "VALIDATED",
                "fp8": {
                    "enabled": True,
                    "install": {
                        "status": "installed",
                        "enabled": True,
                        "converted_modules": ["transformer_blocks.0.ff.conv_point"],
                        "skipped_modules": [],
                        "backend": {
                            "compute_capability": [9, 0],
                            "fp8_dtype": "torch.float8_e4m3fn",
                            "cutlass_fp8_supported": True,
                        },
                    },
                    "active_modules": [
                        {
                            "module": "transformer_blocks.0.ff.conv_point",
                            "weight_dtype": "torch.float8_e4m3fn",
                        }
                    ],
                    "active_module_count": 1,
                },
                "residual_compute_apps": [],
                "validity": {
                    "status": "VALIDATED",
                    "checks": {"width": True, "height": True, "frames": True},
                },
            }
            (outputs / "benchmark.json").write_text(json.dumps(benchmark))
            smoke = {
                "status": "passed",
                "checks": {
                    "finite": True,
                    "cosine": True,
                    "relative_rmse": True,
                    "real_fp8_calls": True,
                    "fp8_weight_dtype": True,
                },
                "tolerances": {"min_cosine": 0.995, "max_relative_rmse": 0.1},
                "cases": [
                    {
                        "cosine_similarity": 0.999,
                        "relative_rmse": 0.03,
                        "fp8_calls": 1,
                        "weight_dtype": "torch.float8_e4m3fn",
                    }
                ],
            }
            (run_dir / "fp8_component_smoke.json").write_text(json.dumps(smoke))
            point = {
                "fp8_evidence": {
                    "component_smoke": "runs/test/fp8_component_smoke.json"
                }
            }

            issues, summary = verifier.check_fp8_evidence(
                point, run_dir, worktree
            )
            self.assertEqual(issues, [])
            self.assertEqual(summary["active_module_count"], 1)
            self.assertEqual(summary["video_validity"], "VALIDATED")

            benchmark["fp8"]["active_module_count"] = 0
            (outputs / "benchmark.json").write_text(json.dumps(benchmark))
            issues, _ = verifier.check_fp8_evidence(point, run_dir, worktree)
            self.assertIn("fp8_active_modules_do_not_match_install", issues)

    def test_sana_legacy_and_corrected_timing_scopes_are_equivalent(self):
        verifier = _verifier_module()
        old_scope = (
            "warm_single_prompt_gen.generate_including_text_encoder_denoise_"
            "vae_decode_and_video_write_excluding_model_load_and_one_step_warmup"
        )
        new_scope = (
            "first_gen.generate_call_including_one_step_runtime_warmup_text_"
            "encoder_denoise_vae_decode_and_video_write_excluding_model_load"
        )
        self.assertEqual(
            verifier._timing_scope_key(old_scope),
            verifier._timing_scope_key(new_scope),
        )

    def test_derived_timing_receipts_match_immutable_logs(self):
        report_root = ROOT / "reports/fp8_executor"
        receipt = json.loads((report_root / "TIMING-RECEIPTS.json").read_text())
        ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
        metric_keys = (
            "outer_total_s",
            "warmup_request_s",
            "warm_steady_state_s",
            "denoise_s",
            "decode_s",
        )
        for run in receipt["runs"]:
            log_path = report_root / run["log"]
            self.assertEqual(
                hashlib.sha256(log_path.read_bytes()).hexdigest(),
                run["log_sha256"],
            )
            transcript = ansi.sub("", log_path.read_text())
            for key in metric_keys:
                match = re.search(receipt["parser"]["patterns"][key], transcript)
                self.assertIsNotNone(match, f"{run['id']} missing {key}")
                self.assertAlmostEqual(float(match.group(1)), run[key], places=4)
            lines = transcript.splitlines()
            for line_number in run["source_lines"].values():
                self.assertGreaterEqual(len(lines), line_number)
                self.assertTrue(lines[line_number - 1].strip())

    def test_verifier_cli_accepts_positive_and_measured_negative_fp8_delivery(self):
        scope = (
            "first_gen.generate_call_including_one_step_runtime_warmup_text_"
            "encoder_denoise_vae_decode_and_video_write_excluding_model_load"
        )
        baseline = {
            "total_s": 10.0,
            "timing_scope": scope,
            "run_dir": "runs/frozen-baseline",
        }
        verifier = ROOT / "orchestration/bin/verify_delivery.py"

        with tempfile.TemporaryDirectory() as tmp:
            worktree = Path(tmp).resolve()
            baseline_path = worktree / "BASELINE.json"
            baseline_path.write_text(json.dumps(baseline))
            run_dir = worktree / "runs/fp8-positive"
            outputs = run_dir / "outputs"
            outputs.mkdir(parents=True)
            (outputs / "out.mp4").write_bytes(b"test-provenance-only")
            (run_dir / "job-started.json").write_text("{}")
            (run_dir / "metadata.json").write_text(
                json.dumps({"config_id": "fp8-positive"})
            )
            benchmark = {
                "status": "VALIDATED",
                "total_s": 8.0,
                "timing_scope": scope,
                "fp8": {
                    "enabled": True,
                    "install": {
                        "status": "installed",
                        "enabled": True,
                        "converted_modules": ["transformer_blocks.0.ff.conv_point"],
                        "skipped_modules": [],
                        "backend": {
                            "compute_capability": [9, 0],
                            "fp8_dtype": "torch.float8_e4m3fn",
                            "cutlass_fp8_supported": True,
                        },
                    },
                    "active_modules": [
                        {
                            "module": "transformer_blocks.0.ff.conv_point",
                            "weight_dtype": "torch.float8_e4m3fn",
                        }
                    ],
                    "active_module_count": 1,
                },
                "residual_compute_apps": [],
                "validity": {
                    "status": "VALIDATED",
                    "checks": {"width": True, "height": True, "frames": True},
                },
            }
            (outputs / "benchmark.json").write_text(json.dumps(benchmark))
            smoke = {
                "status": "passed",
                "checks": {
                    "finite": True,
                    "cosine": True,
                    "relative_rmse": True,
                    "real_fp8_calls": True,
                    "fp8_weight_dtype": True,
                },
                "tolerances": {"min_cosine": 0.995, "max_relative_rmse": 0.1},
                "cases": [
                    {
                        "cosine_similarity": 0.999,
                        "relative_rmse": 0.03,
                        "fp8_calls": 1,
                        "weight_dtype": "torch.float8_e4m3fn",
                    }
                ],
            }
            (run_dir / "fp8_component_smoke.json").write_text(json.dumps(smoke))
            positive = {
                "schema_version": 2,
                "status": "complete",
                "component": "fp8",
                "model_id": "sana_video_2b_h100",
                "baseline": baseline,
                "frontier_points": [
                    {
                        "config_id": "fp8-positive",
                        "run_dir": "runs/fp8-positive",
                        "performance": {
                            "frontier_axis": "latency",
                            "baseline_total_s": 10.0,
                            "config_total_s": 8.0,
                            "speedup": 1.25,
                        },
                        "fp8_evidence": {
                            "component_smoke": (
                                "runs/fp8-positive/fp8_component_smoke.json"
                            )
                        },
                    }
                ],
            }
            delivery_path = worktree / "DELIVERY.json"
            delivery_path.write_text(json.dumps(positive))
            command = [
                sys.executable,
                str(verifier),
                "--worktree",
                str(worktree),
                "--model",
                "sana_video_2b_h100",
                "--tech",
                "fp8",
                "--baseline",
                str(baseline_path),
            ]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(json.loads(result.stdout)["objective_ok"])

            trajectory = worktree / "TRAJECTORY.jsonl"
            trajectory.write_text(
                json.dumps(
                    {
                        "round": 1,
                        "candidate": "fp8-all",
                        "decision": {"outcome": "reject"},
                    }
                )
                + "\n"
            )
            (worktree / "FP8-SEARCH-STATE.json").write_text(
                json.dumps({"component": "fp8", "status": "structured_negative"})
            )
            (worktree / "negative-smoke.json").write_text(json.dumps(smoke))
            negative = {
                "schema_version": 2,
                "status": "structured_negative",
                "component": "fp8",
                "model_id": "sana_video_2b_h100",
                "baseline": baseline,
                "frontier_points": [],
                "negative_evidence": {
                    "reason": "measured regression",
                    "attempt_count": 1,
                    "trajectory": "TRAJECTORY.jsonl",
                    "search_state": "FP8-SEARCH-STATE.json",
                    "evidence_files": ["negative-smoke.json"],
                },
            }
            delivery_path.write_text(json.dumps(negative))
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = json.loads(result.stdout)
            self.assertTrue(output["objective_ok"])
            self.assertTrue(output["structured_negative"])
            self.assertEqual(output["points"], [])


if __name__ == "__main__":
    unittest.main()
