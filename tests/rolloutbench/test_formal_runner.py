from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rolloutbench.pilot_runner import RunContext, Unit
from rolloutbench.quality_contract import DENSE_REFERENCE_ID, K22_FAILURE_CONTRACT
from rolloutbench.validators import build_quality_plan
from rolloutbench.vbench_runner import build_vbench_pair_plan


def _k22_test_invocation(root: Path) -> tuple[dict, dict]:
    runtime_root = root / "runtime"
    critical = runtime_root / "python" / "runtime.py"
    critical.parent.mkdir(parents=True)
    critical.write_text("# pinned runtime\n")
    relative = "python/runtime.py"
    source = {
        "harness_archival_parent": "d2c6407cc9b9133f3fff49fe4b561f14980d3f8b",
        "runtime_authority_sha": "b0b7eb4d0a7f1f46118a356485f4523cf52e96dd",
        "runtime_compat_sha": "5bc0c43fb7fe548af4119a8831c4e286c982c71f",
        "runtime_root": str(runtime_root),
        "required_runtime_paths": [relative],
        "critical_file_sha256": {
            relative: hashlib.sha256(critical.read_bytes()).hexdigest()
        },
    }
    invocation = {
        "output_path": str(root / "benchmark.json"),
        "episode_id": "K22",
        "runtime_ref": K22_FAILURE_CONTRACT["runtime_ref"],
        "expected_failure_contract": dict(K22_FAILURE_CONTRACT),
        "env": {
            "SANA_HARNESS_ARCHIVE_SHA": source["harness_archival_parent"],
            "SANA_RUNTIME_AUTHORITY_SHA": source["runtime_authority_sha"],
            "SANA_RUNTIME_COMPAT_SHA": source["runtime_compat_sha"],
            "SANA_RUNTIME_ROOT": source["runtime_root"],
            "ROLLOUTBENCH_REQUIRED_RUNTIME_PATHS_JSON": json.dumps([relative]),
        },
    }
    return invocation, source


def _write_k22_test_evidence(
    output: Path,
    source: dict,
    *,
    generation_extra: str = "",
    post_sentinel_extra: str = "",
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    marker = K22_FAILURE_CONTRACT["expected_log_marker"]
    terminal_exception = f"Exception: Error executing request None: {marker}"
    generate_fail_sentinel = "GENERATE_FAIL: no result returned (res=None)"
    run_log = output.parent / "run.log"
    run_log.write_text(
        "ModuleNotFoundError: No module named 'optional_startup_backend'\n"
        "==== STAGE 4: generate ====\n"
        + (f"ValueError: {marker}\n" * 12)
        + terminal_exception
        + "\n"
        + generation_extra
        + generate_fail_sentinel
        + "\n"
        + post_sentinel_extra
    )
    run_config = {
        "schema_version": 1,
        "config_id": K22_FAILURE_CONTRACT["config_id"],
        "source": source,
        "expected_failure_contract": {
            key: K22_FAILURE_CONTRACT[key]
            for key in (
                "episode_id",
                "failure_code",
                "expected_log_marker",
                "config_sha256",
                "runtime_ref",
            )
        },
    }
    (output.parent / "run_config.json").write_text(json.dumps(run_config))
    marker_count = run_log.read_bytes().count(marker.encode())
    post_sentinel_line_count = len(
        [line for line in post_sentinel_extra.splitlines() if line.strip()]
    )
    failure = {
        "episode_id": K22_FAILURE_CONTRACT["episode_id"],
        "failure_code": K22_FAILURE_CONTRACT["failure_code"],
        "stage": K22_FAILURE_CONTRACT["stage"],
        "expected_log_marker": marker,
        "observed_marker_count": marker_count,
        "marker_matched": True,
        "generate_fail_sentinel": generate_fail_sentinel,
        "generate_fail_sentinel_count": 1,
        "post_sentinel_line_count": post_sentinel_line_count,
        "post_sentinel_cleanup_only": True,
        "terminal_exception": terminal_exception,
        "terminal_exception_matched": True,
        "child_returncode": K22_FAILURE_CONTRACT["child_returncode"],
        "config_id": K22_FAILURE_CONTRACT["config_id"],
        "config_sha256": K22_FAILURE_CONTRACT["config_sha256"],
        "runtime_ref": K22_FAILURE_CONTRACT["runtime_ref"],
        "run_log": {
            "path": str(run_log),
            "sha256": hashlib.sha256(run_log.read_bytes()).hexdigest(),
        },
    }
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "FAILED",
                "returncode": 4,
                "generation_s": None,
                "total_s": None,
                "process_wall_s": 2.0,
                "phase_timings": {
                    "marker_timeline": [{"marker": "generation_started"}]
                },
                "residual_compute_apps": [],
                "failure": failure,
            }
        )
    )


class FormalRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = RunContext(
            plan_id="plan", plan_sha256="a" * 64, run_sha256="b" * 64,
            preparation_sha256="c" * 64,
            run={"run_id": "pilot-serial1-repeat-01", "system": "serial1", "workers": [{"worker_id": 0, "gpu_uuid": "GPU-1"}], "episodes": [{"episode_id": "C02", "candidate": {}, "runtime_checkout": {}, "quality_pairs": []}]},
            preparation={"experiment_root": "/experiment", "derived_root": "/derived", "runtime_receipts": {"C02": {}}, "materialization_receipts": {"C02": {}}},
            plan_path=Path("/plan.json"), preparation_path=Path("/preparation.json"),
        )
        self.unit = Unit("C02:primary", "C02", "cache", 0, (), 0, "primary", (), None, ("C02",))

    def test_primary_is_built_internally_and_wrapped_with_exact_unit_binding(self) -> None:
        from rolloutbench.formal_runner import build_formal_invocation

        base = {"argv": ["bash", "run"], "cwd": "/repo", "env": {}, "output_path": "/experiment/runs/plan/pilot-serial1-repeat-01/C02/primary/out.mp4", "episode_id": "C02", "run_id": "pilot-serial1-repeat-01", "quality_pair_id": None, "quality_role": None, "gpu_uuid": "GPU-1", "runtime_ref": "r", "runtime_tree_oid": "t", "expected_failure_contract": None, "harness": {}}
        identity = {key: base[key] for key in ("argv", "env", "cwd", "output_path", "episode_id", "run_id", "quality_pair_id", "quality_role", "gpu_uuid", "runtime_ref", "runtime_tree_oid", "expected_failure_contract", "harness")}
        base["command_fingerprint"] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with patch("rolloutbench.formal_runner.build_episode_invocation", return_value=base):
            invocation = build_formal_invocation(
                self.context, self.unit, self.context.run["workers"][0],
                lease_files={"GPU-1": "/lease.json"}, profile={}, quality_protocol={},
            )
        self.assertEqual(self.unit.unit_id, invocation["unit_id"])
        self.assertEqual("primary", invocation["unit_kind"])
        self.assertEqual(["C02"], invocation["preparation_episode_ids"])
        self.assertEqual(base["command_fingerprint"], invocation["command_fingerprint"])

    def test_legacy_dispatch_path_cannot_bypass_external_authorization(self) -> None:
        from rolloutbench.formal_runner import FormalRunnerError, dispatch_formal_serial

        context = RunContext(**{**self.context.__dict__, "run": {**self.context.run, "system": "fifo2"}})
        with self.assertRaisesRegex(FormalRunnerError, "externally authorized"):
            dispatch_formal_serial(context, object(), Path("/state"), {}, {}, {})

    def test_dense_generation_resolves_the_frozen_dense_episode(self) -> None:
        from rolloutbench.formal_runner import build_formal_invocation

        pair = {"pair_id": "C02:scene:seed-42"}
        dense = {
            "episode_id": DENSE_REFERENCE_ID,
            "candidate": {},
            "runtime_checkout": {},
        }
        context = RunContext(
            **{
                **self.context.__dict__,
                "run": {**self.context.run, "quality_dense_reference": dense},
                "preparation": {
                    **self.context.preparation,
                    "runtime_receipts": {DENSE_REFERENCE_ID: {}},
                    "materialization_receipts": {DENSE_REFERENCE_ID: {}},
                },
            }
        )
        unit = Unit(
            "DENSE:quality:x:dense_generate",
            DENSE_REFERENCE_ID,
            "cache",
            0,
            (),
            0,
            "quality_dense_generate",
            (),
            pair,
            (DENSE_REFERENCE_ID,),
        )
        base = {
            "argv": ["bash", "run"],
            "cwd": "/repo",
            "env": {},
            "output_path": "/experiment/runs/plan/pilot-serial1-repeat-01/DENSE/out.mp4",
            "episode_id": DENSE_REFERENCE_ID,
            "run_id": "pilot-serial1-repeat-01",
            "quality_pair_id": pair["pair_id"],
            "quality_role": "dense",
            "gpu_uuid": "GPU-1",
            "runtime_ref": "r",
            "runtime_tree_oid": "t",
            "expected_failure_contract": None,
            "harness": {},
        }
        identity = {
            key: base[key]
            for key in (
                "argv", "env", "cwd", "output_path", "episode_id", "run_id",
                "quality_pair_id", "quality_role", "gpu_uuid", "runtime_ref",
                "runtime_tree_oid", "expected_failure_contract", "harness",
            )
        }
        base["command_fingerprint"] = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        with patch(
            "rolloutbench.formal_runner.build_episode_invocation", return_value=base
        ) as builder:
            result = build_formal_invocation(
                context,
                unit,
                context.run["workers"][0],
                lease_files={"GPU-1": "/lease.json"},
                profile={},
                quality_protocol={},
            )
        self.assertEqual(DENSE_REFERENCE_ID, builder.call_args.kwargs["episode"]["episode_id"])
        self.assertEqual(DENSE_REFERENCE_ID, result["episode_id"])

    def test_vbench_role_plan_binds_the_stage_gpu_uuid(self) -> None:
        from rolloutbench.formal_runner import build_vbench_role_plan

        pair = {"pair_id": "C02:scene:seed-42"}
        unit = Unit(
            "C02:quality:C02:scene:seed-42:dense_vbench",
            "C02",
            "cache",
            0,
            (),
            0,
            "quality_dense_vbench",
            (),
            pair,
            (DENSE_REFERENCE_ID, "C02"),
        )
        expected = {"plan_fingerprint": "f" * 64}
        with patch(
            "rolloutbench.formal_runner.build_vbench_pair_plan",
            return_value=expected,
        ) as builder:
            result = build_vbench_role_plan(
                self.context,
                unit,
                dense_video_path="/dense.mp4",
                candidate_video_path="/candidate.mp4",
                dense_receipt={},
                candidate_receipt={},
                profile={
                    "vbench_source_path": "/vbench",
                    "vbench_cache_path": "/cache",
                    "vbench_python_bin": "/python",
                },
                quality_protocol={"vbench": {"git_ref": "r"}},
                gpu_uuid="GPU-1",
            )
        self.assertEqual(expected, result)
        self.assertEqual("GPU-1", builder.call_args.kwargs["gpu_uuid"])

    def test_lpips_invocation_preserves_system_executable_path(self) -> None:
        from rolloutbench.formal_runner import build_formal_invocation

        pair = {
            "pair_id": "C02:scene:seed-42",
            "dense_artifact_id": "dense/quality_v1/scene/seed-42",
            "candidate_artifact_id": "candidate/C02/quality_v1/scene/seed-42",
        }
        dense = Unit(
            "DENSE:quality:C02:scene:seed-42:dense_generate",
            DENSE_REFERENCE_ID,
            "cache",
            0,
            (),
            0,
            "quality_dense_generate",
            (),
            pair,
            (DENSE_REFERENCE_ID,),
        )
        candidate = Unit(
            "C02:quality:C02:scene:seed-42:candidate_generate",
            "C02",
            "cache",
            0,
            (),
            0,
            "quality_candidate_generate",
            (),
            pair,
            ("C02",),
        )
        lpips = Unit(
            "C02:quality:C02:scene:seed-42:lpips",
            "C02",
            "cache",
            0,
            (dense.unit_id, candidate.unit_id),
            0,
            "quality_lpips",
            (),
            pair,
            (DENSE_REFERENCE_ID, "C02"),
        )
        with (
            patch(
                "rolloutbench.formal_runner.expand_run_units",
                return_value=(dense, candidate, lpips),
            ),
            patch(
                "rolloutbench.formal_runner._completed_invocation",
                side_effect=({}, {}),
            ),
            patch(
                "rolloutbench.formal_runner._completed_video",
                side_effect=((Path("/dense.mp4"), {}), (Path("/candidate.mp4"), {})),
            ),
        ):
            invocation = build_formal_invocation(
                self.context,
                lpips,
                self.context.run["workers"][0],
                lease_files={"GPU-1": "/lease.json"},
                profile={
                    "vbench_cache_path": "/cache",
                    "vbench_python_bin": "/python",
                },
                quality_protocol={},
                ledger=object(),
            )

        self.assertEqual(
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            invocation["env"]["PATH"],
        )

    def test_compare_builds_a_parser_verified_receipt_and_checks_lpips_mean(self) -> None:
        from rolloutbench.formal_runner import FormalRunnerError, FormalStageExecutor

        repo = Path(__file__).resolve().parents[2]
        protocol = json.loads(
            (repo / "benchmarks/sana_video_2b_h100_v0/quality_protocol.json").read_text()
        )
        pair = build_quality_plan(protocol, ["C02"])[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "VBench"
            (source / "vbench").mkdir(parents=True)
            (source / "evaluate.py").write_text("# fixture\n")
            (source / "vbench/VBench_full_info.json").write_text("[]\n")
            cache = root / "cache"
            cache.mkdir()
            python = root / "python"
            python.write_bytes(b"")
            dense_video = root / "dense.mp4"
            candidate_video = root / "candidate.mp4"
            dense_video.write_bytes(b"dense")
            candidate_video.write_bytes(b"candidate")
            with patch("rolloutbench.vbench_runner._validate_source_checkout"):
                plan = build_vbench_pair_plan(
                    pair,
                    quality_protocol=protocol,
                    dense_video_path=dense_video.resolve(),
                    candidate_video_path=candidate_video.resolve(),
                    dense_receipt={
                        "artifact_id": pair["dense_artifact_id"],
                        "video_path": str(dense_video.resolve()),
                        "sha256": hashlib.sha256(b"dense").hexdigest(),
                    },
                    candidate_receipt={
                        "artifact_id": pair["candidate_artifact_id"],
                        "video_path": str(candidate_video.resolve()),
                        "sha256": hashlib.sha256(b"candidate").hexdigest(),
                    },
                    vbench_source_path=source.resolve(),
                    vbench_source_ref=protocol["vbench"]["git_ref"],
                    vbench_cache_path=cache.resolve(),
                    python_bin=python.resolve(),
                    output_path=(root / "official").resolve(),
                    gpu_uuid="GPU-1",
                )
            dense_result = root / "dense-result.json"
            candidate_result = root / "candidate-result.json"
            dense_result.write_text(json.dumps({
                metric: [1.0, []] for metric in pair["metrics"]
            }))
            candidate_result.write_text(json.dumps({
                metric: [0.999, []] for metric in pair["metrics"]
            }))
            values = [0.1] * 81
            lpips_identity = {
                "schema_version": 1,
                "status": "COMPLETED",
                "metric": "lpips_v0.1_alex",
                "pair_id": pair["pair_id"],
                "alignment": "all_corresponding_decoded_frames",
                "frame_count": 81,
                "frame_shape_hwc": [480, 832, 3],
                "dense_video": {
                    "path": plan["videos"]["dense"]["path"],
                    "sha256": plan["videos"]["dense"]["sha256"],
                },
                "candidate_video": {
                    "path": plan["videos"]["candidate"]["path"],
                    "sha256": plan["videos"]["candidate"]["sha256"],
                },
                "values": values,
                "mean": 0.1,
                "performance_claim": False,
            }
            lpips = {
                **lpips_identity,
                "result_fingerprint": hashlib.sha256(
                    json.dumps(
                        lpips_identity, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
            }
            lpips_path = root / "lpips.json"
            lpips_path.write_text(json.dumps(lpips))
            output = root / "scores.json"
            invocation = {
                "formal_compare": {
                    "pair": pair,
                    "plan": plan,
                    "quality_protocol": protocol,
                    "dense_result": str(dense_result),
                    "candidate_result": str(candidate_result),
                    "lpips": str(lpips_path),
                },
                "output_path": str(output),
            }
            result = FormalStageExecutor(object()).execute(
                invocation, log_dir=root / "logs"
            )
            payload = json.loads(output.read_text())
            self.assertEqual(0, result.returncode)
            self.assertEqual("PARSED", payload["status"])
            self.assertEqual(len(pair["metrics"]), len(payload["score_rows"]))
            self.assertEqual(
                plan["plan_fingerprint"],
                payload["evidence_chain"]["plan_fingerprint"],
            )

            lpips_identity["mean"] = 0.2
            bad = {
                **lpips_identity,
                "result_fingerprint": hashlib.sha256(
                    json.dumps(
                        lpips_identity, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
            }
            lpips_path.write_text(json.dumps(bad))
            from rolloutbench.formal_runner import verify_formal_compare_output

            with self.assertRaisesRegex(FormalRunnerError, "LPIPS"):
                verify_formal_compare_output(output, pair, protocol)
            with self.assertRaisesRegex(FormalRunnerError, "LPIPS"):
                FormalStageExecutor(object()).execute(
                    {**invocation, "output_path": str(root / "bad.json")},
                    log_dir=root / "bad-logs",
                )

    def test_declared_k22_failure_is_a_completed_fail_closed_contract(self) -> None:
        from rolloutbench.formal_runner import FormalStageExecutor
        from rolloutbench.pilot_runner import ProcessResult

        class Delegate:
            def __init__(self, warning_source_line: str) -> None:
                self.warning_source_line = warning_source_line

            def execute(self, invocation, *, log_dir):
                output = Path(invocation["output_path"])
                _write_k22_test_evidence(
                    output,
                    source,
                    post_sentinel_extra=(
                        "[09-02 15:16:34] Generator was garbage collected without "
                        "being shut down. Attempting to shut down the local server "
                        "and client.\n"
                        "/runtime/python3.12/multiprocessing/resource_tracker.py:279: "
                        "UserWarning: resource_tracker: There appear to be 1 leaked "
                        "semaphore objects to clean up at shutdown\n"
                        + self.warning_source_line
                    ),
                )
                log_dir.mkdir(parents=True)
                stdout = log_dir / "stdout.log"
                stderr = log_dir / "stderr.log"
                stdout.write_bytes(b"layout mismatch\n")
                stderr.write_bytes(b"")
                return ProcessResult(
                    1,
                    2.0,
                    stdout,
                    stderr,
                    hashlib.sha256(stdout.read_bytes()).hexdigest(),
                    stdout.stat().st_size,
                    hashlib.sha256(stderr.read_bytes()).hexdigest(),
                    0,
                )

        for warning_source_line in (
            "warnings.warn('resource_tracker: There appear to be %d '\n",
            "warnings.warn('resource_tracker: There appear to be %d ')\n",
        ):
            with (
                self.subTest(warning_source_line=warning_source_line),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                invocation, source = _k22_test_invocation(root)
                result = FormalStageExecutor(Delegate(warning_source_line)).execute(
                    invocation,
                    log_dir=root / "logs",
                )
                self.assertEqual(0, result.returncode)

    def test_k22_rejects_unrelated_fatal_errors_after_generation_starts(self) -> None:
        from rolloutbench.formal_runner import FormalRunnerError, FormalStageExecutor
        from rolloutbench.pilot_runner import ProcessResult

        class MixedFailureDelegate:
            def execute(self, invocation, *, log_dir):
                output = Path(invocation["output_path"])
                _write_k22_test_evidence(
                    output,
                    source,
                    generation_extra=generation_extra,
                )
                log_dir.mkdir(parents=True)
                stdout, stderr = log_dir / "stdout.log", log_dir / "stderr.log"
                stdout.write_text("")
                stderr.write_text("")
                return ProcessResult(
                    1,
                    2.0,
                    stdout,
                    stderr,
                    hashlib.sha256(stdout.read_bytes()).hexdigest(),
                    0,
                    hashlib.sha256(stderr.read_bytes()).hexdigest(),
                    0,
                )

        extras = (
            "torch.OutOfMemoryError: CUDA out of memory\n",
            "ModuleNotFoundError: No module named 'runtime_dependency'\n",
            "AssertionError\n",
            "KeyboardInterrupt\n",
            "SystemExit: 1\n",
            "GeneratorExit\n",
            "ExceptionGroup: unrelated terminal failure (1 sub-exception)\n",
            "Segmentation fault (core dumped)\n",
            "Bus error (core dumped)\n",
            "unclassified terminal footer\n",
        )
        for generation_extra in extras:
            with self.subTest(extra=generation_extra), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                invocation, source = _k22_test_invocation(root)
                with self.assertRaisesRegex(FormalRunnerError, "fail closed"):
                    FormalStageExecutor(MixedFailureDelegate()).execute(
                        invocation, log_dir=root / "logs"
                    )

    def test_k22_rejects_unknown_content_after_the_generate_fail_sentinel(self) -> None:
        from rolloutbench.formal_runner import FormalRunnerError, FormalStageExecutor
        from rolloutbench.pilot_runner import ProcessResult

        class PostSentinelDelegate:
            def execute(self, invocation, *, log_dir):
                output = Path(invocation["output_path"])
                _write_k22_test_evidence(
                    output, source, post_sentinel_extra=post_sentinel_extra
                )
                log_dir.mkdir(parents=True)
                stdout, stderr = log_dir / "stdout.log", log_dir / "stderr.log"
                stdout.write_text("")
                stderr.write_text("")
                return ProcessResult(
                    1,
                    2.0,
                    stdout,
                    stderr,
                    hashlib.sha256(stdout.read_bytes()).hexdigest(),
                    0,
                    hashlib.sha256(stderr.read_bytes()).hexdigest(),
                    0,
                )

        for post_sentinel_extra in (
            "unclassified fatal after sentinel\n",
            "torch.OutOfMemoryError: CUDA out of memory\n",
            "Bus error (core dumped)\n",
        ):
            with self.subTest(extra=post_sentinel_extra), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                invocation, source = _k22_test_invocation(root)
                with self.assertRaisesRegex(FormalRunnerError, "fail closed"):
                    FormalStageExecutor(PostSentinelDelegate()).execute(
                        invocation, log_dir=root / "logs"
                    )

    def test_rejected_probe_is_a_completed_preflight_decision(self) -> None:
        from rolloutbench.formal_runner import FormalStageExecutor
        from rolloutbench.pilot_runner import ProcessResult

        class RejectedProbeDelegate:
            def execute(self, invocation, *, log_dir):
                output = Path(invocation["output_path"])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "status": "rejected",
                            "gpu": {"uuid": "GPU-1"},
                            "lease_uuid": "GPU-1",
                        }
                    )
                )
                log_dir.mkdir(parents=True)
                stdout = log_dir / "stdout.log"
                stderr = log_dir / "stderr.log"
                stdout.write_text("probe rejected\n")
                stderr.write_text("")
                return ProcessResult(
                    2,
                    1.0,
                    stdout,
                    stderr,
                    hashlib.sha256(stdout.read_bytes()).hexdigest(),
                    stdout.stat().st_size,
                    hashlib.sha256(stderr.read_bytes()).hexdigest(),
                    stderr.stat().st_size,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = FormalStageExecutor(RejectedProbeDelegate()).execute(
                {
                    "kind": "gpu_preflight_probe",
                    "output_path": str(root / "probe-result.json"),
                    "episode_id": "K15",
                    "gpu_uuid": "GPU-1",
                },
                log_dir=root / "logs",
            )
        self.assertEqual(0, result.returncode)

    def test_probe_status_requires_the_exact_integer_returncode(self) -> None:
        from rolloutbench.formal_runner import FormalRunnerError, FormalStageExecutor
        from rolloutbench.pilot_runner import ProcessResult

        class ProbeDelegate:
            def __init__(self, status, returncode):
                self.status = status
                self.returncode = returncode

            def execute(self, invocation, *, log_dir):
                output = Path(invocation["output_path"])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "status": self.status,
                            "gpu": {"uuid": "GPU-1"},
                            "lease_uuid": "GPU-1",
                        }
                    )
                )
                log_dir.mkdir(parents=True)
                stdout = log_dir / "stdout.log"
                stderr = log_dir / "stderr.log"
                stdout.write_text("")
                stderr.write_text("")
                return ProcessResult(
                    self.returncode,
                    1.0,
                    stdout,
                    stderr,
                    hashlib.sha256(stdout.read_bytes()).hexdigest(),
                    stdout.stat().st_size,
                    hashlib.sha256(stderr.read_bytes()).hexdigest(),
                    stderr.stat().st_size,
                )

        cases = (
            ("passed", 0, True),
            ("passed", 2, False),
            ("rejected", 0, False),
            ("passed", False, False),
        )
        for status, returncode, accepted in cases:
            with self.subTest(status=status, returncode=returncode):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    invocation = {
                        "kind": "gpu_preflight_probe",
                        "output_path": str(root / "probe-result.json"),
                        "episode_id": "K15",
                        "gpu_uuid": "GPU-1",
                    }
                    executor = FormalStageExecutor(
                        ProbeDelegate(status, returncode)
                    )
                    if accepted:
                        result = executor.execute(invocation, log_dir=root / "logs")
                        self.assertEqual(0, result.returncode)
                    else:
                        with self.assertRaisesRegex(
                            FormalRunnerError, "preflight probe"
                        ):
                            executor.execute(invocation, log_dir=root / "logs")

    def test_probe_does_not_accept_import_failure_or_wrong_gpu(self) -> None:
        from rolloutbench.formal_runner import FormalRunnerError, FormalStageExecutor
        from rolloutbench.pilot_runner import ProcessResult

        class BadProbeDelegate:
            def __init__(self, payload):
                self.payload = payload

            def execute(self, invocation, *, log_dir):
                output = Path(invocation["output_path"])
                output.parent.mkdir(parents=True, exist_ok=True)
                if self.payload is not None:
                    output.write_text(json.dumps(self.payload))
                log_dir.mkdir(parents=True)
                stdout = log_dir / "stdout.log"
                stderr = log_dir / "stderr.log"
                stdout.write_text("")
                stderr.write_text("ModuleNotFoundError: requests\n")
                return ProcessResult(
                    1 if self.payload is None else 2,
                    1.0,
                    stdout,
                    stderr,
                    hashlib.sha256(stdout.read_bytes()).hexdigest(),
                    stdout.stat().st_size,
                    hashlib.sha256(stderr.read_bytes()).hexdigest(),
                    stderr.stat().st_size,
                )

        cases = (
            (None, "missing output"),
            (
                {
                    "schema_version": 1,
                    "status": "rejected",
                    "gpu": {"uuid": "GPU-other"},
                    "lease_uuid": "GPU-other",
                },
                "wrong GPU",
            ),
        )
        for payload, label in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaisesRegex(FormalRunnerError, "preflight probe"):
                    FormalStageExecutor(BadProbeDelegate(payload)).execute(
                        {
                            "kind": "gpu_preflight_probe",
                            "output_path": str(root / "probe-result.json"),
                            "episode_id": "K15",
                            "gpu_uuid": "GPU-1",
                        },
                        log_dir=root / "logs",
                    )

    def test_k22_does_not_accept_oom_or_import_failures(self) -> None:
        from rolloutbench.formal_runner import FormalRunnerError, FormalStageExecutor
        from rolloutbench.pilot_runner import ProcessResult

        class WrongFailureDelegate:
            def __init__(self, failure_text: str) -> None:
                self.failure_text = failure_text

            def execute(self, invocation, *, log_dir):
                output = Path(invocation["output_path"])
                output.parent.mkdir(parents=True, exist_ok=True)
                run_log = output.parent / "run.log"
                run_log.write_text(self.failure_text + "\n")
                (output.parent / "run_config.json").write_text(
                    json.dumps(
                        {
                            "config_id": K22_FAILURE_CONTRACT["config_id"],
                            "source": {
                                "runtime_authority_sha": K22_FAILURE_CONTRACT[
                                    "runtime_ref"
                                ]
                            },
                        }
                    )
                )
                output.write_text(
                    json.dumps(
                        {
                            "status": "FAILED",
                            "returncode": 4,
                            "generation_s": None,
                            "process_wall_s": 1.0,
                            "residual_compute_apps": [],
                            "failure": {
                                "failure_code": "UNRELATED_RUNTIME_FAILURE"
                            },
                        }
                    )
                )
                log_dir.mkdir(parents=True)
                stdout = log_dir / "stdout.log"
                stderr = log_dir / "stderr.log"
                stdout.write_text("")
                stderr.write_text(self.failure_text)
                return ProcessResult(
                    1,
                    1.0,
                    stdout,
                    stderr,
                    hashlib.sha256(stdout.read_bytes()).hexdigest(),
                    stdout.stat().st_size,
                    hashlib.sha256(stderr.read_bytes()).hexdigest(),
                    stderr.stat().st_size,
                )

        for failure_text in ("CUDA out of memory", "ModuleNotFoundError: torch"):
            with self.subTest(failure=failure_text), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                invocation, _source = _k22_test_invocation(root)
                with self.assertRaisesRegex(FormalRunnerError, "fail closed"):
                    FormalStageExecutor(WrongFailureDelegate(failure_text)).execute(
                        invocation,
                        log_dir=root / "logs",
                    )

    def test_formal_evidence_write_is_idempotent_but_never_overwritten(self) -> None:
        from rolloutbench.formal_runner import FormalRunnerError, _write_atomic

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sealed.json"
            _write_atomic(path, b"one\n")
            _write_atomic(path, b"one\n")
            with self.assertRaisesRegex(FormalRunnerError, "conflicting"):
                _write_atomic(path, b"two\n")


if __name__ == "__main__":
    unittest.main()
