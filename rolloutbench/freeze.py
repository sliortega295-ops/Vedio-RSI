from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .schema import validate_suite_directory


KERNEL_REF = "e8684f3fa9077d1387de44bbb0521a38ac6b7097"
CACHE_REF = "e7cf11c877a91220af2f2ea2cc5e38000c0765f8"
INTEGRATION_REF = "8b066889950b638b59b5703c39909b4ac0bf9cca"
HARNESS_REF = "d2c6407cc9b9133f3fff49fe4b561f14980d3f8b"
VBENCH_REF = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
MODEL_REVISION = "db5f398b13ca086d09a50ce156c20527773841b1"
BASELINE_SHA256 = "3fdafdf00554ae4bafc91bc7729c3ba3e96af4edebac809782e6d6122ed23954"
TRAJECTORY_PATH = "TRAJECTORY.jsonl"

_REMOTE_ROOT = "/home/jiangzhikun/yongyan_liu/Experiments/SolAgent/20260831-sana-video-2b-full-exploration"
_FORMAL_CACHE_ROUNDS = {2, 3, 4, 6, 7, 9, 10, 11, 12}
_CACHE_EXCLUSIONS = {1: "provenance_failed", 5: "calibration_only", 8: "calibration_only"}
_PREFLIGHT_ONLY_KERNEL_ROUNDS = {15, 18, 21, 23}


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        for row in rows
    )


def _git_show(repo_root: Path, ref: str, path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"{ref}:{path}"], cwd=repo_root, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"cannot read authoritative source {ref}:{path}: {message}") from exc


def _extract_config_path(record: dict[str, Any]) -> str | None:
    provenance_path = record.get("provenance", {}).get("config_path")
    if provenance_path:
        return str(provenance_path)
    command = str(record.get("run", {}).get("command", ""))
    match = re.search(r"scripts/launch_config\.py\s+(\S+)", command)
    return match.group(1) if match else None


def _git_item(
    repo_root: Path,
    ref: str,
    path: str,
    reported_sha256: str | None,
    *,
    reported_optional: bool = False,
) -> dict[str, Any]:
    blob_sha256 = _sha256(_git_show(repo_root, ref, path))
    if reported_sha256 is None and not reported_optional:
        raise RuntimeError(f"authority did not report a required SHA-256 for {ref}:{path}")
    if reported_sha256 is not None and reported_sha256 != blob_sha256:
        raise RuntimeError(
            f"authority SHA-256 mismatch for {ref}:{path}: reported {reported_sha256}, blob {blob_sha256}"
        )
    return {
        "path": path,
        "authority_reported_sha256": reported_sha256,
        "blob_sha256": blob_sha256,
        "hash_scope": "raw_git_blob_bytes",
    }


def _global_fifo_index(component: str, round_number: int) -> int:
    if component == "cache":
        return 2 * (round_number - 1) + 1
    if round_number <= 12:
        return 2 * (round_number - 1)
    return 24 + round_number - 13


def _validation_contract(component: str, historical_status: str, round_number: int) -> dict[str, Any]:
    if component == "cache":
        if round_number not in _FORMAL_CACHE_ROUNDS:
            return {
                "contract": "historical_calibration_replay_v1",
                "stages": [
                    "acquire_gpu",
                    "preflight",
                    "generate",
                    "collect",
                    "legacy_sanity",
                    "decide",
                ],
                "earliest_legal_exit": "after_decide",
            }
        return {
            "contract": "full_lossy_quality_v1",
            "stages": [
                "acquire_gpu",
                "preflight",
                "generate",
                "collect",
                "quality_v1",
                "decide",
            ],
            "earliest_legal_exit": "after_decide",
        }
    if round_number in _PREFLIGHT_ONLY_KERNEL_ROUNDS:
        return {
            "contract": "gpu_preflight_exact",
            "stages": ["acquire_gpu", "preflight", "microbenchmark", "decide"],
            "earliest_legal_exit": "after_decide",
            "early_exit_trigger": "preflight_gate_reject_after_microbenchmark",
            "preflight_completion_requires": ["gpu_lease", "probe_source", "microbenchmark", "probe_result"],
        }
    return {
        "contract": "full_exact",
        "stages": ["acquire_gpu", "preflight", "generate", "collect", "exact_validate", "decide"],
        "earliest_legal_exit": "after_recorded_failure" if historical_status == "failed" else "after_decide",
    }


def _episode(
    repo_root: Path,
    component: str,
    ref: str,
    line_number: int,
    source_line: bytes,
) -> dict[str, Any]:
    record = json.loads(source_line)
    round_number = int(record["round"])
    prefix = "K" if component == "kernel" else "C"
    episode_id = f"{prefix}{round_number:02d}"
    run = record.get("run", {})
    validity = record.get("validity", {})
    decision = record.get("decision", {})
    provenance = record.get("provenance", {})
    historical_status = str(run.get("status"))
    config_path = _extract_config_path(record)

    config = None
    if config_path is not None:
        config = _git_item(
            repo_root,
            ref,
            config_path,
            provenance.get("config_sha256"),
            reported_optional=episode_id == "C01",
        )

    probe = None
    if component == "kernel" and round_number in _PREFLIGHT_ONLY_KERNEL_ROUNDS:
        probe = {
            "source": _git_item(
                repo_root,
                ref,
                provenance["probe_source_path"],
                provenance["probe_source_sha256"],
            ),
            "result": _git_item(
                repo_root,
                ref,
                provenance["probe_result_path"],
                provenance["probe_result_sha256"],
            ),
        }

    if component == "kernel":
        quality_eligibility = "not_applicable_lossless"
    elif round_number in _FORMAL_CACHE_ROUNDS:
        quality_eligibility = "formal"
    else:
        quality_eligibility = _CACHE_EXCLUSIONS[round_number]

    previous_id = None if round_number == 1 else f"{prefix}{round_number - 1:02d}"
    reuse_inputs = (
        [{"artifact": "torch_compile_cache", "episode_id": "K01", "required": True}]
        if episode_id == "K02"
        else []
    )
    return {
        "schema_version": 2,
        "episode_id": episode_id,
        "component": component,
        "stream": component,
        "round": round_number,
        "global_fifo_index": _global_fifo_index(component, round_number),
        "depends_on": [previous_id] if previous_id else [],
        "candidate_type": "exact_kernel" if component == "kernel" else "lossy_cache",
        "quality_eligibility": quality_eligibility,
        "hypothesis": record.get("hypothesis"),
        "source": {
            "git_ref": ref,
            "path": TRAJECTORY_PATH,
            "line_number": line_number,
            "line_sha256": _sha256(source_line),
            "hash_scope": "raw_jsonl_line_without_newline",
        },
        "candidate": {
            "authority_ref": ref,
            "candidate_commit": provenance.get("candidate_commit"),
            "parent_sha": provenance.get("parent_sha"),
            "config": config,
            "probe": probe,
        },
        "validation": _validation_contract(component, historical_status, round_number),
        "resources": {
            "gpu_count": 1,
            "gpu_class": "NVIDIA H100 80GB HBM3",
            "exclusive_gpu_lease": True,
            "cpu_only": False,
        },
        "reuse": {
            "inputs": reuse_inputs,
            "namespace_scope": "system_repeat_worker",
            "undeclared_cross_episode_reuse_allowed": False,
        },
        "replay": {
            "fault_injection": (
                {
                    "kind": "external_capacity_interrupt",
                    "deterministic": True,
                    "scheduler_visible_before_occurrence": False,
                }
                if episode_id == "K05"
                else None
            )
        },
        "golden": {
            "role": "acceptance_oracle_only",
            "scheduler_visible": False,
            "historical_status": historical_status,
            "validation_status": validity.get("status"),
            "decision": {
                "outcome": decision.get("outcome"),
                "frontier_status": decision.get("frontier_status"),
                "reason": decision.get("reason"),
            },
            "historical_execution": {
                "run_dir": run.get("run_dir"),
                "run_dir_availability": "remote_only_verified" if run.get("run_dir") else "not_applicable",
                "output_video_availability": (
                    "remote_only_verified"
                    if historical_status == "validated"
                    else "missing"
                    if historical_status == "failed"
                    else "not_applicable"
                ),
                "total_s": run.get("total_s"),
                "warmed_request_s": run.get("warmed_request_s"),
                "gpu_uuid": run.get("gpu_uuid"),
                "returncode": run.get("returncode"),
                "timing_scope": run.get("timing_scope"),
            },
        },
    }


def _read_episodes(repo_root: Path) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for component, ref, expected_count in (
        ("kernel", KERNEL_REF, 23),
        ("cache", CACHE_REF, 12),
    ):
        lines = _git_show(repo_root, ref, TRAJECTORY_PATH).splitlines()
        if len(lines) != expected_count:
            raise RuntimeError(
                f"{component} authority must contain {expected_count} records, found {len(lines)}"
            )
        for line_number, line in enumerate(lines, start=1):
            episode = _episode(repo_root, component, ref, line_number, line)
            if episode["round"] != line_number:
                raise RuntimeError(f"{component} authority has out-of-order round {episode['round']}")
            episodes.append(episode)
    return episodes


def _suite(file_payloads: dict[str, bytes]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "suite_id": "sana_video_2b_h100_v0",
        "description": "Frozen 35-episode replay of the SANA-Video 2B H100 Kernel and Cache searches.",
        "authority": {
            "kernel_ref": KERNEL_REF,
            "cache_ref": CACHE_REF,
            "integration_ref": INTEGRATION_REF,
            "historical_harness_ref": HARNESS_REF,
            "trajectory_path": TRAJECTORY_PATH,
            "baseline_sha256": BASELINE_SHA256,
        },
        "model": {
            "id": "SANA-Video 2B",
            "revision": MODEL_REVISION,
            "runtime_authority_ref": "b0b7eb4d0a7f1f46118a356485f4523cf52e96dd",
            "runtime_compat_ref": "5bc0c43fb7fe548af4119a8831c4e286c982c71f",
        },
        "workload": {
            "workload_fingerprint": "9af9057207c33a3608c2fdd2047ef313c364dec6a785aa823b4d9338c555a40f",
            "prompt_sha256": "870a3daedac57a0654d62de074a480daa3d7bee82b26988ec17b75496fb2643c",
            "width": 832,
            "height": 480,
            "frames": 81,
            "fps": 16,
            "denoising_steps": 50,
            "seed": 42,
            "guidance_scale": 6,
            "motion_score": 30,
            "flow_shift": 8,
            "cfg_branches_per_step": 2,
            "logical_dit_calls": 100,
            "transformer_blocks_per_call": 20,
            "transformer_dtype": "bfloat16",
            "text_encoder_dtype": "bfloat16",
            "vae_dtype": "float32",
        },
        "environment": {
            "python": "3.12.14",
            "torch": "2.11.0+cu128",
            "triton": "3.6.0",
            "cuda": "12.8",
            "gpu": "NVIDIA H100 80GB HBM3",
        },
        "counts": {
            "total": 35,
            "components": {"kernel": 23, "cache": 12},
            "historical_status": {"validated": 29, "not_run": 4, "failed": 2},
        },
        "ordering": {
            "episodes_file_order": "K01-K23_then_C01-C12",
            "global_fifo_index_base": 0,
            "global_fifo_rule": "round_robin_K_then_C_through_round_12_then_K13_to_K23",
            "global_fifo_expansion": "K01,C01,K02,C02,...,K12,C12,K13,...,K23",
        },
        "timing": {
            "historical_candidate_scope": "warm_single_prompt_gen.generate_including_text_encoder_denoise_vae_decode_and_video_write_excluding_model_load_and_one_step_warmup",
            "ttvf_q_start": "after_atomic_GPU_lease_acquired_and_before_worker_bootstrap",
            "ttvf_q_end": "after_all_35_decisions_and_quality_v1_frontier_are_atomically_sealed",
            "ttvf_q_includes": [
                "worker_bootstrap",
                "build",
                "compile",
                "model_load",
                "warmup",
                "generate",
                "collect",
                "quality",
                "decision",
                "queue_idle",
            ],
            "blinded_visual_check_inside_ttvf_q": False,
        },
        "systems": ["serial1", "fifo2", "optroll1", "optroll2"],
        "pilot_episodes": ["K01", "K02", "K15", "K19", "K20", "K22", "C01", "C02", "C09", "C12"],
        "frontier_contracts": {
            "legacy_oracle": {"kernel": "K20", "cache": "C12"},
            "quality_v1": {"winner": None, "must_be_derived_from_quality_protocol": True},
        },
        "oracle_policy": "golden fields are acceptance-only and MUST NOT be exposed to any scheduler",
        "file_hashes": {
            name: {"sha256": _sha256(raw), "hash_scope": "raw_file_bytes"}
            for name, raw in sorted(file_payloads.items())
        },
        "episodes_sha256": _sha256(file_payloads["episodes.jsonl"]),
        "artifacts_sha256": _sha256(file_payloads["artifacts.json"]),
        "quality_protocol_sha256": _sha256(file_payloads["quality_protocol.json"]),
        "hash_design": "The three SHA-256 receipts cover each named file's raw bytes. suite.json is intentionally excluded to avoid a self-referential hash cycle.",
        "scope_boundaries": {
            "included": ["H100 trace replay", "exact Kernel candidates", "lossy Cache candidates"],
            "excluded": ["FP8", "Memory", "second model", "B200", "B300", "official full VBench"],
        },
    }


def _artifacts() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "large_artifacts_local": False,
        "policy": "Git objects and explicitly counted remote outputs were verified on 2026-09-01; they were not copied into this repository. Unprobed quality dependencies and new outputs remain missing or regenerate.",
        "artifacts": [
            {
                "artifact_id": "authority_trajectory_git_objects",
                "availability": "git_available",
                "refs": [KERNEL_REF, CACHE_REF, INTEGRATION_REF],
                "digests": {
                    "kernel_TRAJECTORY.jsonl": "f6e63b657096eabd9c7a9aa2bc6d048d693148b77d9728222b65b960b84c8768",
                    "kernel_DELIVERY.json": "2704e014dcf9920ab35eab1ccafa7a9b55e29162038d303f1592fd3f849b42c2",
                    "cache_TRAJECTORY.jsonl": "5b0a183370f40311ff2667563b8ae7c9bc8ef83088d4cf1b7bc126fae9d98500",
                    "cache_DELIVERY.json": "fdd0cfbad81b6162b3746dc123d558f6cf284165db18870bf8b84bdb9e588b85",
                    "integration_INTEGRATED-DELIVERY.json": "159b50878a54fc21ff26ba89cd57b7d52e7d190afd609fec28121c392d884cbb",
                },
            },
            {
                "artifact_id": "historical_remote_run_outputs",
                "availability": "remote_only_verified",
                "remote_host": "original H100 experiment host",
                "storage_type": "Ustor_file_posix",
                "verified_at": "2026-09-01",
                "verified_counts": {
                    "kernel_mp4": 17,
                    "cache_mp4": 12,
                    "cache_visual_verdict": 12,
                    "cache_assess_verdict": 12,
                },
                "roots": {
                    "baseline": f"{_REMOTE_ROOT}/artifacts/baseline/sana2b-baseline_bl-0001/worktree/runs/20260831-091727-sana_video_2b_h100_dense_baseline-formal-dense-retry3",
                    "kernel": f"{_REMOTE_ROOT}/artifacts/executors/sana2b-kernel_aw-0001/worktree",
                    "cache": f"{_REMOTE_ROOT}/artifacts/executors/sana2b-cache_ca-0001/worktree",
                    "integration": f"{_REMOTE_ROOT}/artifacts/integration/sana2b-kernel-cache-final/worktree",
                },
            },
            {
                "artifact_id": "vbench_pinned_source_checkout",
                "availability": "local_source_verified",
                "repository": "https://github.com/Vchitect/VBench.git",
                "git_ref": VBENCH_REF,
                "local_path": "/home/lyy/Experiments/SolRolloutBench/20260901-v0/sources/VBench",
                "verified_at": "2026-09-01",
            },
            {
                "artifact_id": "vbench_metric_weights_and_runtime_receipt",
                "availability": "missing",
                "required_before_quality_run": True,
            },
            {
                "artifact_id": "quality_v1_dense_and_candidate_videos",
                "availability": "regenerate",
                "reason": "The eight-pair mini VBench gate was not part of the historical 35-round search.",
                "local_outputs_present": False,
            },
        ],
    }


def _quality_protocol() -> dict[str, Any]:
    prompt_suites = [
        {
            "suite": "subject_consistency",
            "source_path": "prompts/prompts_per_dimension/subject_consistency.txt",
            "source_sha256": "b6b72ab4799acd7fa9bee99c34b938df0c49ca303a0abf38f17ebe15c8085c49",
            "selected_line_number_one_based": 3,
            "prompt": "a person washing the dishes",
            "selection_sha256": "0388d9179df4da12015f44777e6c56016d42bb83915d5e099e240703e0a1ab3f",
            "metrics": ["subject_consistency", "motion_smoothness"],
        },
        {
            "suite": "scene",
            "source_path": "prompts/prompts_per_dimension/scene.txt",
            "source_sha256": "87572cdb4b2d3d15bedd63edc5ac0326999fc6182d28494e02d11d41579f0bd5",
            "selected_line_number_one_based": 13,
            "prompt": "bedroom",
            "selection_sha256": "024d5d57f16e1c00291a5477cf75798fb9215703a64a1650b381a9b24a0725f9",
            "metrics": ["background_consistency"],
        },
        {
            "suite": "temporal_flickering",
            "source_path": "prompts/prompts_per_dimension/temporal_flickering.txt",
            "source_sha256": "9dac1e50d27f8a44f9329113183db36b61dfc41393b86fbf3ec5e07391f0cda6",
            "selected_line_number_one_based": 2,
            "prompt": "a toilet, frozen in time",
            "selection_sha256": "04175c2dacc00b9a608873b11c815adb59ec4ed5e44b7238c08879536564e2d5",
            "metrics": ["temporal_flickering"],
        },
        {
            "suite": "overall_consistency",
            "source_path": "prompts/prompts_per_dimension/overall_consistency.txt",
            "source_sha256": "98989092f921763a33517fd11105702a7fe63dacee6a260bb260d9759da4e29f",
            "selected_line_number_one_based": 73,
            "prompt": "A cute fluffy panda eating Chinese food in a restaurant",
            "selection_sha256": "02369a8b9987de43da8a8a246f421a20211baea835a61e9d9cfc77c4d8c928a2",
            "metrics": ["aesthetic_quality", "imaging_quality", "overall_consistency"],
        },
    ]
    return {
        "schema_version": 1,
        "protocol_id": "VBench-7D-mini-quality-v1",
        "status": "predeclared_not_run",
        "vbench": {
            "repository": "https://github.com/Vchitect/VBench.git",
            "git_ref": VBENCH_REF,
            "claim_boundary": "paper-inspired mini gate; not official full VBench",
        },
        "formal_cache_candidates": ["C02", "C03", "C04", "C06", "C07", "C09", "C10", "C11", "C12"],
        "excluded_cache_candidates": {
            "C01": "provenance_failed",
            "C05": "calibration_only",
            "C08": "calibration_only",
        },
        "prompt_selection": {
            "method": "For each source file, compute SHA256(utf8(source_relative_path) + NUL + utf8(exact_prompt_text)) for every nonempty line and select the lexicographically smallest digest.",
            "selected_before_candidate_scores": True,
            "prompt_suites": prompt_suites,
        },
        "seeds": [42, 12345],
        "matched_pairs_per_candidate": 8,
        "dimensions": [
            "subject_consistency",
            "motion_smoothness",
            "background_consistency",
            "temporal_flickering",
            "aesthetic_quality",
            "imaging_quality",
            "overall_consistency",
        ],
        "acceptance": {
            "comparison": "candidate relative drop from matched dense output",
            "max_mean_relative_drop": 0.005,
            "max_single_dimension_drop": 0.02,
            "both_thresholds_required": True,
        },
        "lpips": {
            "frames": 81,
            "alignment": "all corresponding decoded frames",
            "role": "secondary_ranking_only",
            "hard_threshold": None,
        },
        "blinded_visual_check": {
            "required_for_final_publication": True,
            "inside_deterministic_ttvf": False,
        },
    }


def freeze_suite(output_dir: Path | str, *, repo_root: Path | str = Path.cwd()) -> Path:
    output = Path(output_dir)
    root = Path(repo_root)
    payloads = {
        "episodes.jsonl": _jsonl_bytes(_read_episodes(root)),
        "artifacts.json": _json_bytes(_artifacts()),
        "quality_protocol.json": _json_bytes(_quality_protocol()),
    }
    payloads["suite.json"] = _json_bytes(_suite(payloads))
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (output / name).write_bytes(payload)
    validate_suite_directory(output, repo_root=root)
    return output
