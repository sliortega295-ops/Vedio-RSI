"""Fail-closed contracts for evaluating one frozen quality pair with VBench.

This module only plans commands and parses already-produced official VBench JSON.
It never launches VBench, creates output directories, or claims that quality ran.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .quality_contract import (
    FORMAL_CACHE_IDS,
    QUALITY_METRICS_BY_SUITE,
    QUALITY_SEEDS,
    VBENCH_REF,
)
from .validators import build_quality_plan


class VBenchContractError(ValueError):
    """Raised when a VBench pair plan or result fails its frozen contract."""


_PAIR_FIELDS = (
    "pair_id",
    "candidate_id",
    "prompt_suite",
    "prompt",
    "source_path",
    "source_sha256",
    "selection_sha256",
    "selected_line_number_one_based",
    "seed",
    "metrics",
    "dense_artifact_id",
    "candidate_artifact_id",
)
_FORMAL_SOURCE_VERIFICATION = "FORMAL"
_TEST_SOURCE_VERIFICATION = "NON_FORMAL_TEST_ONLY"


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _absolute_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise VBenchContractError(f"{label} must be an absolute path")
    return path


def _validate_source_checkout(source: Path) -> None:
    try:
        head = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        dirty = subprocess.check_output(
            [
                "git",
                "--no-optional-locks",
                "-C",
                str(source),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            text=True,
            stderr=subprocess.PIPE,
        ).splitlines()
        symbolic = subprocess.run(
            ["git", "-C", str(source), "symbolic-ref", "-q", "HEAD"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise VBenchContractError("VBench source is not a readable Git checkout") from exc
    if head != VBENCH_REF or dirty or symbolic.returncode != 1:
        raise VBenchContractError(
            "VBench source must be the exact detached clean frozen Git checkout"
        )


def _validate_pair(
    pair: Mapping[str, Any], quality_protocol: Mapping[str, Any]
) -> tuple[str, ...]:
    if not isinstance(pair, Mapping):
        raise VBenchContractError("quality pair must be an object")
    missing = [field for field in _PAIR_FIELDS if field not in pair]
    if missing:
        raise VBenchContractError(f"quality pair is incomplete: {missing}")

    string_fields = (
        "pair_id",
        "candidate_id",
        "prompt_suite",
        "prompt",
        "source_path",
        "dense_artifact_id",
        "candidate_artifact_id",
    )
    if any(not isinstance(pair[field], str) or not pair[field] for field in string_fields):
        raise VBenchContractError("quality pair contains an empty or non-string field")
    for field in ("source_sha256", "selection_sha256"):
        if not _is_digest(pair[field]):
            raise VBenchContractError(f"quality pair {field} is not a SHA-256 digest")

    seed = pair["seed"]
    line_number = pair["selected_line_number_one_based"]
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed not in QUALITY_SEEDS
        or not isinstance(line_number, int)
        or isinstance(line_number, bool)
        or line_number <= 0
    ):
        raise VBenchContractError("quality pair seed or source line is invalid")

    candidate_id = pair["candidate_id"]
    prompt_suite = pair["prompt_suite"]
    if candidate_id not in FORMAL_CACHE_IDS:
        raise VBenchContractError("quality pair candidate is not a formal cache candidate")
    expected_metrics = QUALITY_METRICS_BY_SUITE.get(prompt_suite)
    metrics = pair["metrics"]
    if (
        not isinstance(metrics, list)
        or not metrics
        or any(not isinstance(metric, str) or not metric for metric in metrics)
        or len(metrics) != len(set(metrics))
        or expected_metrics is None
        or tuple(metrics) != expected_metrics
    ):
        raise VBenchContractError(
            "quality pair metrics must exactly match its frozen prompt suite declaration"
        )

    expected_pair_id = f"{candidate_id}:{prompt_suite}:seed-{seed}"
    artifact_suffix = f"{prompt_suite}/seed-{seed}"
    expected_artifacts = {
        "pair_id": expected_pair_id,
        "dense_artifact_id": f"dense/quality_v1/{artifact_suffix}",
        "candidate_artifact_id": (
            f"candidate/{candidate_id}/quality_v1/{artifact_suffix}"
        ),
    }
    mismatched = [
        field for field, expected in expected_artifacts.items() if pair[field] != expected
    ]
    if mismatched:
        raise VBenchContractError(
            f"quality pair identity/artifact fields are inconsistent: {mismatched}"
        )
    if not isinstance(quality_protocol, Mapping):
        raise VBenchContractError("quality protocol must be an object")
    try:
        protocol_ref = quality_protocol["vbench"]["git_ref"]
        canonical_pairs = build_quality_plan(quality_protocol, [candidate_id])
    except (KeyError, TypeError, ValueError) as exc:
        raise VBenchContractError("quality protocol is not a valid frozen protocol") from exc
    if protocol_ref != VBENCH_REF:
        raise VBenchContractError("quality protocol VBench ref is not frozen")
    canonical_pair = next(
        (
            row
            for row in canonical_pairs
            if row["pair_id"] == pair["pair_id"]
            and row["prompt_suite"] == prompt_suite
            and row["seed"] == seed
        ),
        None,
    )
    if canonical_pair is None:
        raise VBenchContractError("quality pair is absent from the canonical frozen protocol")
    mismatched = [
        field for field in _PAIR_FIELDS if pair[field] != canonical_pair[field]
    ]
    if mismatched:
        raise VBenchContractError(
            "quality pair differs from the canonical frozen protocol: "
            f"{mismatched}"
        )
    return tuple(metrics)


def _protocol_fingerprint(quality_protocol: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(quality_protocol)).hexdigest()


def _validated_video(
    *,
    role: str,
    video_path: str | Path,
    receipt: Mapping[str, Any],
    expected_artifact_id: str,
) -> tuple[Path, str]:
    path = _absolute_path(video_path, f"{role} video path")
    if not path.is_file():
        raise VBenchContractError(f"{role} video is not a regular file: {path}")
    if not isinstance(receipt, Mapping):
        raise VBenchContractError(f"{role} video receipt must be an object")
    if receipt.get("artifact_id") != expected_artifact_id:
        raise VBenchContractError(f"{role} video receipt artifact_id mismatch")
    if receipt.get("video_path") != str(path):
        raise VBenchContractError(f"{role} video receipt path mismatch")
    declared_digest = receipt.get("sha256")
    if not _is_digest(declared_digest):
        raise VBenchContractError(f"{role} video receipt has an invalid SHA-256")
    actual_digest = _sha256(path)
    if declared_digest.lower() != actual_digest:
        raise VBenchContractError(f"{role} video receipt SHA-256 mismatch")
    return path, actual_digest


def build_vbench_pair_plan(
    pair: Mapping[str, Any],
    *,
    quality_protocol: Mapping[str, Any],
    dense_video_path: str | Path,
    candidate_video_path: str | Path,
    dense_receipt: Mapping[str, Any],
    candidate_receipt: Mapping[str, Any],
    vbench_source_path: str | Path,
    vbench_source_ref: str,
    vbench_cache_path: str | Path,
    python_bin: str | Path,
    output_path: str | Path,
    source_verification: str = _FORMAL_SOURCE_VERIFICATION,
) -> dict[str, Any]:
    """Build two deterministic, unexecuted VBench custom-input invocations."""

    metrics = _validate_pair(pair, quality_protocol)
    if vbench_source_ref != VBENCH_REF:
        raise VBenchContractError(
            f"VBench source ref must equal the frozen ref {VBENCH_REF}"
        )
    source = _absolute_path(vbench_source_path, "VBench source path")
    evaluate_script = source / "evaluate.py"
    full_info = source / "vbench" / "VBench_full_info.json"
    if not evaluate_script.is_file() or not full_info.is_file():
        raise VBenchContractError(
            "VBench source path lacks evaluate.py or vbench/VBench_full_info.json"
        )
    if source_verification == _FORMAL_SOURCE_VERIFICATION:
        _validate_source_checkout(source)
    elif source_verification != _TEST_SOURCE_VERIFICATION:
        raise VBenchContractError("source verification must be FORMAL or NON_FORMAL_TEST_ONLY")
    cache = _absolute_path(vbench_cache_path, "VBench cache path")
    if not cache.is_dir():
        raise VBenchContractError("VBench cache path is not a directory")
    python = _absolute_path(python_bin, "VBench Python path")
    if not python.is_file():
        raise VBenchContractError("VBench Python path is not a regular file")
    output = _absolute_path(output_path, "VBench output path")

    dense_path, dense_sha = _validated_video(
        role="dense",
        video_path=dense_video_path,
        receipt=dense_receipt,
        expected_artifact_id=str(pair["dense_artifact_id"]),
    )
    candidate_path, candidate_sha = _validated_video(
        role="candidate",
        video_path=candidate_video_path,
        receipt=candidate_receipt,
        expected_artifact_id=str(pair["candidate_artifact_id"]),
    )

    invocations: dict[str, dict[str, Any]] = {}
    for role, video in (("dense", dense_path), ("candidate", candidate_path)):
        role_output = output / role
        argv = [
            str(python),
            str(evaluate_script),
            "--videos_path",
            str(video),
            "--dimension",
            *metrics,
            "--mode",
            "custom_input",
            "--prompt",
            str(pair["prompt"]),
            "--output_path",
            str(role_output),
            "--full_json_dir",
            str(full_info),
            "--load_ckpt_from_local",
            "True",
        ]
        identity = {
            "role": role,
            "argv": argv,
            "cwd": str(source),
            "env": {
                "VBENCH_CACHE_DIR": str(cache),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTHONNOUSERSITE": "1",
            },
            "output_path": str(role_output),
            "result_glob": str(role_output / "results_*_eval_results.json"),
        }
        invocations[role] = {
            **identity,
            "command_fingerprint": hashlib.sha256(_canonical(identity)).hexdigest(),
            "execution_status": "NOT_RUN",
        }

    plan_identity = {
        "quality_pair": {field: pair[field] for field in _PAIR_FIELDS},
        "pair_id": pair["pair_id"],
        "candidate_id": pair["candidate_id"],
        "prompt_suite": pair["prompt_suite"],
        "seed": pair["seed"],
        "prompt": pair["prompt"],
        "metrics": list(metrics),
        "vbench_source_path": str(source),
        "vbench_source_ref": vbench_source_ref,
        "source_verification": source_verification,
        "formality": (
            "FORMAL"
            if source_verification == _FORMAL_SOURCE_VERIFICATION
            else _TEST_SOURCE_VERIFICATION
        ),
        "quality_protocol_fingerprint": _protocol_fingerprint(quality_protocol),
        "vbench_cache_path": str(cache),
        "python_bin": str(python),
        "output_path": str(output),
        "videos": {
            "dense": {
                "path": str(dense_path),
                "artifact_id": pair["dense_artifact_id"],
                "sha256": dense_sha,
            },
            "candidate": {
                "path": str(candidate_path),
                "artifact_id": pair["candidate_artifact_id"],
                "sha256": candidate_sha,
            },
        },
        "invocations": invocations,
    }
    return {
        "schema_version": 1,
        **plan_identity,
        "plan_fingerprint": hashlib.sha256(_canonical(plan_identity)).hexdigest(),
        "execution_status": "NOT_RUN",
        "performance_claim": False,
    }


def _load_metric_results(
    path_value: str | Path, metrics: tuple[str, ...]
) -> dict[str, float]:
    path = _absolute_path(path_value, "VBench eval_results path")
    if not path.is_file():
        raise VBenchContractError(f"VBench eval_results is not a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VBenchContractError(f"invalid VBench eval_results JSON: {error}") from error
    if not isinstance(payload, dict):
        raise VBenchContractError("VBench eval_results must be a JSON object")

    expected = set(metrics)
    actual = set(payload)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise VBenchContractError(f"missing declared VBench metrics: {missing}")
    if extra:
        raise VBenchContractError(f"extra VBench metrics were not requested: {extra}")

    scores: dict[str, float] = {}
    for metric in metrics:
        result = payload[metric]
        if (
            not isinstance(result, list)
            or len(result) != 2
            or not isinstance(result[1], list)
        ):
            raise VBenchContractError(
                f"metric {metric} does not use the official [overall, details] shape"
            )
        overall = result[0]
        if isinstance(overall, bool) or not isinstance(overall, (int, float)):
            raise VBenchContractError(
                f"metric {metric} overall scalar is boolean or invalid"
            )
        score = float(overall)
        if not math.isfinite(score):
            raise VBenchContractError(f"metric {metric} overall scalar is non-finite")
        scores[metric] = score
    return scores


def _load_execution_receipt(path_value: str | Path) -> dict[str, Any]:
    path = _absolute_path(path_value, "VBench execution receipt path")
    if not path.is_file():
        raise VBenchContractError("VBench execution receipt is not a regular file")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VBenchContractError(f"invalid VBench execution receipt JSON: {error}") from error
    if not isinstance(receipt, dict):
        raise VBenchContractError("VBench execution receipt must be an object")
    fingerprint = receipt.get("receipt_fingerprint")
    identity = {key: value for key, value in receipt.items() if key != "receipt_fingerprint"}
    if not _is_digest(fingerprint) or fingerprint != hashlib.sha256(_canonical(identity)).hexdigest():
        raise VBenchContractError("VBench execution receipt fingerprint mismatch")
    return receipt


def _validated_execution_results(
    plan: Mapping[str, Any], receipt: Mapping[str, Any]
) -> tuple[Path, Path]:
    if plan.get("schema_version") != 1:
        raise VBenchContractError("VBench plan schema is invalid")
    if (
        plan.get("formality") != "FORMAL"
        or plan.get("source_verification") != _FORMAL_SOURCE_VERIFICATION
    ):
        raise VBenchContractError("only a formal VBench plan may be parsed")
    plan_identity = {
        key: value
        for key, value in plan.items()
        if key not in {"schema_version", "plan_fingerprint", "execution_status", "performance_claim"}
    }
    if plan.get("plan_fingerprint") != hashlib.sha256(_canonical(plan_identity)).hexdigest():
        raise VBenchContractError("VBench plan fingerprint mismatch")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("record_type") != "vbench_execution_receipt"
        or receipt.get("status") != "COMPLETED"
        or receipt.get("formality") != "FORMAL"
        or receipt.get("plan_fingerprint") != plan.get("plan_fingerprint")
        or receipt.get("quality_protocol_fingerprint") != plan.get("quality_protocol_fingerprint")
    ):
        raise VBenchContractError("execution receipt does not bind the formal VBench plan")
    source = receipt.get("vbench_source")
    if not isinstance(source, Mapping) or source != {
        "path": plan.get("vbench_source_path"),
        "ref": plan.get("vbench_source_ref"),
        "verification": _FORMAL_SOURCE_VERIFICATION,
    }:
        raise VBenchContractError("execution receipt VBench source binding mismatch")
    invocations = receipt.get("invocations")
    planned_invocations = plan.get("invocations")
    videos = plan.get("videos")
    if not isinstance(invocations, Mapping) or not isinstance(planned_invocations, Mapping) or not isinstance(videos, Mapping):
        raise VBenchContractError("execution receipt or plan invocations are invalid")
    results: list[Path] = []
    for role in ("dense", "candidate"):
        entry = invocations.get(role)
        planned = planned_invocations.get(role)
        video = videos.get(role)
        if not isinstance(entry, Mapping) or not isinstance(planned, Mapping) or not isinstance(video, Mapping):
            raise VBenchContractError(f"execution receipt {role} invocation is invalid")
        if (
            entry.get("command_fingerprint") != planned.get("command_fingerprint")
            or entry.get("video_path") != video.get("path")
            or entry.get("video_sha256") != video.get("sha256")
        ):
            raise VBenchContractError(f"execution receipt {role} argv or video binding mismatch")
        result_path = _absolute_path(str(entry.get("result_path", "")), f"{role} VBench eval_results path")
        if not result_path.is_file() or not _is_digest(entry.get("result_sha256")):
            raise VBenchContractError(f"execution receipt {role} result is invalid")
        if _sha256(result_path) != entry["result_sha256"]:
            raise VBenchContractError(f"execution receipt {role} result SHA-256 mismatch")
        results.append(result_path)
    return results[0], results[1]


def parse_vbench_pair_results(
    pair: Mapping[str, Any],
    *,
    quality_protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    execution_receipt_path: str | Path,
) -> dict[str, Any]:
    """Parse formal VBench results only through an immutable execution receipt."""

    metrics = _validate_pair(pair, quality_protocol)
    if plan.get("quality_protocol_fingerprint") != _protocol_fingerprint(quality_protocol):
        raise VBenchContractError("VBench plan quality protocol fingerprint mismatch")
    if plan.get("quality_pair") != {field: pair[field] for field in _PAIR_FIELDS}:
        raise VBenchContractError("VBench plan pair binding mismatch")
    receipt = _load_execution_receipt(execution_receipt_path)
    dense_path, candidate_path = _validated_execution_results(plan, receipt)
    dense_scores = _load_metric_results(dense_path, metrics)
    candidate_scores = _load_metric_results(candidate_path, metrics)
    return {
        "schema_version": 1,
        "status": "PARSED",
        "pair_id": pair["pair_id"],
        "candidate_id": pair["candidate_id"],
        "metrics": list(metrics),
        "result_files": {
            "dense": {"path": str(dense_path), "sha256": _sha256(dense_path)},
            "candidate": {
                "path": str(candidate_path),
                "sha256": _sha256(candidate_path),
            },
        },
        "score_rows": [
            {
                "pair_id": pair["pair_id"],
                "metric": metric,
                "dense_score": dense_scores[metric],
                "candidate_score": candidate_scores[metric],
            }
            for metric in metrics
        ],
        "performance_claim": False,
    }
