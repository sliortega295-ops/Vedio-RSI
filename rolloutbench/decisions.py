from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .events import EventLedger
from .failure_evidence import K22EvidenceError, validate_k22_failure_artifacts
from .pilot_runner import RunContext, Unit
from .quality_contract import K22_FAILURE_CONTRACT
from .validators import validate_kernel_structure


class DecisionError(RuntimeError):
    """Raised when fresh execution evidence cannot support a run decision."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise DecisionError(f"{label} is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DecisionError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise DecisionError(f"{label} must be an object")
    return value


def _write_same(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    content = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != content:
            raise DecisionError(f"refusing to overwrite conflicting receipt: {path}")
    else:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
            directory_fd = os.open(
                path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def _run_root(context: RunContext) -> Path:
    return (
        Path(context.preparation["experiment_root"])
        / "runs"
        / context.plan_id
        / context.plan_sha256
        / str(context.run["run_id"])
        / context.run_sha256
    )


def _episode(context: RunContext, episode_id: str) -> Mapping[str, Any]:
    rows = [
        row
        for row in context.run.get("episodes", [])
        if isinstance(row, Mapping) and row.get("episode_id") == episode_id
    ]
    if len(rows) != 1:
        raise DecisionError(f"cannot resolve planned episode {episode_id}")
    return rows[0]


def _primary_completion(
    context: RunContext, ledger: EventLedger, episode_id: str
) -> dict[str, Any]:
    unit_id = f"{episode_id}:primary"
    rows = [
        row["payload"]
        for row in ledger.read()
        if row["event_type"] == "stage_completed"
        and row["payload"].get("episode_id") == unit_id
        and row["payload"].get("stage") == "primary"
    ]
    if len(rows) != 1:
        raise DecisionError(f"primary completion is missing or ambiguous for {episode_id}")
    completion = rows[0]
    if (
        completion.get("plan_id") != context.plan_id
        or completion.get("plan_sha256") != context.plan_sha256
        or completion.get("run_id") != context.run["run_id"]
        or completion.get("run_sha256") != context.run_sha256
    ):
        raise DecisionError("primary completion is not bound to this run")
    output = Path(str(completion.get("output_path", "")))
    if not output.is_file() or output.is_symlink():
        raise DecisionError("primary output is missing or unsafe")
    if (
        _sha256(output) != completion.get("output_sha256")
        or output.stat().st_size != completion.get("output_size_bytes")
    ):
        raise DecisionError("primary output changed after completion")
    return completion


def _artifact_receipts(output_dir: Path) -> dict[str, dict[str, Any]]:
    collection_path = output_dir / "collection.json"
    collection = _regular_json(collection_path, "collection receipt")
    artifacts = collection.get("artifacts")
    if collection.get("status") != "VALIDATED" or not isinstance(artifacts, Mapping):
        raise DecisionError("collection receipt is incomplete")
    required = ("out.mp4", "run_config.json", "benchmark.json", "quality.json")
    result: dict[str, dict[str, Any]] = {}
    for name in required:
        path = output_dir / name
        row = artifacts.get(name)
        if (
            not isinstance(row, Mapping)
            or not path.is_file()
            or path.is_symlink()
            or row.get("path") != str(path)
            or row.get("sha256") != _sha256(path)
            or row.get("bytes") != path.stat().st_size
        ):
            raise DecisionError(f"collection artifact is stale: {name}")
        result[name] = {
            "path": str(path),
            "sha256": str(row["sha256"]),
            "size_bytes": int(row["bytes"]),
        }
    result["collection.json"] = {
        "path": str(collection_path),
        "sha256": _sha256(collection_path),
        "size_bytes": collection_path.stat().st_size,
    }
    return result


def _positive_number(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise DecisionError(f"{label} must be finite and positive")
    return float(value)


def _is_finite_zero(value: Any) -> bool:
    """Return true only for a real, finite numeric zero (never bool/string)."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) == 0.0
    )


def _validate_workload(run_config: Mapping[str, Any], suite: Mapping[str, Any]) -> None:
    observed = run_config.get("workload")
    expected = suite.get("workload")
    if not isinstance(observed, Mapping) or not isinstance(expected, Mapping):
        raise DecisionError("workload receipt is missing")
    mapping = {
        "width": "width",
        "height": "height",
        "frames": "frames",
        "fps": "fps",
        "steps": "denoising_steps",
        "guidance_scale": "guidance_scale",
        "motion_score": "motion_score",
    }
    mismatched = [
        source
        for source, target in mapping.items()
        if observed.get(source) != expected.get(target)
    ]
    if mismatched:
        raise DecisionError(f"frozen workload mismatch: {sorted(mismatched)}")
    if observed.get("seed") not in {42, 12345}:
        raise DecisionError("workload seed is outside the formal seed set")


def _generation_evidence(
    episode: Mapping[str, Any],
    completion: Mapping[str, Any],
    suite: Mapping[str, Any],
) -> dict[str, Any]:
    output = Path(str(completion["output_path"]))
    output_dir = output.parent
    artifacts = _artifact_receipts(output_dir)
    benchmark = _regular_json(output_dir / "benchmark.json", "benchmark receipt")
    run_config = _regular_json(output_dir / "run_config.json", "run configuration")
    quality = _regular_json(output_dir / "quality.json", "video validity receipt")
    _validate_workload(run_config, suite)
    video = benchmark.get("video")
    validity = benchmark.get("validity")
    if (
        output.name != "out.mp4"
        or benchmark.get("status") != "VALIDATED"
        or benchmark.get("returncode") != 0
        or not isinstance(video, Mapping)
        or video.get("path") != str(output)
        or video.get("sha256") != completion.get("output_sha256")
        or validity != quality
        or quality.get("status") != "VALIDATED"
        or benchmark.get("residual_compute_apps") != []
    ):
        raise DecisionError("generation receipt did not validate the completed video")
    generation_s = _positive_number(benchmark.get("generation_s"), "generation_s")
    process_wall_s = _positive_number(
        benchmark.get("process_wall_s"), "process_wall_s"
    )
    phase_timings = benchmark.get("phase_timings")
    if not isinstance(phase_timings, Mapping):
        raise DecisionError("phase timing receipt is missing")
    structural = benchmark.get("structural_invariants")
    if not isinstance(structural, Mapping):
        raise DecisionError("structural invariant receipt is missing")

    component = str(episode.get("component"))
    structural_result: dict[str, Any] | None = None
    cache_summary = benchmark.get("cache")
    knobs = run_config.get("optimization_knobs")
    if not isinstance(knobs, Mapping):
        raise DecisionError("optimization knob receipt is missing")
    if component == "kernel":
        if (
            knobs.get("cache_family") != "off"
            or not _is_finite_zero(knobs.get("cache_threshold"))
            or not _is_finite_zero(knobs.get("easycache_threshold"))
            or cache_summary is not None
            or structural.get("counter_kind")
            != "contract_derived_not_profiler_observed"
            or structural.get("evidence_basis")
            != "successful_fixed_scheduler_request_plus_clean_harness_runtime_config"
        ):
            raise DecisionError("exact Kernel run enabled or reported skipped work")
        structural_result = validate_kernel_structure(
            suite["workload"], structural
        )
        if structural_result.get("pass") is not True:
            raise DecisionError("exact Kernel structure validation failed")
    elif component == "cache":
        if (
            not isinstance(cache_summary, Mapping)
            or cache_summary.get("family") != knobs.get("cache_family")
            or cache_summary.get("total_decisions") != 50
            or not isinstance(cache_summary.get("hits"), int)
            or not isinstance(cache_summary.get("computes"), int)
            or cache_summary["hits"] + cache_summary["computes"] != 50
            or structural.get("logical_dit_calls") != 100
        ):
            raise DecisionError("Cache run has an invalid skip-decision receipt")
    else:
        raise DecisionError(f"unknown episode component: {component}")

    return {
        "execution_status": "VALIDATED",
        "evidence_kind": "generation",
        "ranking_eligible": True,
        "generation_s": generation_s,
        "process_wall_s": process_wall_s,
        "ranking_latency_s": process_wall_s,
        "ranking_latency_contract": (
            "one_shot_child_process_wall_including_import_model_load_compile_warmup_"
            "generation_and_runtime_teardown"
        ),
        "max_device_memory_used_mib": benchmark.get(
            "max_device_memory_used_mib"
        ),
        "phase_timings": dict(phase_timings),
        "structural_validation": structural_result,
        "cache_summary": dict(cache_summary)
        if isinstance(cache_summary, Mapping)
        else None,
        "artifacts": artifacts,
    }


def _probe_evidence(completion: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(str(completion["output_path"]))
    result = _regular_json(output, "GPU preflight result")
    status = result.get("status")
    if status not in {"passed", "rejected"}:
        raise DecisionError("GPU preflight result has no terminal status")
    return {
        "execution_status": "VALIDATED",
        "evidence_kind": "gpu_preflight",
        "ranking_eligible": False,
        "probe_status": status,
        "generation_s": None,
        "process_wall_s": _positive_number(completion.get("wall_s"), "probe wall_s"),
        "ranking_latency_s": None,
        "ranking_latency_contract": "not_frontier_eligible_preflight_probe",
        "artifacts": {
            "probe-result.json": {
                "path": str(output),
                "sha256": _sha256(output),
                "size_bytes": output.stat().st_size,
            }
        },
    }


def _expected_k22_runtime_source(
    context: RunContext, suite: Mapping[str, Any]
) -> dict[str, Any]:
    runtime_receipts = context.preparation.get("runtime_receipts")
    receipt = runtime_receipts.get("K22") if isinstance(runtime_receipts, Mapping) else None
    authority = suite.get("authority")
    model = suite.get("model")
    if (
        not isinstance(receipt, Mapping)
        or not isinstance(authority, Mapping)
        or not isinstance(model, Mapping)
    ):
        raise DecisionError("K22 runtime source authority is missing")
    worktree = Path(str(receipt.get("worktree_path", "")))
    return {
        "harness_archival_parent": authority.get("historical_harness_ref"),
        "runtime_authority_sha": model.get("runtime_authority_ref"),
        "runtime_compat_sha": model.get("runtime_compat_ref"),
        "runtime_root": str(worktree / "external" / "sol_runtime"),
        "required_runtime_paths": receipt.get("required_runtime_paths"),
        "critical_file_sha256": receipt.get("critical_runtime_file_sha256"),
    }


def _expected_failure_evidence(
    context: RunContext,
    episode: Mapping[str, Any],
    completion: Mapping[str, Any],
    suite: Mapping[str, Any],
) -> dict[str, Any]:
    contract = episode.get("expected_failure_contract")
    if dict(contract or {}) != dict(K22_FAILURE_CONTRACT) or episode.get("episode_id") != "K22":
        raise DecisionError("expected failure contract is not the frozen K22 contract")
    output = Path(str(completion["output_path"]))
    try:
        validated = validate_k22_failure_artifacts(
            output,
            expected_source=_expected_k22_runtime_source(context, suite),
        )
    except K22EvidenceError as exc:
        raise DecisionError(
            "candidate did not match its expected fail-closed contract"
        ) from exc
    benchmark = validated["benchmark"]
    child_returncode = validated["child_returncode"]
    return {
        "execution_status": "EXPECTED_FAILURE_VALIDATED",
        "evidence_kind": "expected_fail_closed_generation",
        "ranking_eligible": False,
        "generation_s": None,
        "process_wall_s": _positive_number(
            benchmark.get("process_wall_s"), "failed process_wall_s"
        ),
        "ranking_latency_s": None,
        "ranking_latency_contract": "expected_failure_not_frontier_eligible",
        "failure_contract": dict(contract),
        "child_returncode": child_returncode,
        "artifacts": {
            "benchmark.json": {
                "path": str(output),
                "sha256": _sha256(output),
                "size_bytes": output.stat().st_size,
            }
        },
    }


def collect_primary_evidence(
    context: RunContext,
    ledger: EventLedger,
    suite: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Validate and freeze one fresh primary receipt for every planned episode."""

    results: dict[str, dict[str, Any]] = {}
    for episode_value in context.run.get("episodes", []):
        if not isinstance(episode_value, Mapping):
            raise DecisionError("planned episode is malformed")
        episode_id = str(episode_value.get("episode_id", ""))
        episode = _episode(context, episode_id)
        completion = _primary_completion(context, ledger, episode_id)
        candidate = episode.get("candidate")
        if not isinstance(candidate, Mapping):
            raise DecisionError("planned candidate is missing")
        if episode.get("expected_failure_contract") is not None:
            evidence = _expected_failure_evidence(
                context, episode, completion, suite
            )
        elif candidate.get("probe") is not None:
            evidence = _probe_evidence(completion)
        else:
            evidence = _generation_evidence(episode, completion, suite)
        receipt = {
            "schema_version": 1,
            "record_type": "fresh_primary_candidate_evidence",
            "plan_id": context.plan_id,
            "plan_sha256": context.plan_sha256,
            "run_id": context.run["run_id"],
            "run_sha256": context.run_sha256,
            "episode_id": episode_id,
            "component": episode.get("component"),
            "candidate_type": episode.get("candidate_type"),
            "result": evidence,
            "performance_claim": False,
        }
        location = _write_same(
            _run_root(context) / "primary-evidence" / f"{episode_id}.json",
            receipt,
        )
        results[episode_id] = {
            "execution_status": evidence["execution_status"],
            "evidence_kind": evidence["evidence_kind"],
            "ranking_eligible": evidence["ranking_eligible"],
            "generation_s": evidence.get("generation_s"),
            "process_wall_s": evidence.get("process_wall_s"),
            "ranking_latency_s": evidence.get("ranking_latency_s"),
            "ranking_latency_contract": evidence.get("ranking_latency_contract"),
            "receipt_path": location["path"],
            "receipt_sha256": location["sha256"],
            "receipt_size_bytes": location["size_bytes"],
        }
        if evidence.get("probe_status") is not None:
            results[episode_id]["probe_status"] = evidence["probe_status"]
    return results


def _episode_decision(
    episode: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    episode_id = str(episode["episode_id"])
    component = str(episode["component"])
    eligibility = str(episode.get("quality_eligibility"))
    if evidence.get("execution_status") == "EXPECTED_FAILURE_VALIDATED":
        outcome = "expected_failure_rejected"
        frontier_eligible = False
    elif evidence.get("evidence_kind") == "gpu_preflight":
        outcome = f"preflight_{evidence.get('probe_status')}"
        frontier_eligible = False
    elif component == "kernel":
        outcome = "exact_validated"
        frontier_eligible = True
    elif eligibility == "provenance_failed":
        outcome = "excluded_provenance_failed"
        frontier_eligible = False
    elif eligibility == "calibration_only":
        outcome = "calibration_observed"
        frontier_eligible = False
    else:
        raise DecisionError(
            f"formal Cache candidate {episode_id} lacks its quality decision"
        )
    return {
        "outcome": outcome,
        "component": component,
        "frontier_eligible": frontier_eligible,
        "measured_generation_s": evidence.get("generation_s"),
        "process_wall_s": evidence.get("process_wall_s"),
        "ranking_latency_s": evidence.get("ranking_latency_s"),
        "ranking_latency_contract": evidence.get("ranking_latency_contract"),
        "primary_evidence_path": evidence["receipt_path"],
        "primary_evidence_sha256": evidence["receipt_sha256"],
        "decision_semantics": "fresh_evidence_v1_not_historical_oracle",
    }


def _validate_existing_formal_quality_decision(
    context: RunContext,
    episode: Mapping[str, Any],
    evidence: Mapping[str, Any],
    existing: Mapping[str, Any],
) -> bool:
    episode_id = str(episode.get("episode_id", ""))
    expected_path = _run_root(context) / "quality-decisions" / f"{episode_id}.json"
    if existing.get("receipt_path") != str(expected_path):
        return False
    try:
        receipt = _regular_json(expected_path, "formal quality decision receipt")
    except DecisionError:
        return False
    if _sha256(expected_path) != existing.get("receipt_sha256"):
        return False
    pair_ids = {
        str(pair.get("pair_id", ""))
        for pair in episode.get("quality_pairs", [])
        if isinstance(pair, Mapping)
    }
    pair_evidence = receipt.get("pair_evidence")
    if (
        len(pair_ids) != 8
        or not isinstance(pair_evidence, Mapping)
        or set(pair_evidence) != pair_ids
    ):
        return False
    for row in pair_evidence.values():
        if not isinstance(row, Mapping):
            return False
        path = Path(str(row.get("path", "")))
        if (
            not path.is_absolute()
            or not path.is_file()
            or path.is_symlink()
            or _sha256(path) != row.get("sha256")
            or path.stat().st_size != row.get("size_bytes")
        ):
            return False
    result = receipt.get("result")
    outcome = existing.get("outcome")
    expected_outcome = (
        "quality_pass"
        if isinstance(result, Mapping) and result.get("pass") is True
        else "quality_rejected"
    )
    expected_header = {
        "schema_version": 1,
        "record_type": "formal_quality_candidate_decision",
        "plan_id": context.plan_id,
        "plan_sha256": context.plan_sha256,
        "run_id": context.run["run_id"],
        "run_sha256": context.run_sha256,
        "candidate_id": episode_id,
        "primary_evidence": dict(evidence),
        "performance_claim": False,
    }
    if any(receipt.get(key) != value for key, value in expected_header.items()):
        return False
    if (
        not isinstance(result, Mapping)
        or result.get("status") not in {"PASS", "FAIL"}
        or result.get("candidate_id") != episode_id
        or result.get("eligibility") != "formal"
        or result.get("protocol_id") != existing.get("contract")
        or outcome != expected_outcome
        or existing.get("frontier_eligible") is not (outcome == "quality_pass")
        or existing.get("quality_result") != result
        or existing.get("primary_evidence_path") != evidence.get("receipt_path")
        or existing.get("primary_evidence_sha256") != evidence.get("receipt_sha256")
        or existing.get("measured_generation_s") != evidence.get("generation_s")
        or existing.get("process_wall_s") != evidence.get("process_wall_s")
        or existing.get("ranking_latency_s") != evidence.get("ranking_latency_s")
        or existing.get("ranking_latency_contract")
        != evidence.get("ranking_latency_contract")
    ):
        return False
    return True


def finalize_run_decisions(
    context: RunContext,
    ledger: EventLedger,
    suite: Mapping[str, Any],
    primary_evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Seal all remaining episode decisions and one provisional run frontier."""

    episodes = [
        row
        for row in context.run.get("episodes", [])
        if isinstance(row, Mapping)
    ]
    if len(episodes) != len(context.run.get("episodes", [])):
        raise DecisionError("planned episode list is malformed")
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        evidence = primary_evidence.get(episode_id)
        if not isinstance(evidence, Mapping):
            raise DecisionError(f"primary evidence is missing for {episode_id}")
        existing = ledger.reconstruct().decisions.get(episode_id)
        if existing is not None:
            if (
                episode.get("component") == "cache"
                and episode.get("quality_eligibility") == "formal"
            ):
                valid = _validate_existing_formal_quality_decision(
                    context, episode, evidence, existing
                )
            else:
                expected = _episode_decision(episode, evidence)
                valid = all(existing.get(key) == value for key, value in expected.items())
            if not valid:
                raise DecisionError(f"pre-existing decision conflicts for {episode_id}")
            continue
        decision = _episode_decision(episode, evidence)
        receipt = {
            "schema_version": 1,
            "record_type": "fresh_candidate_decision",
            "plan_id": context.plan_id,
            "plan_sha256": context.plan_sha256,
            "run_id": context.run["run_id"],
            "run_sha256": context.run_sha256,
            "episode_id": episode_id,
            "decision": decision,
            "performance_claim": False,
        }
        location = _write_same(
            _run_root(context) / "decisions" / f"{episode_id}.json", receipt
        )
        decision = {
            **decision,
            "decision_receipt_path": location["path"],
            "decision_receipt_sha256": location["sha256"],
        }
        ledger.seal_decision(episode_id, decision)

    decisions = ledger.reconstruct().decisions
    expected_ids = {str(row["episode_id"]) for row in episodes}
    if set(decisions) != expected_ids:
        raise DecisionError("run did not seal exactly one decision per planned episode")

    ranked: dict[str, list[dict[str, Any]]] = {"kernel": [], "cache": []}
    for episode_id, decision in decisions.items():
        outcome = decision.get("outcome")
        eligible = (
            outcome == "exact_validated"
            or outcome == "quality_pass"
        )
        latency = decision.get("ranking_latency_s")
        component = decision.get("component")
        if (
            eligible
            and component in ranked
            and isinstance(latency, (int, float))
            and not isinstance(latency, bool)
            and math.isfinite(float(latency))
            and float(latency) > 0
        ):
            ranked[str(component)].append(
                {"episode_id": episode_id, "ranking_latency_s": float(latency)}
            )
    for rows in ranked.values():
        rows.sort(key=lambda row: (row["ranking_latency_s"], row["episode_id"]))
    frontier = {
        "schema_version": 1,
        "record_type": "provisional_single_repetition_frontier",
        "status": "PROVISIONAL_SINGLE_REPETITION",
        "plan_id": context.plan_id,
        "plan_sha256": context.plan_sha256,
        "run_id": context.run["run_id"],
        "run_sha256": context.run_sha256,
        "decision_count": len(decisions),
        "frontier": {
            component: rows[0]["episode_id"] if rows else None
            for component, rows in ranked.items()
        },
        "ranked": ranked,
        "legacy_oracle": suite.get("frontier_contracts", {}).get("legacy_oracle"),
        "selection_basis": (
            "fresh_single_run_ranking_latency_s_one_shot_process_wall"
        ),
        "final_frontier_requires_repetition_aggregation": True,
        "performance_claim": False,
    }
    location = _write_same(_run_root(context) / "frontier" / "run.json", frontier)
    event = ledger.append(
        "frontier_updated",
        {
            "scope": "provisional_single_repetition",
            "frontier": frontier["frontier"],
            "receipt_path": location["path"],
            "receipt_sha256": location["sha256"],
            "performance_claim": False,
        },
        idempotency_key="frontier:provisional-single-repetition",
    )
    return {
        "decision_count": len(decisions),
        "decisions": decisions,
        "frontier": frontier,
        "frontier_receipt": location,
        "frontier_event_id": event["event_id"],
    }
