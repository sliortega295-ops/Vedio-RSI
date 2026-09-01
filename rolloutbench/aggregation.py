from __future__ import annotations

import hashlib
import fcntl
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .pilot_runner import (
    PilotRunnerError,
    RunContext,
    load_run_context,
    open_run_ledger,
)
from .quality_contract import FORMAL_CACHE_IDS
from .scheduler import SYSTEMS
from .schema import validate_suite_directory
from .validators import select_quality_frontier


class AggregationError(RuntimeError):
    """Raised when completed repetitions cannot form one benchmark result."""


_GPU_UNIT_KINDS = frozenset(
    {
        "primary",
        "quality_dense_generate",
        "quality_candidate_generate",
        "quality_dense_vbench",
        "quality_candidate_vbench",
        "quality_lpips",
    }
)
_QUALITY_UNIT_KINDS = frozenset(
    {
        "quality_dense_generate",
        "quality_candidate_generate",
        "quality_dense_vbench",
        "quality_candidate_vbench",
        "quality_lpips",
        "quality_compare",
    }
)
_SYSTEM_ORDER = ("serial1", "fifo2", "optroll1", "optroll2")
_SYSTEM_LABELS = {
    "serial1": "Historical Sol-Agent serial 1-GPU",
    "fifo2": "Naive FIFO 2-GPU",
    "optroll1": "OptRoll typed 1-GPU",
    "optroll2": "OptRoll typed 2-GPU",
}


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


def _file_receipt(path_value: Any, label: str) -> dict[str, Any]:
    if not isinstance(path_value, (str, os.PathLike)):
        raise AggregationError(f"{label} path is missing or unsafe")
    path = Path(path_value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise AggregationError(f"{label} path is missing or unsafe")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AggregationError(f"cannot read {label}") from exc
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _verify_file_receipt(receipt: Any, label: str) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise AggregationError(f"{label} receipt is missing")
    expected = _file_receipt(receipt.get("path"), label)
    if (
        receipt.get("sha256") != expected["sha256"]
        or receipt.get("size_bytes") != expected["size_bytes"]
    ):
        raise AggregationError(f"{label} receipt changed")
    return expected


def _load_bound_json(path_value: Any, digest: Any, label: str) -> dict[str, Any]:
    path = Path(str(path_value))
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise AggregationError(f"{label} path is missing or unsafe")
    if _sha256(path) != digest:
        raise AggregationError(f"{label} digest changed")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AggregationError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise AggregationError(f"{label} must be an object")
    return value


def _validate_run_frontier(
    frontier: Mapping[str, Any],
    *,
    plan_id: str,
    plan_sha256: str,
    run_id: str,
    run_sha256: str,
    decision_count: int,
) -> None:
    frontier_value = frontier.get("frontier")
    ranked = frontier.get("ranked")
    if (
        frontier.get("schema_version") != 1
        or frontier.get("record_type")
        != "provisional_single_repetition_frontier"
        or frontier.get("status") != "PROVISIONAL_SINGLE_REPETITION"
        or frontier.get("plan_id") != plan_id
        or frontier.get("plan_sha256") != plan_sha256
        or frontier.get("run_id") != run_id
        or frontier.get("run_sha256") != run_sha256
        or frontier.get("decision_count") != decision_count
        or not isinstance(frontier_value, Mapping)
        or set(frontier_value) != {"kernel", "cache"}
        or not isinstance(ranked, Mapping)
        or set(ranked) != {"kernel", "cache"}
        or any(not isinstance(ranked[key], list) for key in ("kernel", "cache"))
        or frontier.get("selection_basis")
        != "fresh_single_run_ranking_latency_s_one_shot_process_wall"
        or frontier.get("final_frontier_requires_repetition_aggregation") is not True
        or frontier.get("performance_claim") is not False
    ):
        raise AggregationError("run frontier receipt has an invalid schema or binding")


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _interval_union(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((start, end) for start, end in intervals if end >= start)
    if not ordered:
        return 0.0
    total = 0.0
    left, right = ordered[0]
    for start, end in ordered[1:]:
        if start <= right:
            right = max(right, end)
        else:
            total += right - left
            left, right = start, end
    return total + right - left


def _stage_intervals(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    starts: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    intervals: list[dict[str, Any]] = []
    for event in events:
        payload = event.get("payload", {})
        if not isinstance(payload, Mapping) or not str(event.get("event_type", "")).startswith("stage_"):
            continue
        key = (
            str(payload.get("episode_id")),
            str(payload.get("stage")),
            int(payload.get("attempt", 1)),
        )
        if event.get("event_type") == "stage_started":
            starts[key] = event
        elif event.get("event_type") == "stage_completed":
            start = starts.get(key)
            if start is None or start.get("boot_id") != event.get("boot_id"):
                raise AggregationError("stage interval lacks a same-boot start event")
            start_ns = int(start["monotonic_ns"])
            end_ns = int(event["monotonic_ns"])
            if end_ns < start_ns:
                raise AggregationError("stage completion precedes its start")
            intervals.append(
                {
                    "unit_id": key[0],
                    "unit_kind": payload.get("unit_kind"),
                    "worker_id": payload.get("worker_id"),
                    "boot_id": event.get("boot_id"),
                    "start_s": start_ns / 1e9,
                    "end_s": end_ns / 1e9,
                    "wall_s": (end_ns - start_ns) / 1e9,
                    "output_path": payload.get("output_path"),
                }
            )
    return intervals


def _generation_phase_receipt(path_value: Any) -> dict[str, Any] | None:
    output = Path(str(path_value))
    if output.name != "out.mp4":
        return None
    benchmark_path = output.parent / "benchmark.json"
    if not benchmark_path.is_file() or benchmark_path.is_symlink():
        raise AggregationError("generation stage benchmark is missing")
    try:
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AggregationError("generation stage benchmark is invalid") from exc
    if not isinstance(benchmark, Mapping):
        raise AggregationError("generation stage benchmark must be an object")
    phase = benchmark.get("phase_timings")
    generation_s = benchmark.get("generation_s")
    if (
        benchmark.get("status") != "VALIDATED"
        or not isinstance(phase, Mapping)
        or not isinstance(generation_s, (int, float))
        or isinstance(generation_s, bool)
        or not math.isfinite(float(generation_s))
        or float(generation_s) <= 0
    ):
        raise AggregationError("generation stage timing receipt is incomplete")
    combined = phase.get("model_load_compile_warmup_s")
    return {
        "generation_s": float(generation_s),
        "model_load_compile_warmup_s": (
            float(combined)
            if isinstance(combined, (int, float))
            and not isinstance(combined, bool)
            and math.isfinite(float(combined))
            and float(combined) >= 0
            else None
        ),
    }


def _run_record(
    context: RunContext,
    state_root: Path | str,
) -> dict[str, Any]:
    ledger = open_run_ledger(context, state_root)
    events = ledger.read()
    decisions = ledger.reconstruct().decisions
    episode_ids = {
        str(row["episode_id"])
        for row in context.run.get("episodes", [])
        if isinstance(row, Mapping)
    }
    if set(decisions) != episode_ids:
        raise AggregationError(f"run decisions are incomplete: {context.run['run_id']}")
    completions = [
        event
        for event in events
        if event.get("event_type") == "run_completed"
        and event.get("payload", {}).get("run_id") == context.run["run_id"]
    ]
    starts = [event for event in events if event.get("event_type") == "run_started"]
    frontiers = [
        event
        for event in events
        if event.get("event_type") == "frontier_updated"
        and event.get("payload", {}).get("scope")
        == "provisional_single_repetition"
    ]
    if not starts or not completions or len(frontiers) != 1:
        raise AggregationError(f"run lifecycle is incomplete: {context.run['run_id']}")
    start, frontier_event, completion = starts[0], frontiers[0], completions[-1]
    if start.get("boot_id") == frontier_event.get("boot_id"):
        ttvf_s = (
            int(frontier_event["monotonic_ns"]) - int(start["monotonic_ns"])
        ) / 1e9
        ttvf_clock = "same_boot_monotonic"
    else:
        try:
            start_utc = datetime.fromisoformat(str(start["utc"]))
            end_utc = datetime.fromisoformat(str(frontier_event["utc"]))
            ttvf_s = (end_utc - start_utc).total_seconds()
        except (KeyError, TypeError, ValueError) as exc:
            raise AggregationError("cross-boot TTVF UTC receipts are invalid") from exc
        ttvf_clock = "cross_boot_utc_includes_recovery_downtime"
    if ttvf_s <= 0:
        raise AggregationError("TTVF interval is not positive")
    frontier_path = completion["payload"].get("frontier_receipt_path")
    frontier_sha256 = completion["payload"].get("frontier_receipt_sha256")
    frontier = _load_bound_json(
        frontier_path,
        frontier_sha256,
        "run frontier",
    )
    _validate_run_frontier(
        frontier,
        plan_id=context.plan_id,
        plan_sha256=context.plan_sha256,
        run_id=str(context.run["run_id"]),
        run_sha256=context.run_sha256,
        decision_count=len(decisions),
    )
    frontier_file = Path(str(frontier_path))
    ledger_receipt = _file_receipt(ledger.path, "run event ledger")
    intervals = _stage_intervals(events)
    worker_count = len(context.run.get("workers", []))
    worker_intervals: dict[tuple[int, str], list[tuple[float, float]]] = defaultdict(list)
    for row in intervals:
        if row["unit_kind"] in _GPU_UNIT_KINDS:
            worker_id = row["worker_id"]
            if type(worker_id) is not int or worker_id < 0 or worker_id >= worker_count:
                raise AggregationError("GPU stage worker is outside the run")
            worker_intervals[(worker_id, str(row["boot_id"]))].append(
                (row["start_s"], row["end_s"])
            )
    worker_busy = {
        str(worker_id): sum(
            _interval_union(interval_rows)
            for (interval_worker, _boot_id), interval_rows in worker_intervals.items()
            if interval_worker == worker_id
        )
        for worker_id in range(worker_count)
    }
    gpu_busy_s = sum(worker_busy.values())
    gpu_capacity_s = worker_count * ttvf_s
    if gpu_busy_s > gpu_capacity_s + 1e-6:
        raise AggregationError("GPU busy time exceeds the measured TTVF capacity")

    generation_phases = [
        phase
        for row in intervals
        if row["unit_kind"]
        in {"primary", "quality_dense_generate", "quality_candidate_generate"}
        and (phase := _generation_phase_receipt(row["output_path"])) is not None
    ]
    combined_values = [
        row["model_load_compile_warmup_s"]
        for row in generation_phases
        if row["model_load_compile_warmup_s"] is not None
    ]
    primary_latencies = [
        float(decision["ranking_latency_s"])
        for decision in decisions.values()
        if isinstance(decision.get("ranking_latency_s"), (int, float))
        and not isinstance(decision.get("ranking_latency_s"), bool)
        and math.isfinite(float(decision["ranking_latency_s"]))
        and float(decision["ranking_latency_s"]) > 0
    ]
    quality_wall_s = sum(
        row["wall_s"]
        for row in intervals
        if row["unit_kind"] in _QUALITY_UNIT_KINDS
    )
    return {
        "run_id": context.run["run_id"],
        "repeat_index": context.run["repeat_index"],
        "run_sha256": context.run_sha256,
        "ttvf_s": ttvf_s,
        "ttvf_clock": ttvf_clock,
        "decision_count": len(decisions),
        "decisions": decisions,
        "frontier": frontier,
        "frontier_receipt": {
            "path": str(frontier_file),
            "sha256": frontier_sha256,
            "size_bytes": frontier_file.stat().st_size,
        },
        "ledger_receipt": ledger_receipt,
        "stage_interval_count": len(intervals),
        "gpu_busy_s": gpu_busy_s,
        "gpu_capacity_s": gpu_capacity_s,
        "gpu_queue_idle_s": max(gpu_capacity_s - gpu_busy_s, 0.0),
        "scheduler_gpu_utilization": gpu_busy_s / gpu_capacity_s,
        "worker_busy_s": worker_busy,
        "quality_wall_s": quality_wall_s,
        "measured_generation_s": sum(
            row["generation_s"] for row in generation_phases
        ),
        "model_load_compile_warmup_s": sum(combined_values),
        "phase_receipt_coverage": {
            "generation_stage_count": len(generation_phases),
            "combined_build_envelope_count": len(combined_values),
            "model_load_compile_warmup_separately_observed": False,
        },
        "candidate_ranking_latency_p50_s": _percentile(primary_latencies, 0.50),
        "candidate_ranking_latency_p95_s": _percentile(primary_latencies, 0.95),
    }


def _candidate_series(
    records: Sequence[Mapping[str, Any]], component: str
) -> tuple[dict[str, list[float]], dict[str, list[Mapping[str, Any]]]]:
    latencies: dict[str, list[float]] = defaultdict(list)
    decisions: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        for episode_id, decision in record["decisions"].items():
            if decision.get("component") != component:
                continue
            decisions[episode_id].append(decision)
            latency = decision.get("ranking_latency_s")
            if (
                isinstance(latency, (int, float))
                and not isinstance(latency, bool)
                and math.isfinite(float(latency))
                and float(latency) > 0
            ):
                latencies[episode_id].append(float(latency))
    return dict(latencies), dict(decisions)


def _cv(values: Sequence[float]) -> float:
    return 0.0 if len(values) < 2 else statistics.stdev(values) / statistics.fmean(values)


def _kernel_frontier(
    repetitions: int,
    latencies: Mapping[str, Sequence[float]],
    decisions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    ranked: list[tuple[float, str]] = []
    for episode_id, rows in decisions.items():
        values = list(latencies.get(episode_id, []))
        if (
            len(rows) == repetitions
            and all(row.get("outcome") == "exact_validated" for row in rows)
            and len(values) == repetitions
        ):
            ranked.append((statistics.median(values), episode_id))
    ranked.sort()
    return {
        "status": "SELECTED" if ranked else "NO_ELIGIBLE_CANDIDATE",
        "winner": ranked[0][1] if ranked else None,
        "median_latency_s": ranked[0][0] if ranked else None,
        "eligible_ranked": [
            {"candidate_id": episode_id, "median_latency_s": latency}
            for latency, episode_id in ranked
        ],
        "selection_basis": (
            "fresh_repetition_median_ranking_latency_s_one_shot_process_wall"
        ),
    }


def _cache_frontier(
    protocol: Mapping[str, Any],
    repetitions: int,
    latencies: Mapping[str, Sequence[float]],
    decisions: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    full_scope: bool,
) -> dict[str, Any]:
    available = [candidate for candidate in FORMAL_CACHE_IDS if candidate in decisions]
    results: list[dict[str, Any]] = []
    for candidate_id in available:
        rows = list(decisions[candidate_id])
        passed = (
            len(rows) == repetitions
            and all(row.get("outcome") == "quality_pass" for row in rows)
        )
        results.append(
            {
                "protocol_id": protocol.get("protocol_id"),
                "candidate_id": candidate_id,
                "eligibility": "formal",
                "status": "PASS" if passed else "FAIL",
                "pass": passed,
            }
        )
    if full_scope:
        return select_quality_frontier(protocol, results, latencies)
    ranked = sorted(
        (
            statistics.median(latencies[candidate_id]),
            candidate_id,
        )
        for candidate_id, result in (
            (str(row["candidate_id"]), row) for row in results
        )
        if result["pass"] and len(latencies.get(candidate_id, [])) == repetitions
    )
    return {
        "status": "PILOT_SELECTED" if ranked else "PILOT_NO_PASSING_CANDIDATE",
        "winner": ranked[0][1] if ranked else None,
        "median_latency": ranked[0][0] if ranked else None,
        "selection_basis": (
            "pilot_subset_fresh_repetition_median_ranking_latency_s_"
            "one_shot_process_wall"
        ),
        "eligible_ranked": [
            {"candidate_id": candidate, "median_latency": latency}
            for latency, candidate in ranked
        ],
        "formal_full_frontier": False,
    }


def aggregate_system(
    contexts: Sequence[RunContext],
    state_root: Path | str,
    suite: Mapping[str, Any],
    quality_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate all planned repetitions for one system from durable ledgers."""

    if not contexts:
        raise AggregationError("no run contexts were supplied")
    plan_ids = {context.plan_id for context in contexts}
    plan_hashes = {context.plan_sha256 for context in contexts}
    preparation_hashes = {context.preparation_sha256 for context in contexts}
    plan_paths = {context.plan_path for context in contexts}
    preparation_paths = {context.preparation_path for context in contexts}
    systems = {str(context.run.get("system")) for context in contexts}
    scopes = {str(context.run.get("scope")) for context in contexts}
    episode_sequences = {
        tuple(
            str(row.get("episode_id", ""))
            for row in context.run.get("episodes", [])
            if isinstance(row, Mapping)
        )
        for context in contexts
    }
    if (
        len(plan_ids) != 1
        or len(plan_hashes) != 1
        or len(preparation_hashes) != 1
        or len(plan_paths) != 1
        or len(preparation_paths) != 1
        or len(systems) != 1
        or len(scopes) != 1
        or len(episode_sequences) != 1
    ):
        raise AggregationError("contexts do not belong to one plan/system/scope")
    episode_ids = list(next(iter(episode_sequences)))
    if (
        not episode_ids
        or len(episode_ids) != len(set(episode_ids))
        or any(not episode_id for episode_id in episode_ids)
    ):
        raise AggregationError("contexts have an invalid episode sequence")
    ordered = sorted(contexts, key=lambda item: item.run.get("repeat_index", -1))
    expected_indices = list(range(1, len(ordered) + 1))
    if [context.run.get("repeat_index") for context in ordered] != expected_indices:
        raise AggregationError("repetition indices are incomplete or unordered")
    repetitions = len(ordered)
    if repetitions not in {3, 5}:
        raise AggregationError("formal aggregation requires three or five repetitions")
    state_path = Path(state_root)
    if state_path.is_symlink():
        raise AggregationError("state root must not be a symlink")
    state_path = state_path.resolve()
    if not state_path.is_dir():
        raise AggregationError("state root is missing")
    plan_receipt = _file_receipt(next(iter(plan_paths)), "experiment plan")
    preparation_receipt = _file_receipt(
        next(iter(preparation_paths)), "preparation receipt"
    )
    if plan_receipt["sha256"] != next(iter(plan_hashes)):
        raise AggregationError("experiment plan changed after context loading")
    if preparation_receipt["sha256"] != next(iter(preparation_hashes)):
        raise AggregationError("preparation receipt changed after context loading")
    records = [_run_record(context, state_root) for context in ordered]
    ledger_receipts: dict[str, dict[str, Any]] = {}
    for record in records:
        receipt = _verify_file_receipt(
            record.get("ledger_receipt"),
            f"{record.get('run_id')} event ledger",
        )
        ledger_receipts[str(record["run_id"])] = receipt
    kernel_latencies, kernel_decisions = _candidate_series(records, "kernel")
    cache_latencies, cache_decisions = _candidate_series(records, "cache")

    cv_rows = {
        candidate: _cv(values[:3])
        for candidate, values in {**kernel_latencies, **cache_latencies}.items()
        if len(values) >= 3
    }
    unstable = sorted(candidate for candidate, value in cv_rows.items() if value > 0.03)
    needs_more = repetitions == 3 and bool(unstable)
    kernel = _kernel_frontier(repetitions, kernel_latencies, kernel_decisions)
    cache = _cache_frontier(
        quality_protocol,
        repetitions,
        cache_latencies,
        cache_decisions,
        full_scope=next(iter(scopes)) == "full",
    )
    final_frontier = {
        "kernel": kernel.get("winner"),
        "cache": cache.get("winner"),
    }
    legacy = dict(suite.get("frontier_contracts", {}).get("legacy_oracle", {}))
    total_ttvf = sum(float(record["ttvf_s"]) for record in records)
    total_decisions = sum(int(record["decision_count"]) for record in records)
    total_gpu_busy = sum(float(record["gpu_busy_s"]) for record in records)
    total_gpu_capacity = sum(float(record["gpu_capacity_s"]) for record in records)
    result_status = (
        "NEEDS_TWO_ADDITIONAL_REPETITIONS"
        if needs_more
        else "PILOT_AGGREGATED"
        if next(iter(scopes)) == "pilot"
        else "FULL_AGGREGATED"
    )
    return {
        "schema_version": 1,
        "record_type": "sol_rolloutbench_system_result",
        "status": result_status,
        "plan_id": next(iter(plan_ids)),
        "plan_sha256": next(iter(plan_hashes)),
        "system": next(iter(systems)),
        "scope": next(iter(scopes)),
        "repetitions": repetitions,
        "episode_ids": episode_ids,
        "suite_contract": {
            "suite_id": suite.get("suite_id"),
            "episodes_sha256": suite.get("episodes_sha256"),
            "quality_protocol_sha256": suite.get("quality_protocol_sha256"),
            "quality_protocol_id": quality_protocol.get("protocol_id"),
        },
        "source_receipts": {
            "plan": plan_receipt,
            "preparation": preparation_receipt,
            "state_root": str(state_path),
            "run_ledgers": ledger_receipts,
        },
        "adaptive_repeat_rule": {
            "threshold_cv": 0.03,
            "candidate_cv_first_three": cv_rows,
            "candidates_above_threshold": unstable,
            "additional_repetitions_required": needs_more,
        },
        "frontier": None if needs_more else final_frontier,
        "kernel_frontier": kernel,
        "cache_frontier": cache,
        "legacy_oracle": legacy,
        "frontier_agreement_with_legacy_oracle": (
            None if needs_more else final_frontier == legacy
        ),
        "metrics": {
            "time_to_validated_frontier": {
                "aggregation": "sum_of_isolated_repetition_intervals_excluding_human_gaps",
                "total_s": total_ttvf,
                "median_repetition_s": statistics.median(
                    float(record["ttvf_s"]) for record in records
                ),
            },
            "gpu_hours": total_gpu_busy / 3600.0,
            "validated_decisions_per_hour": total_decisions / (total_ttvf / 3600.0),
            "scheduler_gpu_utilization": total_gpu_busy / total_gpu_capacity,
            "gpu_queue_idle_s": sum(
                float(record["gpu_queue_idle_s"]) for record in records
            ),
            "quality_wall_s": sum(float(record["quality_wall_s"]) for record in records),
            "measured_generation_s": sum(
                float(record["measured_generation_s"]) for record in records
            ),
            "model_load_compile_warmup_s": sum(
                float(record["model_load_compile_warmup_s"]) for record in records
            ),
            "phase_boundary": (
                "model load, compile, and warmup are a combined archived-runtime envelope"
            ),
            "candidate_ranking_latency_p50_s": _percentile(
                [
                    float(value)
                    for values in (*kernel_latencies.values(), *cache_latencies.values())
                    for value in values
                ],
                0.50,
            ),
            "candidate_ranking_latency_p95_s": _percentile(
                [
                    float(value)
                    for values in (*kernel_latencies.values(), *cache_latencies.values())
                    for value in values
                ],
                0.95,
            ),
            "nvidia_smi_utilization": "PARTIAL_GENERATION_SAMPLES_NOT_FULL_RUN_INTEGRAL",
            "recovery_time": "NOT_RUN_UNLESS_FAULT_INJECTION_RECEIPTS_EXIST",
            "low_fidelity_false_reject_rate": "NOT_APPLICABLE_NO_APPROXIMATE_PREFILTER_IN_V0",
        },
        "runs": records,
        "blinded_visual_check": "OUTSIDE_TTVF_NOT_RUN_BY_AGGREGATOR",
        "performance_claim": False,
    }


def _write_locked_json(
    path: Path | str,
    value: Mapping[str, Any],
    *,
    conflict_message: str,
) -> dict[str, Any]:
    target = Path(path)
    content = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False
    ).encode("utf-8") + b"\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.parent / f".{target.name}.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if target.exists():
                if (
                    not target.is_file()
                    or target.is_symlink()
                    or target.read_bytes() != content
                ):
                    raise AggregationError(conflict_message)
            else:
                temporary: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        dir=target.parent, delete=False
                    ) as handle:
                        temporary = Path(handle.name)
                        handle.write(content)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, target)
                    temporary = None
                    directory_fd = os.open(
                        target.parent,
                        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                    )
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return {
        "path": str(target),
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "status": value.get("status"),
    }


def write_system_result(path: Path | str, result: Mapping[str, Any]) -> dict[str, Any]:
    return _write_locked_json(
        path,
        result,
        conflict_message="refusing to overwrite a conflicting system result",
    )


def _finite_metric(value: Any, label: str, *, allow_zero: bool = False) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (float(value) < 0 if allow_zero else float(value) <= 0)
    ):
        constraint = "non-negative" if allow_zero else "positive"
        raise AggregationError(f"{label} must be finite and {constraint}")
    return float(value)


def _load_system_result_file(path_value: Path | str) -> tuple[dict[str, Any], dict[str, Any]]:
    supplied = Path(path_value)
    path = supplied if supplied.is_absolute() else (Path.cwd() / supplied).absolute()
    if not path.is_file() or path.is_symlink():
        raise AggregationError(f"system result is missing or unsafe: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AggregationError(f"system result is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AggregationError("system result must be a JSON object")
    return value, {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _comparison_suite(
    suite_dir: Path | str, repo_root: Path | str | None
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    repository = (
        Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    )
    suite_path = Path(suite_dir)
    if not suite_path.is_absolute():
        suite_path = repository / suite_path
    validate_suite_directory(suite_path, repo_root=repository)
    suite_raw = (suite_path / "suite.json").read_bytes()
    protocol_raw = (suite_path / "quality_protocol.json").read_bytes()
    episodes_raw = (suite_path / "episodes.jsonl").read_bytes()
    suite = json.loads(suite_raw)
    protocol = json.loads(protocol_raw)
    try:
        rows = [
            json.loads(line)
            for line in episodes_raw.decode("utf-8").splitlines()
            if line
        ]
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AggregationError("frozen episode ledger is invalid") from exc
    episode_ids = [
        str(row.get("episode_id", "")) for row in rows if isinstance(row, Mapping)
    ]
    if (
        len(episode_ids) != 35
        or len(episode_ids) != len(rows)
        or len(set(episode_ids)) != 35
        or hashlib.sha256(episodes_raw).hexdigest() != suite.get("episodes_sha256")
        or hashlib.sha256(protocol_raw).hexdigest()
        != suite.get("quality_protocol_sha256")
    ):
        raise AggregationError("frozen suite comparison contract is inconsistent")
    return suite, protocol, episode_ids


def _replay_system_result(
    result: Mapping[str, Any],
    suite: Mapping[str, Any],
    quality_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild a sealed result from its exact plan, preparation, and ledgers."""

    sources = result.get("source_receipts")
    if not isinstance(sources, Mapping):
        raise AggregationError("system result has no replayable source receipts")
    plan_receipt = _verify_file_receipt(sources.get("plan"), "experiment plan")
    preparation_receipt = _verify_file_receipt(
        sources.get("preparation"), "preparation receipt"
    )
    state_root_value = sources.get("state_root")
    state_root = Path(str(state_root_value))
    if (
        not isinstance(state_root_value, str)
        or not state_root.is_absolute()
        or state_root.is_symlink()
        or not state_root.is_dir()
        or str(state_root.resolve()) != state_root_value
    ):
        raise AggregationError("system result state root is missing or unsafe")
    try:
        plan = json.loads(Path(plan_receipt["path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AggregationError("experiment plan is invalid JSON") from exc
    if not isinstance(plan, Mapping):
        raise AggregationError("experiment plan must be an object")
    repetitions = result.get("repetitions")
    plan_repetitions = plan.get("repetitions")
    if (
        plan.get("plan_id") != result.get("plan_id")
        or plan_receipt["sha256"] != result.get("plan_sha256")
        or type(plan_repetitions) is not int
        or plan_repetitions not in {3, 5}
        or type(repetitions) is not int
        or repetitions not in {3, 5}
        or repetitions > plan_repetitions
    ):
        raise AggregationError("system result is not bound to its formal plan")
    plan_runs = plan.get("runs")
    if not isinstance(plan_runs, list):
        raise AggregationError("experiment plan run set is missing")
    system = result.get("system")
    planned = sorted(
        (
            row
            for row in plan_runs
            if isinstance(row, Mapping) and row.get("system") == system
        ),
        key=lambda row: row.get("repeat_index", -1),
    )
    if (
        len(planned) != plan_repetitions
        or [row.get("repeat_index") for row in planned]
        != list(range(1, plan_repetitions + 1))
    ):
        raise AggregationError("formal plan system repetitions are incomplete")
    selected = planned[:repetitions]
    result_runs = result.get("runs")
    expected_run_ids = [str(row.get("run_id")) for row in selected]
    if (
        not isinstance(result_runs, list)
        or [row.get("run_id") for row in result_runs if isinstance(row, Mapping)]
        != expected_run_ids
    ):
        raise AggregationError("system result does not use the first planned repetitions")
    ledger_receipts = sources.get("run_ledgers")
    if not isinstance(ledger_receipts, Mapping) or set(ledger_receipts) != set(
        expected_run_ids
    ):
        raise AggregationError("system result ledger receipt set is incomplete")
    contexts: list[RunContext] = []
    try:
        for planned_run, run_id in zip(selected, expected_run_ids):
            context = load_run_context(
                plan_receipt["path"], preparation_receipt["path"], run_id
            )
            expected_ledger = (
                state_root
                / "plans"
                / context.plan_id
                / context.plan_sha256
                / run_id
                / context.run_sha256
                / "events.jsonl"
            )
            receipt = ledger_receipts[run_id]
            if (
                not isinstance(receipt, Mapping)
                or receipt.get("path") != str(expected_ledger)
                or context.run_sha256
                != hashlib.sha256(_canonical(planned_run)).hexdigest()
            ):
                raise AggregationError("system result ledger is not bound to its run")
            _verify_file_receipt(receipt, f"{run_id} event ledger")
            contexts.append(context)
        replayed = aggregate_system(
            contexts, state_root, suite, quality_protocol
        )
    except AggregationError:
        raise
    except (OSError, ValueError, TypeError, PilotRunnerError) as exc:
        raise AggregationError("system result source replay failed") from exc
    if _canonical(replayed) != _canonical(result):
        raise AggregationError("system result does not replay from sealed source evidence")
    return replayed


def compare_system_results(
    result_paths: Sequence[Path | str],
    suite_dir: Path | str,
    *,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate and compare exactly four sealed system-result files."""

    if len(result_paths) != len(SYSTEMS):
        raise AggregationError("comparison requires exactly four system result files")
    suite, protocol, public_ids = _comparison_suite(suite_dir, repo_root)
    loaded = [_load_system_result_file(path) for path in result_paths]
    systems: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for result, receipt in loaded:
        system = result.get("system")
        if system not in SYSTEMS or system in systems:
            raise AggregationError("system results contain an unknown or duplicate system")
        systems[str(system)] = (result, receipt)
    if set(systems) != set(SYSTEMS):
        raise AggregationError("system result set is incomplete")

    expected_suite_contract = {
        "suite_id": suite.get("suite_id"),
        "episodes_sha256": suite.get("episodes_sha256"),
        "quality_protocol_sha256": suite.get("quality_protocol_sha256"),
        "quality_protocol_id": protocol.get("protocol_id"),
    }
    plan_ids = {str(result.get("plan_id")) for result, _ in systems.values()}
    plan_hashes = {str(result.get("plan_sha256")) for result, _ in systems.values()}
    scopes = {str(result.get("scope")) for result, _ in systems.values()}
    repetitions_set = {result.get("repetitions") for result, _ in systems.values()}
    if (
        len(plan_ids) != 1
        or len(plan_hashes) != 1
        or len(scopes) != 1
        or len(repetitions_set) != 1
        or any(not value or value == "None" for value in plan_ids)
        or not all(
            len(value) == 64 and all(character in "0123456789abcdef" for character in value)
            for value in plan_hashes
        )
    ):
        raise AggregationError("system results do not share one plan/scope/repetition contract")
    scope = next(iter(scopes))
    repetitions = next(iter(repetitions_set))
    if scope not in {"pilot", "full"} or type(repetitions) is not int or repetitions not in {3, 5}:
        raise AggregationError("comparison scope or repetition count is invalid")
    expected_ids = public_ids if scope == "full" else list(suite.get("pilot_episodes", []))
    expected_status = "FULL_AGGREGATED" if scope == "full" else "PILOT_AGGREGATED"
    legacy = dict(suite.get("frontier_contracts", {}).get("legacy_oracle", {}))

    rows: list[dict[str, Any]] = []
    frontiers: list[dict[str, Any]] = []
    for system in _SYSTEM_ORDER:
        result, receipt = systems[system]
        if (
            result.get("schema_version") != 1
            or result.get("record_type") != "sol_rolloutbench_system_result"
            or result.get("status") != expected_status
            or result.get("performance_claim") is not False
            or result.get("suite_contract") != expected_suite_contract
            or result.get("episode_ids") != expected_ids
        ):
            raise AggregationError(f"{system} result header or suite binding is invalid")
        result = _replay_system_result(result, suite, protocol)
        frontier = result.get("frontier")
        if not isinstance(frontier, dict) or set(frontier) != {"kernel", "cache"}:
            raise AggregationError(f"{system} result has no comparable frontier")
        kernel_frontier = result.get("kernel_frontier")
        cache_frontier = result.get("cache_frontier")
        if (
            not isinstance(kernel_frontier, Mapping)
            or not isinstance(cache_frontier, Mapping)
            or kernel_frontier.get("winner") != frontier["kernel"]
            or cache_frontier.get("winner") != frontier["cache"]
        ):
            raise AggregationError(f"{system} component frontiers are inconsistent")
        if scope == "full" and (
            frontier != legacy
            or result.get("frontier_agreement_with_legacy_oracle") is not True
        ):
            raise AggregationError(f"{system} did not reproduce the frozen full frontier")
        frontiers.append(dict(frontier))

        runs = result.get("runs")
        if not isinstance(runs, list) or len(runs) != repetitions:
            raise AggregationError(f"{system} repetition records are incomplete")
        run_ttvf: list[float] = []
        run_busy: list[float] = []
        run_capacity: list[float] = []
        seen_run_hashes: set[str] = set()
        for repeat_index, run in enumerate(runs, start=1):
            if not isinstance(run, Mapping):
                raise AggregationError(f"{system} has a malformed run record")
            decisions = run.get("decisions")
            run_hash = run.get("run_sha256")
            expected_run_id = f"{scope}-{system}-repeat-{repeat_index:02d}"
            if (
                run.get("run_id") != expected_run_id
                or run.get("repeat_index") != repeat_index
                or not isinstance(run_hash, str)
                or len(run_hash) != 64
                or run_hash in seen_run_hashes
                or not isinstance(decisions, Mapping)
                or list(result["episode_ids"]) != expected_ids
                or set(decisions) != set(expected_ids)
                or run.get("decision_count") != len(expected_ids)
            ):
                raise AggregationError(f"{system} run does not bind the exact episode set")
            seen_run_hashes.add(run_hash)
            run_frontier = run.get("frontier")
            frontier_receipt = run.get("frontier_receipt")
            if not isinstance(run_frontier, Mapping) or not isinstance(
                frontier_receipt, Mapping
            ):
                raise AggregationError(f"{system} run frontier receipt is inconsistent")
            _validate_run_frontier(
                run_frontier,
                plan_id=str(result.get("plan_id")),
                plan_sha256=str(result.get("plan_sha256")),
                run_id=expected_run_id,
                run_sha256=run_hash,
                decision_count=len(expected_ids),
            )
            replayed_frontier = _load_bound_json(
                frontier_receipt.get("path"),
                frontier_receipt.get("sha256"),
                f"{system} run frontier",
            )
            frontier_file = Path(str(frontier_receipt.get("path")))
            if (
                replayed_frontier != dict(run_frontier)
                or frontier_file.stat().st_size != frontier_receipt.get("size_bytes")
            ):
                raise AggregationError(f"{system} run frontier evidence changed")
            run_ttvf_value = _finite_metric(run.get("ttvf_s"), "run TTVF")
            run_busy_value = _finite_metric(
                run.get("gpu_busy_s"), "GPU busy", allow_zero=True
            )
            run_capacity_value = _finite_metric(
                run.get("gpu_capacity_s"), "GPU capacity"
            )
            run_idle_value = _finite_metric(
                run.get("gpu_queue_idle_s"), "run GPU queue idle", allow_zero=True
            )
            run_utilization = _finite_metric(
                run.get("scheduler_gpu_utilization"),
                "run scheduler GPU utilization",
                allow_zero=True,
            )
            if (
                run_busy_value > run_capacity_value + 1e-6
                or run_utilization > 1.0
                or not math.isclose(
                    run_idle_value,
                    run_capacity_value - run_busy_value,
                    rel_tol=1e-9,
                    abs_tol=1e-6,
                )
                or not math.isclose(
                    run_utilization,
                    run_busy_value / run_capacity_value,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            ):
                raise AggregationError(f"{system} per-run GPU metrics do not replay")
            run_ttvf.append(run_ttvf_value)
            run_busy.append(run_busy_value)
            run_capacity.append(run_capacity_value)

        metrics = result.get("metrics")
        if not isinstance(metrics, Mapping):
            raise AggregationError(f"{system} metrics are missing")
        ttvf = metrics.get("time_to_validated_frontier")
        if not isinstance(ttvf, Mapping):
            raise AggregationError(f"{system} TTVF receipt is missing")
        total_s = _finite_metric(ttvf.get("total_s"), "total TTVF")
        gpu_hours = _finite_metric(
            metrics.get("gpu_hours"), "GPU hours", allow_zero=True
        )
        decisions_per_hour = _finite_metric(
            metrics.get("validated_decisions_per_hour"), "validated decisions/hour"
        )
        utilization = _finite_metric(
            metrics.get("scheduler_gpu_utilization"),
            "scheduler GPU utilization",
            allow_zero=True,
        )
        queue_idle_s = _finite_metric(
            metrics.get("gpu_queue_idle_s"), "GPU queue idle", allow_zero=True
        )
        expected_decisions = len(expected_ids) * repetitions
        if (
            utilization > 1.0
            or not math.isclose(total_s, sum(run_ttvf), rel_tol=1e-9, abs_tol=1e-6)
            or not math.isclose(
                gpu_hours,
                sum(run_busy) / 3600.0,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                utilization,
                sum(run_busy) / sum(run_capacity),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                decisions_per_hour,
                expected_decisions / (total_s / 3600.0),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                queue_idle_s,
                sum(capacity - busy for capacity, busy in zip(run_capacity, run_busy)),
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
        ):
            raise AggregationError(f"{system} aggregate metrics do not replay")
        rows.append(
            {
                "system": system,
                "label": _SYSTEM_LABELS[system],
                "result_receipt": receipt,
                "time_to_validated_frontier_total_s": total_s,
                "gpu_hours": gpu_hours,
                "validated_decisions_per_hour": decisions_per_hour,
                "scheduler_gpu_utilization": utilization,
                "gpu_queue_idle_s": queue_idle_s,
                "frontier": frontier,
            }
        )

    if any(frontier != frontiers[0] for frontier in frontiers[1:]):
        raise AggregationError("four systems did not reach the same validated frontier")
    serial_ttvf = next(
        row["time_to_validated_frontier_total_s"]
        for row in rows
        if row["system"] == "serial1"
    )
    for row in rows:
        row["ttvf_speedup_over_serial1"] = (
            serial_ttvf / row["time_to_validated_frontier_total_s"]
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            row["time_to_validated_frontier_total_s"],
            _SYSTEM_ORDER.index(row["system"]),
        ),
    )
    identity = {
        "plan_id": next(iter(plan_ids)),
        "plan_sha256": next(iter(plan_hashes)),
        "scope": scope,
        "repetitions": repetitions,
        "suite_contract": expected_suite_contract,
        "episode_ids": expected_ids,
        "frontier": frontiers[0],
        "input_receipts": {
            row["system"]: row["result_receipt"] for row in rows
        },
        "ttvf_total_s": {
            row["system"]: row["time_to_validated_frontier_total_s"] for row in rows
        },
    }
    return {
        "schema_version": 1,
        "record_type": "sol_rolloutbench_four_system_comparison",
        "status": (
            "FULL_COMPARISON_VALIDATED"
            if scope == "full"
            else "PILOT_COMPARISON_VALIDATED"
        ),
        **identity,
        "comparison_fingerprint": hashlib.sha256(_canonical(identity)).hexdigest(),
        "frontier_decision_agreement": True,
        "systems": rows,
        "ttvf_ranking": [row["system"] for row in ranked],
        "fastest_system": ranked[0]["system"],
        "comparison_basis": (
            "sum_of_isolated_repetition_time_to_validated_frontier_intervals"
        ),
        "claim_boundaries": {
            "nvidia_smi_utilization": "PARTIAL_NOT_USED_FOR_WINNER",
            "blinded_visual_check": "NOT_RUN_OUTSIDE_TTVF",
            "fault_injection_recovery": "NOT_RUN_UNLESS_SEPARATELY_RECEIPTED",
            "winner_metric": "time_to_validated_frontier_total_s",
        },
        "performance_claim": False,
    }


def write_system_comparison(
    path: Path | str, comparison: Mapping[str, Any]
) -> dict[str, Any]:
    return _write_locked_json(
        path,
        comparison,
        conflict_message="refusing to overwrite a conflicting four-system comparison",
    )
