from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
import types
from pathlib import Path
from typing import Any, Mapping, Sequence

from .aggregation import (
    AggregationError,
    _cache_frontier,
    _candidate_series,
    _canonical,
    _cv,
    _file_receipt,
    _kernel_frontier,
    _percentile,
    _run_record,
    _verify_file_receipt,
    _write_locked_json,
)
from .pilot_runner import RunContext, load_run_context


class AdaptivePilotError(RuntimeError):
    """Raised when an exploratory adaptive-pilot result is not replayable."""


_SYSTEM_ORDER = ("serial1", "fifo2", "optroll1", "optroll2")
_EXPECTED_COMPLETED = {
    "serial1": 3,
    "fifo2": 3,
    "optroll1": 2,
    "optroll2": 2,
}
_SEMANTIC_DECISION_KEYS = (
    "component",
    "outcome",
    "frontier_eligible",
    "decision_semantics",
    "ranking_latency_contract",
    "contract",
)
_SUITE_FILENAMES = (
    "suite.json",
    "episodes.jsonl",
    "artifacts.json",
    "quality_protocol.json",
)


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def parse_completed_repetitions(values: Sequence[str]) -> dict[str, int]:
    """Parse repeated SYSTEM=N CLI values for the frozen adaptive pilot."""

    parsed: dict[str, int] = {}
    for value in values:
        system, separator, count_text = value.partition("=")
        if not separator or system not in _EXPECTED_COMPLETED or system in parsed:
            raise AdaptivePilotError(
                "completed repetitions must contain each system exactly once as SYSTEM=N"
            )
        try:
            count = int(count_text)
        except ValueError as exc:
            raise AdaptivePilotError("completed repetition count must be an integer") from exc
        if count <= 0:
            raise AdaptivePilotError("completed repetition count must be positive")
        parsed[system] = count
    if parsed != _EXPECTED_COMPLETED:
        raise AdaptivePilotError(
            f"adaptive pilot requires exactly {_EXPECTED_COMPLETED}, got {parsed}"
        )
    return parsed


def load_contexts_by_system(
    plan_path: Path | str,
    preparation_path: Path | str,
    completed: Mapping[str, int],
) -> dict[str, list[RunContext]]:
    """Load the first N plan-bound run contexts for every benchmark system."""

    if dict(completed) != _EXPECTED_COMPLETED:
        raise AdaptivePilotError("completed repetition matrix is not the frozen pilot policy")
    try:
        plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdaptivePilotError("cannot load experiment plan") from exc
    runs = plan.get("runs") if isinstance(plan, Mapping) else None
    if not isinstance(runs, list):
        raise AdaptivePilotError("experiment plan has no run list")
    result: dict[str, list[RunContext]] = {}
    for system in _SYSTEM_ORDER:
        planned = sorted(
            (
                row
                for row in runs
                if isinstance(row, Mapping) and row.get("system") == system
            ),
            key=lambda row: row.get("repeat_index", -1),
        )
        count = completed[system]
        selected = planned[:count]
        if (
            len(selected) != count
            or [row.get("repeat_index") for row in selected]
            != list(range(1, count + 1))
        ):
            raise AdaptivePilotError(f"planned repetitions are incomplete for {system}")
        result[system] = [
            load_run_context(plan_path, preparation_path, str(row["run_id"]))
            for row in selected
        ]
    return result


def semantic_decisions(decisions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Remove paths, hashes and measurements while retaining decision meaning."""

    result: dict[str, Any] = {}
    for episode_id in sorted(decisions):
        decision = decisions[episode_id]
        semantic = {key: decision.get(key) for key in _SEMANTIC_DECISION_KEYS}
        quality = decision.get("quality_result")
        if isinstance(quality, Mapping):
            lpips = quality.get("lpips")
            semantic["quality_result"] = {
                "protocol_id": quality.get("protocol_id"),
                "eligibility": quality.get("eligibility"),
                "status": quality.get("status"),
                "pass": quality.get("pass"),
                "errors": quality.get("errors"),
                "thresholds": quality.get("thresholds"),
                "lpips": (
                    {
                        "status": lpips.get("status"),
                        "role": lpips.get("role"),
                        "hard_acceptance_effect": lpips.get(
                            "hard_acceptance_effect"
                        ),
                        "errors": lpips.get("errors"),
                    }
                    if isinstance(lpips, Mapping)
                    else None
                ),
            }
        result[episode_id] = semantic
    return result


def _relative_range(values: Sequence[float]) -> float:
    mean = statistics.fmean(values)
    if not math.isfinite(mean) or mean <= 0:
        raise AdaptivePilotError("TTVF mean must be finite and positive")
    return (max(values) - min(values)) / mean


def _read_file_receipt(
    path: Path, label: str
) -> tuple[bytes, dict[str, Any]]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise AdaptivePilotError(f"{label} path is missing or unsafe")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AdaptivePilotError(f"cannot load {label}") from exc
    return raw, {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _load_json_receipt(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, receipt = _read_file_receipt(path, label)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AdaptivePilotError(f"cannot load {label}") from exc
    if not isinstance(value, dict):
        raise AdaptivePilotError(f"{label} must be a JSON object")
    return value, receipt


def _suite_source_receipts(
    suite_path: Path | str,
    suite: Mapping[str, Any],
    quality_protocol: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    supplied = Path(suite_path)
    if supplied.is_symlink():
        raise AdaptivePilotError("suite directory must not be a symlink")
    directory = supplied.resolve()
    if not directory.is_dir():
        raise AdaptivePilotError("suite directory is missing")
    raw_files: dict[str, bytes] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for name in _SUITE_FILENAMES:
        raw, receipt = _read_file_receipt(
            directory / name, f"suite file {name}"
        )
        raw_files[name] = raw
        receipts[name] = receipt
    try:
        loaded_suite = json.loads(raw_files["suite.json"])
        loaded_protocol = json.loads(raw_files["quality_protocol.json"])
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AdaptivePilotError("suite JSON file is invalid") from exc
    if not isinstance(loaded_suite, dict) or not isinstance(loaded_protocol, dict):
        raise AdaptivePilotError("suite JSON files must contain objects")
    if _canonical(loaded_suite) != _canonical(dict(suite)):
        raise AdaptivePilotError("supplied suite object does not match suite.json")
    if _canonical(loaded_protocol) != _canonical(dict(quality_protocol)):
        raise AdaptivePilotError(
            "supplied quality protocol does not match quality_protocol.json"
        )
    try:
        episode_rows = [
            json.loads(line)
            for line in raw_files["episodes.jsonl"].decode("utf-8").splitlines()
            if line
        ]
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AdaptivePilotError("cannot load suite episode ledger") from exc
    public_ids = [
        str(row.get("episode_id", ""))
        for row in episode_rows
        if isinstance(row, Mapping)
    ]
    if (
        len(public_ids) != len(episode_rows)
        or len(public_ids) != len(set(public_ids))
        or any(not value for value in public_ids)
    ):
        raise AdaptivePilotError("suite episode ledger is invalid")
    return receipts, public_ids


def _analysis_source_manifest() -> dict[str, Any]:
    package = Path(__file__).resolve().parent
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(package.glob("*.py")):
        receipt = _file_receipt(path, f"analysis source {path.name}")
        files[path.name] = {
            "sha256": receipt["sha256"],
            "size_bytes": receipt["size_bytes"],
        }
    if "adaptive_pilot.py" not in files or "aggregation.py" not in files:
        raise AdaptivePilotError("analysis source manifest is incomplete")
    return {
        "scope": "on_disk_top_level_rolloutbench_python_modules_sampled_during_replay",
        "manifest_sha256": _object_sha256(files),
        "files": files,
    }


def _loaded_analysis_code_manifest() -> dict[str, Any]:
    """Fingerprint live Python code separately from the on-disk source tree.

    Source files can change after Python imports them.  Keeping this receipt
    separate prevents the disk manifest from being presented as proof of the
    bytecode that actually performed the replay.
    """

    def stable_value(value: Any) -> Any:
        if isinstance(value, types.CodeType):
            return {
                "kind": "code",
                "argcount": value.co_argcount,
                "posonlyargcount": value.co_posonlyargcount,
                "kwonlyargcount": value.co_kwonlyargcount,
                "nlocals": value.co_nlocals,
                "stacksize": value.co_stacksize,
                "flags": value.co_flags,
                "code_hex": value.co_code.hex(),
                "consts": [stable_value(item) for item in value.co_consts],
                "names": list(value.co_names),
                "varnames": list(value.co_varnames),
                "freevars": list(value.co_freevars),
                "cellvars": list(value.co_cellvars),
                "filename": value.co_filename,
                "name": value.co_name,
                "qualname": value.co_qualname,
                "firstlineno": value.co_firstlineno,
                "linetable_hex": value.co_linetable.hex(),
                "exceptiontable_hex": value.co_exceptiontable.hex(),
            }
        if isinstance(value, bytes):
            return {"kind": "bytes", "hex": value.hex()}
        if isinstance(value, tuple):
            return {"kind": "tuple", "items": [stable_value(item) for item in value]}
        if isinstance(value, frozenset):
            items = [stable_value(item) for item in value]
            return {
                "kind": "frozenset",
                "items": sorted(items, key=lambda item: _canonical(item)),
            }
        if isinstance(value, dict):
            return {
                "kind": "dict",
                "items": [
                    [stable_value(key), stable_value(item)]
                    for key, item in sorted(
                        value.items(), key=lambda pair: repr(pair[0])
                    )
                ],
            }
        if value is None or isinstance(value, (bool, int, float, str)):
            return {"kind": type(value).__name__, "value": value}
        return {
            "kind": f"{type(value).__module__}.{type(value).__qualname__}",
            "repr": repr(value),
        }

    def function_digest(value: types.FunctionType) -> str:
        return hashlib.sha256(
            _canonical(
                {
                    "code": stable_value(value.__code__),
                    "defaults": stable_value(value.__defaults__),
                    "kwdefaults": stable_value(value.__kwdefaults__),
                }
            )
        ).hexdigest()

    module_names = (
        "rolloutbench.adaptive_pilot",
        "rolloutbench.aggregation",
        "rolloutbench.events",
        "rolloutbench.pilot_runner",
    )
    modules: dict[str, dict[str, str]] = {}
    self_attestation_helpers = {
        "_loaded_analysis_code_manifest",
        "_analysis_implementation_receipt",
    }
    for module_name in module_names:
        module = sys.modules.get(module_name)
        if module is None:
            raise AdaptivePilotError(
                f"analysis module is not loaded: {module_name}"
            )
        callables: dict[str, str] = {}
        for name, value in sorted(vars(module).items()):
            if (
                module_name == "rolloutbench.adaptive_pilot"
                and name in self_attestation_helpers
            ):
                continue
            if (
                isinstance(value, types.FunctionType)
                and value.__module__ == module_name
            ):
                callables[name] = function_digest(value)
                continue
            if not isinstance(value, type) or value.__module__ != module_name:
                continue
            for method_name, method in sorted(vars(value).items()):
                if isinstance(method, (staticmethod, classmethod)):
                    method = method.__func__
                if not isinstance(method, types.FunctionType):
                    continue
                callables[f"{name}.{method_name}"] = function_digest(method)
        if not callables:
            raise AdaptivePilotError(
                f"analysis module has no fingerprintable code: {module_name}"
            )
        modules[module_name] = callables
    return {
        "scope": "live_loaded_python_callables_used_by_adaptive_replay",
        "python_version": sys.version,
        "manifest_sha256": _object_sha256(modules),
        "modules": modules,
    }


def _analysis_implementation_receipt() -> dict[str, Any]:
    disk_source = _analysis_source_manifest()
    loaded_code = _loaded_analysis_code_manifest()
    return {
        "scope": "separate_disk_source_and_live_loaded_code_receipts",
        "manifest_sha256": _object_sha256(
            {
                "disk_source_manifest_sha256": disk_source["manifest_sha256"],
                "loaded_code_manifest_sha256": loaded_code["manifest_sha256"],
            }
        ),
        "disk_source": disk_source,
        "loaded_code": loaded_code,
        "provenance_note": (
            "disk_source records files sampled during replay; loaded_code "
            "independently fingerprints the in-memory Python callables that "
            "performed the replay"
        ),
    }


def _validate_plan_suite_binding(
    plan: Mapping[str, Any],
    *,
    plan_id: str,
    scope: str,
    episode_ids: Sequence[str],
    suite: Mapping[str, Any],
    quality_protocol: Mapping[str, Any],
    suite_receipts: Mapping[str, Mapping[str, Any]],
    public_episode_ids: Sequence[str],
) -> None:
    suite_hashes = {
        name: receipt.get("sha256") for name, receipt in suite_receipts.items()
    }
    pilot_ids = suite.get("pilot_episodes")
    planned_runs = plan.get("runs")
    plan_matrix_valid = (
        isinstance(planned_runs, list)
        and len(planned_runs) == len(_SYSTEM_ORDER) * 5
        and all(
            isinstance(row, Mapping) and row.get("system") in _SYSTEM_ORDER
            for row in planned_runs
        )
    )
    if plan_matrix_valid:
        for system in _SYSTEM_ORDER:
            rows = [
                row
                for row in planned_runs
                if isinstance(row, Mapping) and row.get("system") == system
            ]
            rows.sort(
                key=lambda row: (
                    row.get("repeat_index")
                    if type(row.get("repeat_index")) is int
                    else -1
                )
            )
            episode_rows_valid = all(
                isinstance(row.get("episodes"), list)
                and len(row["episodes"]) == len(episode_ids)
                and all(isinstance(item, Mapping) for item in row["episodes"])
                and [item.get("episode_id") for item in row["episodes"]]
                == list(episode_ids)
                for row in rows
            )
            if (
                len(rows) != 5
                or not all(
                    type(row.get("repeat_index")) is int for row in rows
                )
                or [row.get("repeat_index") for row in rows] != list(range(1, 6))
                or [row.get("run_id") for row in rows]
                != [f"pilot-{system}-repeat-{index:02d}" for index in range(1, 6)]
                or any(row.get("scope") != scope for row in rows)
                or not episode_rows_valid
            ):
                plan_matrix_valid = False
                break
    if (
        plan.get("plan_id") != plan_id
        or plan.get("scope") != scope
        or plan.get("suite_id") != suite.get("suite_id")
        or plan.get("suite_file_sha256") != suite_hashes
        or plan.get("quality_protocol_id") != quality_protocol.get("protocol_id")
        or suite.get("systems") != list(_SYSTEM_ORDER)
        or plan.get("repetitions") != 5
        or pilot_ids != list(episode_ids)
        or any(episode_id not in public_episode_ids for episode_id in episode_ids)
        or not plan_matrix_valid
    ):
        raise AdaptivePilotError("plan, suite and pilot episode contract do not match")


def _validate_context_matrix(
    contexts_by_system: Mapping[str, Sequence[RunContext]],
) -> tuple[str, str, str, Path, Path, str, list[str]]:
    if set(contexts_by_system) != set(_SYSTEM_ORDER):
        raise AdaptivePilotError("adaptive pilot requires all four systems")
    flattened = [
        context
        for system in _SYSTEM_ORDER
        for context in contexts_by_system[system]
    ]
    if not flattened:
        raise AdaptivePilotError("adaptive pilot has no run contexts")
    plan_ids = {context.plan_id for context in flattened}
    plan_hashes = {context.plan_sha256 for context in flattened}
    preparation_hashes = {context.preparation_sha256 for context in flattened}
    plan_paths = {context.plan_path for context in flattened}
    preparation_paths = {context.preparation_path for context in flattened}
    scopes = {str(context.run.get("scope")) for context in flattened}
    if any(
        not isinstance(context.run.get("episodes"), list)
        or not context.run["episodes"]
        or any(
            not isinstance(row, Mapping)
            or not isinstance(row.get("episode_id"), str)
            or not row["episode_id"]
            for row in context.run["episodes"]
        )
        for context in flattened
    ):
        raise AdaptivePilotError("contexts have a malformed episode sequence")
    episode_sequences = {
        tuple(row["episode_id"] for row in context.run["episodes"])
        for context in flattened
    }
    if any(
        len(values) != 1
        for values in (
            plan_ids,
            plan_hashes,
            preparation_hashes,
            plan_paths,
            preparation_paths,
            scopes,
            episode_sequences,
        )
    ):
        raise AdaptivePilotError("contexts do not share one plan, scope and episode set")
    if scopes != {"pilot"}:
        raise AdaptivePilotError("adaptive analysis is restricted to pilot scope")
    episode_ids = list(next(iter(episode_sequences)))
    if not episode_ids or len(episode_ids) != len(set(episode_ids)):
        raise AdaptivePilotError("pilot episode sequence is invalid")
    for system in _SYSTEM_ORDER:
        contexts = list(contexts_by_system[system])
        if len(contexts) != _EXPECTED_COMPLETED[system]:
            raise AdaptivePilotError(f"unexpected completed count for {system}")
        if any(context.run.get("system") != system for context in contexts):
            raise AdaptivePilotError(f"context system mismatch for {system}")
        if not all(
            type(context.run.get("repeat_index")) is int for context in contexts
        ) or [context.run.get("repeat_index") for context in contexts] != list(
            range(1, len(contexts) + 1)
        ):
            raise AdaptivePilotError(f"repetition sequence is invalid for {system}")
    return (
        next(iter(plan_ids)),
        next(iter(plan_hashes)),
        next(iter(preparation_hashes)),
        next(iter(plan_paths)),
        next(iter(preparation_paths)),
        next(iter(scopes)),
        episode_ids,
    )


def _formal_repeat_rule(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(records) < 3:
        return {
            "status": "NOT_EVALUABLE_FEWER_THAN_THREE_REPETITIONS",
            "threshold_cv": 0.03,
            "additional_repetitions_required": None,
            "candidate_cv_first_three": None,
            "candidates_above_threshold": None,
            "candidate_cv_universe": None,
            "frontier_eligible_candidates": None,
            "inconsistent_frontier_eligibility_candidates": None,
            "incomplete_candidates": None,
        }
    kernel_latencies, kernel_decisions = _candidate_series(records[:3], "kernel")
    cache_latencies, cache_decisions = _candidate_series(records[:3], "cache")
    latencies = {**kernel_latencies, **cache_latencies}
    decisions = {**kernel_decisions, **cache_decisions}
    def canonical_frontier_eligible(row: Mapping[str, Any]) -> bool:
        return (
            row.get("component") == "kernel"
            and row.get("outcome") == "exact_validated"
        ) or (
            row.get("component") == "cache"
            and row.get("outcome") == "quality_pass"
        )

    inconsistent_eligibility = sorted(
        candidate
        for candidate, candidate_rows in decisions.items()
        if any(
            row.get("frontier_eligible")
            is not canonical_frontier_eligible(row)
            for row in candidate_rows
        )
    )
    frontier_eligible = sorted(
        candidate
        for candidate, candidate_rows in decisions.items()
        if len(candidate_rows) == 3
        and all(canonical_frontier_eligible(row) for row in candidate_rows)
    )
    cv_universe = sorted(
        candidate
        for candidate, values in latencies.items()
        if len(values) == 3
    )
    incomplete = sorted(
        {
            candidate
            for candidate in frontier_eligible
            if len(latencies.get(candidate, [])) != 3
        }
    )
    if inconsistent_eligibility or incomplete:
        return {
            "status": (
                "NOT_EVALUABLE_INCONSISTENT_FRONTIER_ELIGIBILITY"
                if inconsistent_eligibility
                else "NOT_EVALUABLE_INCOMPLETE_LATENCIES"
            ),
            "threshold_cv": 0.03,
            "additional_repetitions_required": None,
            "candidate_cv_first_three": None,
            "candidates_above_threshold": None,
            "candidate_cv_universe": cv_universe,
            "frontier_eligible_candidates": frontier_eligible,
            "inconsistent_frontier_eligibility_candidates": (
                inconsistent_eligibility
            ),
            "incomplete_candidates": incomplete,
        }
    rows = {
        candidate: _cv(latencies[candidate]) for candidate in cv_universe
    }
    unstable = sorted(candidate for candidate, value in rows.items() if value > 0.03)
    return {
        "status": (
            "NEEDS_TWO_ADDITIONAL_REPETITIONS" if unstable else "FIRST_THREE_STABLE"
        ),
        "threshold_cv": 0.03,
        "additional_repetitions_required": bool(unstable),
        "candidate_cv_first_three": rows,
        "candidates_above_threshold": unstable,
        "candidate_cv_universe": cv_universe,
        "frontier_eligible_candidates": frontier_eligible,
        "inconsistent_frontier_eligibility_candidates": [],
        "incomplete_candidates": [],
    }


def _system_summary(
    system: str,
    records: Sequence[Mapping[str, Any]],
    quality_protocol: Mapping[str, Any],
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ttvf = [float(record["ttvf_s"]) for record in records]
    semantics = [semantic_decisions(record["decisions"]) for record in records]
    signatures = [_object_sha256(value) for value in semantics]
    frontiers = [dict(record["frontier"]["frontier"]) for record in records]
    kernel_latencies, kernel_decisions = _candidate_series(records, "kernel")
    cache_latencies, cache_decisions = _candidate_series(records, "cache")
    kernel = _kernel_frontier(len(records), kernel_latencies, kernel_decisions)
    cache = _cache_frontier(
        quality_protocol,
        len(records),
        cache_latencies,
        cache_decisions,
        full_scope=False,
    )
    relative_range = _relative_range(ttvf)
    decision_agreement = len(set(signatures)) == 1
    frontier_agreement = all(frontier == frontiers[0] for frontier in frontiers)
    adaptive_gate_applicable = system in {"optroll1", "optroll2"}
    stop_after_two = (
        adaptive_gate_applicable
        and len(records) == 2
        and decision_agreement
        and frontier_agreement
        and relative_range <= threshold
    )
    total_busy = sum(float(record["gpu_busy_s"]) for record in records)
    total_capacity = sum(float(record["gpu_capacity_s"]) for record in records)
    ranking_latencies = [
        float(decision["ranking_latency_s"])
        for record in records
        for decision in record["decisions"].values()
        if isinstance(decision.get("ranking_latency_s"), (int, float))
        and not isinstance(decision.get("ranking_latency_s"), bool)
        and math.isfinite(float(decision["ranking_latency_s"]))
        and float(decision["ranking_latency_s"]) > 0
    ]
    compact_runs = [
        {
            "run_id": record["run_id"],
            "repeat_index": record["repeat_index"],
            "ttvf_s": record["ttvf_s"],
            "ttvf_clock": record["ttvf_clock"],
            "frontier": record["frontier"]["frontier"],
            "decision_semantics_sha256": signature,
            "ledger_receipt": record["ledger_receipt"],
            "gpu_busy_s": record["gpu_busy_s"],
            "gpu_capacity_s": record["gpu_capacity_s"],
            "gpu_queue_idle_s": record["gpu_queue_idle_s"],
            "scheduler_gpu_utilization": record["scheduler_gpu_utilization"],
            "quality_wall_s": record["quality_wall_s"],
            "measured_generation_s": record["measured_generation_s"],
            "model_load_compile_warmup_s": record[
                "model_load_compile_warmup_s"
            ],
        }
        for record, signature in zip(records, signatures, strict=True)
    ]
    return (
        {
            "completed_repetitions": len(records),
            "ttvf_s": {
                "samples": ttvf,
                "median": statistics.median(ttvf),
                "mean": statistics.fmean(ttvf),
                "minimum": min(ttvf),
                "maximum": max(ttvf),
                "relative_range": relative_range,
            },
            "semantic_candidate_decision_agreement": decision_agreement,
            "semantic_decision_signatures_sha256": signatures,
            "raw_frontiers": frontiers,
            "raw_frontier_agreement": frontier_agreement,
            "median_selected_frontier": {
                "kernel": kernel.get("winner"),
                "cache": cache.get("winner"),
            },
            "kernel_frontier": kernel,
            "cache_frontier": cache,
            "adaptive_two_repetition_gate": {
                "applicable": adaptive_gate_applicable,
                "threshold_relative_range": threshold,
                "decision_agreement_required": True,
                "raw_frontier_agreement_required": True,
                "threshold_met": (
                    relative_range <= threshold if adaptive_gate_applicable else None
                ),
                "stop_after_two": stop_after_two,
            },
            "formal_three_plus_two_rule": _formal_repeat_rule(records),
            "metrics": {
                "gpu_hours_total": total_busy / 3600.0,
                "gpu_hours_per_repetition": total_busy / len(records) / 3600.0,
                "scheduler_gpu_utilization": total_busy / total_capacity,
                "gpu_queue_idle_s_total": sum(
                    float(record["gpu_queue_idle_s"]) for record in records
                ),
                "quality_wall_s_total": sum(
                    float(record["quality_wall_s"]) for record in records
                ),
                "measured_generation_s_total": sum(
                    float(record["measured_generation_s"]) for record in records
                ),
                "model_load_compile_warmup_s_total": sum(
                    float(record["model_load_compile_warmup_s"])
                    for record in records
                ),
                "candidate_ranking_latency_p50_s": _percentile(
                    ranking_latencies, 0.50
                ),
                "candidate_ranking_latency_p95_s": _percentile(
                    ranking_latencies, 0.95
                ),
            },
            "runs": compact_runs,
        },
        semantics,
    )


def analyze_adaptive_pilot(
    contexts_by_system: Mapping[str, Sequence[RunContext]],
    state_root: Path | str,
    suite: Mapping[str, Any],
    quality_protocol: Mapping[str, Any],
    *,
    suite_path: Path | str,
    ttvf_relative_range_threshold: float = 0.05,
) -> dict[str, Any]:
    """Replay and summarize the explicitly non-formal 3/3/2/2 pilot."""

    if (
        not isinstance(ttvf_relative_range_threshold, (int, float))
        or isinstance(ttvf_relative_range_threshold, bool)
        or not math.isfinite(float(ttvf_relative_range_threshold))
        or not 0 < float(ttvf_relative_range_threshold) < 1
    ):
        raise AdaptivePilotError("TTVF relative-range threshold must be between 0 and 1")
    (
        plan_id,
        plan_sha256,
        preparation_sha256,
        plan_path,
        preparation_path,
        scope,
        episode_ids,
    ) = _validate_context_matrix(contexts_by_system)
    state_path = Path(state_root)
    if state_path.is_symlink():
        raise AdaptivePilotError("state root must not be a symlink")
    state_path = state_path.resolve()
    if not state_path.is_dir():
        raise AdaptivePilotError("state root is missing")
    plan_receipt = _file_receipt(plan_path, "experiment plan")
    preparation_receipt = _file_receipt(preparation_path, "preparation receipt")
    if plan_receipt["sha256"] != plan_sha256:
        raise AdaptivePilotError("experiment plan changed after context loading")
    if preparation_receipt["sha256"] != preparation_sha256:
        raise AdaptivePilotError("preparation changed after context loading")
    plan, verified_plan_receipt = _load_json_receipt(
        plan_path, "experiment plan"
    )
    if verified_plan_receipt != plan_receipt:
        raise AdaptivePilotError("experiment plan changed while it was read")
    suite_receipts, public_episode_ids = _suite_source_receipts(
        suite_path, suite, quality_protocol
    )
    _validate_plan_suite_binding(
        plan,
        plan_id=plan_id,
        scope=scope,
        episode_ids=episode_ids,
        suite=suite,
        quality_protocol=quality_protocol,
        suite_receipts=suite_receipts,
        public_episode_ids=public_episode_ids,
    )
    analysis_implementation = _analysis_implementation_receipt()

    systems: dict[str, Any] = {}
    all_semantics: list[dict[str, Any]] = []
    for system in _SYSTEM_ORDER:
        records = [
            _run_record(context, state_path, repair_ledger_tail=False)
            for context in contexts_by_system[system]
        ]
        for record in records:
            try:
                _verify_file_receipt(
                    record["ledger_receipt"], f"{record['run_id']} event ledger"
                )
            except AggregationError as exc:
                raise AdaptivePilotError(str(exc)) from exc
        summary, semantics = _system_summary(
            system,
            records,
            quality_protocol,
            float(ttvf_relative_range_threshold),
        )
        systems[system] = summary
        all_semantics.extend(semantics)

    baseline = float(systems["serial1"]["ttvf_s"]["median"])
    medians = {
        system: float(systems[system]["ttvf_s"]["median"])
        for system in _SYSTEM_ORDER
    }
    for system in _SYSTEM_ORDER:
        systems[system]["speedup_vs_serial1_median"] = baseline / medians[system]
    semantic_hashes = [_object_sha256(value) for value in all_semantics]
    semantic_agreement = len(set(semantic_hashes)) == 1
    median_frontiers = {
        system: systems[system]["median_selected_frontier"]
        for system in _SYSTEM_ORDER
    }
    median_frontier_agreement = len(
        {_object_sha256(value) for value in median_frontiers.values()}
    ) == 1
    optroll_stop = all(
        systems[system]["adaptive_two_repetition_gate"]["stop_after_two"]
        for system in ("optroll1", "optroll2")
    )
    common_semantics = all_semantics[0] if semantic_agreement else None
    formal_rules = {
        system: systems[system]["formal_three_plus_two_rule"]
        for system in _SYSTEM_ORDER
    }
    formal_policy_satisfied = all(
        row["status"] == "FIRST_THREE_STABLE" for row in formal_rules.values()
    )
    try:
        _verify_file_receipt(plan_receipt, "experiment plan")
        _verify_file_receipt(preparation_receipt, "preparation receipt")
        for name, receipt in suite_receipts.items():
            _verify_file_receipt(receipt, f"suite file {name}")
    except AggregationError as exc:
        raise AdaptivePilotError(str(exc)) from exc
    if _analysis_implementation_receipt() != analysis_implementation:
        raise AdaptivePilotError("analysis implementation changed during replay")
    return {
        "schema_version": 1,
        "record_type": "sol_rolloutbench_exploratory_adaptive_pilot",
        "status": (
            "EXPLORATORY_ADAPTIVE_PILOT_COMPLETE"
            if optroll_stop
            else "EXPLORATORY_ADAPTIVE_PILOT_NEEDS_OPTROLL_REPEAT_3"
        ),
        "claim_scope": "engineering_pilot_not_publication_grade",
        "performance_claim": False,
        "formal_compare_systems_compatible": False,
        "formal_policy_status": (
            "SATISFIED" if formal_policy_satisfied else "NOT_SATISFIED"
        ),
        "plan_id": plan_id,
        "plan_sha256": plan_sha256,
        "scope": scope,
        "episode_ids": episode_ids,
        "episode_count": len(episode_ids),
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
            "suite_files": suite_receipts,
            "analysis_implementation": analysis_implementation,
        },
        "adaptive_policy": {
            "completed_repetitions": dict(_EXPECTED_COMPLETED),
            "optroll_ttvf_relative_range_threshold": float(
                ttvf_relative_range_threshold
            ),
            "optroll_repeat_3_included": False,
            "optroll_stop_after_two": optroll_stop,
        },
        "semantic_candidate_decision_agreement_across_all_runs": semantic_agreement,
        "common_semantic_decisions": common_semantics,
        "median_selected_frontiers": median_frontiers,
        "median_selected_frontier_agreement_across_systems": median_frontier_agreement,
        "systems": systems,
        "comparison": {
            "baseline_system": "serial1",
            "ranking_by_median_ttvf": sorted(medians, key=medians.get),
            "median_ttvf_s": medians,
            "speedup_vs_serial1": {
                system: baseline / medians[system] for system in _SYSTEM_ORDER
            },
            "optroll2_speedup_vs_fifo2": medians["fifo2"] / medians["optroll2"],
        },
        "limitations": [
            "pilot subset contains 10 representative candidates, not all 35 episodes",
            "optroll1 and optroll2 stopped after two under a later exploratory rule",
            "formal candidate-CV status is reported per system and OptRoll has fewer than three repetitions",
            "raw single-run frontier agreement is reported separately from median selection",
            "the result cannot be passed to the formal compare-systems command",
            "no blinded visual review or fault injection was run",
        ],
    }


def write_adaptive_pilot_result(
    path: Path | str, result: Mapping[str, Any]
) -> dict[str, Any]:
    return _write_locked_json(
        path,
        result,
        conflict_message="refusing to overwrite a conflicting adaptive-pilot result",
    )
