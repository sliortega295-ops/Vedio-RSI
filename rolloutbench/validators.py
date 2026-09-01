from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from .quality_contract import (
    FORMAL_CACHE_IDS,
    QUALITY_DIMENSIONS,
    QUALITY_METRICS_BY_SUITE,
    QUALITY_SEEDS,
)


_LATENCY_SELECTION_BASIS = (
    "fresh_repetition_median_ranking_latency_s_one_shot_process_wall"
)


def _formal_candidates(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    candidates = protocol.get("formal_cache_candidates")
    if not isinstance(candidates, list) or tuple(candidates) != FORMAL_CACHE_IDS:
        raise ValueError("quality protocol must declare exactly nine formal cache candidates")
    normalized = tuple(str(item) for item in candidates)
    if len(set(normalized)) != 9:
        raise ValueError("formal cache candidate IDs must be unique")
    return normalized


def build_quality_plan(
    protocol: Mapping[str, Any], candidate_ids: Iterable[str]
) -> list[dict[str, Any]]:
    """Expand the frozen 4-prompt x 2-seed protocol for formal candidates only."""

    formal = _formal_candidates(protocol)
    requested = tuple(str(item) for item in candidate_ids)
    if len(requested) != len(set(requested)):
        raise ValueError("candidate_ids contains duplicates")
    invalid = sorted(set(requested) - set(formal))
    if invalid:
        raise ValueError(f"quality plan accepts formal cache candidates only: {invalid}")

    prompt_suites = protocol.get("prompt_selection", {}).get("prompt_suites")
    seeds = protocol.get("seeds")
    if not isinstance(prompt_suites, list) or len(prompt_suites) != len(QUALITY_METRICS_BY_SUITE):
        raise ValueError("quality plan requires the frozen four prompt suites")
    if not isinstance(seeds, list) or tuple(seeds) != QUALITY_SEEDS:
        raise ValueError("quality plan requires the frozen two seeds")

    required = (
        "suite",
        "prompt",
        "source_path",
        "source_sha256",
        "selection_sha256",
        "selected_line_number_one_based",
        "metrics",
    )
    if requested:
        for prompt_spec in prompt_suites:
            missing = [field for field in required if field not in prompt_spec]
            if missing:
                raise ValueError(f"quality prompt specification is incomplete: {missing}")

    requested_set = set(requested)
    rows: list[dict[str, Any]] = []
    for candidate_id in formal:
        if candidate_id not in requested_set:
            continue
        for prompt_spec in prompt_suites:
            suite = str(prompt_spec["suite"])
            for seed_value in seeds:
                seed = int(seed_value)
                pair_id = f"{candidate_id}:{suite}:seed-{seed}"
                artifact_suffix = f"{suite}/seed-{seed}"
                rows.append(
                    {
                        "pair_id": pair_id,
                        "candidate_id": candidate_id,
                        "prompt_suite": suite,
                        "prompt": str(prompt_spec["prompt"]),
                        "source_path": str(prompt_spec["source_path"]),
                        "source_sha256": str(prompt_spec["source_sha256"]),
                        "selection_sha256": str(prompt_spec["selection_sha256"]),
                        "selected_line_number_one_based": int(
                            prompt_spec["selected_line_number_one_based"]
                        ),
                        "seed": seed,
                        "metrics": list(prompt_spec["metrics"]),
                        "dense_artifact_id": f"dense/quality_v1/{artifact_suffix}",
                        "candidate_artifact_id": (
                            f"candidate/{candidate_id}/quality_v1/{artifact_suffix}"
                        ),
                    }
                )
    expected = len(requested) * 8
    if len(rows) != expected:
        raise ValueError(f"quality plan expansion produced {len(rows)} rows, expected {expected}")
    return rows


def _lpips_summary(
    plan: Sequence[Mapping[str, Any]],
    values: Mapping[str, Sequence[float]] | None,
    *,
    expected_frames: int,
) -> dict[str, Any]:
    expected_pairs = {str(row["pair_id"]) for row in plan}
    if values is None:
        return {
            "status": "MISSING",
            "role": "secondary_ranking_only",
            "hard_acceptance_effect": False,
            "value_count": 0,
            "mean": None,
        }
    actual_pairs = set(values)
    flattened: list[float] = []
    invalid: list[str] = []
    for pair_id in sorted(expected_pairs):
        pair_values = values.get(pair_id)
        if pair_values is None:
            invalid.append(f"missing pair {pair_id}")
            continue
        if not isinstance(pair_values, Sequence) or isinstance(pair_values, (str, bytes)):
            invalid.append(f"pair {pair_id} LPIPS values are not a sequence")
            continue
        if len(pair_values) != expected_frames:
            invalid.append(
                f"pair {pair_id} has {len(pair_values)} frames, expected {expected_frames}"
            )
            continue
        for value in pair_values:
            try:
                number = float(value)
            except (TypeError, ValueError):
                invalid.append(f"pair {pair_id} contains invalid LPIPS")
                break
            if not math.isfinite(number):
                invalid.append(f"pair {pair_id} contains non-finite LPIPS")
                break
            flattened.append(number)
    if actual_pairs - expected_pairs:
        invalid.append(f"unexpected pairs: {sorted(actual_pairs - expected_pairs)}")
    status = "COMPLETE" if not invalid else "INVALID"
    return {
        "status": status,
        "role": "secondary_ranking_only",
        "hard_acceptance_effect": False,
        "value_count": len(flattened),
        "mean": statistics.fmean(flattened) if status == "COMPLETE" else None,
        "errors": invalid,
    }


def evaluate_quality_candidate(
    protocol: Mapping[str, Any],
    plan: Sequence[Mapping[str, Any]],
    score_rows: Iterable[Mapping[str, Any]],
    *,
    lpips_frame_values: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Evaluate the seven higher-is-better dimensions with fail-closed inputs."""

    protocol_id = str(protocol.get("protocol_id"))
    formal = set(_formal_candidates(protocol))
    candidate_ids = {str(row.get("candidate_id")) for row in plan}
    candidate_id = next(iter(candidate_ids)) if len(candidate_ids) == 1 else None
    errors: list[str] = []
    if candidate_id is None:
        errors.append("plan must contain exactly one candidate")
    elif candidate_id not in formal:
        errors.append(f"candidate {candidate_id} is not formal")
    if len(plan) != 8:
        errors.append(f"plan must contain exactly 8 matched pairs, got {len(plan)}")

    expected: dict[tuple[str, str], Mapping[str, Any]] = {}
    for pair in plan:
        pair_id = str(pair.get("pair_id"))
        for metric_value in pair.get("metrics", []):
            metric = str(metric_value)
            key = (pair_id, metric)
            if key in expected:
                errors.append(f"duplicate expected pair/metric {pair_id}/{metric}")
            expected[key] = pair

    scores: dict[tuple[str, str], tuple[float, float]] = {}
    for row in score_rows:
        pair_id = str(row.get("pair_id"))
        metric = str(row.get("metric"))
        key = (pair_id, metric)
        if key not in expected:
            errors.append(f"unexpected pair/metric {pair_id}/{metric}")
            continue
        if key in scores:
            errors.append(f"duplicate score pair/metric {pair_id}/{metric}")
            continue
        try:
            dense = float(row["dense_score"])
            candidate = float(row["candidate_score"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"invalid score for pair/metric {pair_id}/{metric}")
            continue
        if not math.isfinite(dense) or not math.isfinite(candidate):
            errors.append(
                f"nan or non-finite score for pair/metric {pair_id}/{metric}"
            )
            continue
        if dense <= 0:
            errors.append(f"zero or negative dense score for pair/metric {pair_id}/{metric}")
            continue
        scores[key] = (dense, candidate)

    missing = sorted(set(expected) - set(scores))
    errors.extend(f"missing score for pair/metric {pair}/{metric}" for pair, metric in missing)

    dimensions = tuple(str(item) for item in protocol.get("dimensions", []))
    if dimensions != QUALITY_DIMENSIONS:
        errors.append("protocol must declare seven unique dimensions")
    if {metric for _, metric in expected} != set(dimensions):
        errors.append("plan metrics do not cover exactly the seven protocol dimensions")
    by_metric: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (_, metric), values in scores.items():
        by_metric[metric].append(values)
    for metric in dimensions:
        if not by_metric.get(metric):
            errors.append(f"missing metric {metric}")

    lpips = _lpips_summary(
        plan,
        lpips_frame_values,
        expected_frames=int(protocol.get("lpips", {}).get("frames", 81)),
    )
    if errors:
        return {
            "protocol_id": protocol_id,
            "candidate_id": candidate_id,
            "eligibility": "formal" if candidate_id in formal else "ineligible",
            "status": "FAIL_CLOSED",
            "pass": False,
            "errors": sorted(set(errors)),
            "dimensions": {},
            "mean_relative_drop": None,
            "max_relative_drop": None,
            "lpips": lpips,
        }

    dimension_results: dict[str, dict[str, Any]] = {}
    drops: list[float] = []
    for metric in dimensions:
        values = by_metric[metric]
        dense_mean = statistics.fmean(dense for dense, _ in values)
        candidate_mean = statistics.fmean(candidate for _, candidate in values)
        relative_drop = (dense_mean - candidate_mean) / abs(dense_mean)
        drops.append(relative_drop)
        dimension_results[metric] = {
            "matched_pair_count": len(values),
            "dense_mean": dense_mean,
            "candidate_mean": candidate_mean,
            "relative_drop": relative_drop,
        }

    mean_drop = statistics.fmean(drops)
    max_drop = max(drops)
    acceptance = protocol.get("acceptance", {})
    mean_limit = float(acceptance["max_mean_relative_drop"])
    single_limit = float(acceptance["max_single_dimension_drop"])
    passed = mean_drop <= mean_limit + 1e-12 and max_drop <= single_limit + 1e-12
    return {
        "protocol_id": protocol_id,
        "candidate_id": candidate_id,
        "eligibility": "formal",
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "errors": [],
        "dimensions": dimension_results,
        "mean_relative_drop": mean_drop,
        "max_relative_drop": max_drop,
        "thresholds": {
            "max_mean_relative_drop": mean_limit,
            "max_single_dimension_drop": single_limit,
        },
        "lpips": lpips,
    }


def select_quality_frontier(
    protocol: Mapping[str, Any],
    quality_results: Iterable[Mapping[str, Any]],
    measured_latencies: Mapping[str, float | Sequence[float]],
) -> dict[str, Any]:
    """Select the fastest newly measured passing formal candidate."""

    results = list(quality_results)
    if not results:
        return {
            "status": "NOT_RUN",
            "winner": None,
            "median_latency": None,
            "selection_basis": _LATENCY_SELECTION_BASIS,
        }
    formal = set(_formal_candidates(protocol))
    protocol_id = str(protocol.get("protocol_id"))
    seen: dict[str, int] = defaultdict(int)
    errors: list[str] = []
    for result in results:
        candidate_id = str(result.get("candidate_id"))
        seen[candidate_id] += 1
        if candidate_id not in formal:
            errors.append(f"unexpected candidate result {candidate_id}")
        if result.get("eligibility") != "formal":
            errors.append(f"candidate {candidate_id} is not marked formal")
        if result.get("protocol_id") != protocol_id:
            errors.append(f"candidate {candidate_id} protocol mismatch")
        status = result.get("status")
        if status not in {"PASS", "FAIL"}:
            errors.append(f"candidate {candidate_id} has incomplete status {status}")
        elif result.get("pass") is not (status == "PASS"):
            errors.append(f"candidate {candidate_id} status/pass mismatch")
    errors.extend(
        f"duplicate result for {candidate_id}"
        for candidate_id, count in sorted(seen.items())
        if count > 1
    )
    missing = sorted(formal - set(seen))
    if missing:
        errors.append(f"missing formal results: {missing}")
    if errors:
        return {
            "status": "FAIL_CLOSED",
            "winner": None,
            "median_latency": None,
            "selection_basis": _LATENCY_SELECTION_BASIS,
            "errors": errors,
        }

    ranked: list[tuple[float, str]] = []
    rejected_latencies: list[str] = []
    for result in results:
        candidate_id = str(result.get("candidate_id"))
        if (
            candidate_id not in formal
            or result.get("eligibility") != "formal"
            or result.get("status") != "PASS"
        ):
            continue
        raw = measured_latencies.get(candidate_id)
        values = [raw] if isinstance(raw, (int, float)) else list(raw or [])
        try:
            numbers = [float(value) for value in values]
        except (TypeError, ValueError):
            numbers = []
        if len(numbers) < 3 or any(
            not math.isfinite(value) or value <= 0 for value in numbers
        ):
            rejected_latencies.append(candidate_id)
            continue
        ranked.append((statistics.median(numbers), candidate_id))
    if rejected_latencies:
        return {
            "status": "FAIL_CLOSED",
            "winner": None,
            "median_latency": None,
            "selection_basis": _LATENCY_SELECTION_BASIS,
            "errors": [
                "passing candidates require at least three finite positive latency repetitions: "
                f"{sorted(rejected_latencies)}"
            ],
        }
    if not ranked:
        return {
            "status": "NO_PASSING_CANDIDATE",
            "winner": None,
            "median_latency": None,
            "selection_basis": _LATENCY_SELECTION_BASIS,
        }
    median_latency, winner = min(ranked, key=lambda item: (item[0], item[1]))
    return {
        "status": "SELECTED",
        "winner": winner,
        "median_latency": median_latency,
        "selection_basis": _LATENCY_SELECTION_BASIS,
        "eligible_ranked": [
            {"candidate_id": candidate, "median_latency": latency}
            for latency, candidate in sorted(ranked)
        ],
    }


def validate_kernel_structure(
    workload: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    expected = {
        "denoising_steps": workload.get("denoising_steps"),
        "cfg_branches": workload.get("cfg_branches_per_step"),
        "logical_dit_calls": workload.get("logical_dit_calls"),
        "transformer_blocks": workload.get("transformer_blocks_per_call"),
        "skipped_operations": 0,
    }
    observed = {field: receipt.get(field) for field in expected}
    errors = [
        f"workload contract is invalid for {field}"
        for field in expected
        if field != "skipped_operations"
        and (not isinstance(expected[field], int) or isinstance(expected[field], bool) or expected[field] <= 0)
    ]
    errors.extend(
        f"{field}: expected {value}, observed {observed[field]}"
        for field, value in expected.items()
        if observed[field] != value
    )
    return {
        "validator": "kernel_exact_structure_v1",
        "pass": not errors,
        "expected": expected,
        "observed": observed,
        "errors": errors,
    }


def validate_cache_receipt(
    episode: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    formal = episode.get("quality_eligibility") == "formal"
    required = ("generate", "collect", "quality_v1", "decide")
    errors: list[str] = []
    if not formal:
        errors.append("cache receipt is not a formal quality candidate")
    if receipt.get("episode_id") != episode.get("episode_id"):
        errors.append("receipt episode_id mismatch")
    completed = set(receipt.get("completed_stages", []))
    stage_receipts = receipt.get("stage_receipts", {})
    if not isinstance(stage_receipts, Mapping):
        stage_receipts = {}
        errors.append("stage_receipts must be an object")
    for stage in required:
        if stage not in completed:
            errors.append(f"missing completed stage {stage}")
        stage_receipt = stage_receipts.get(stage)
        if not isinstance(stage_receipt, Mapping):
            errors.append(f"missing receipt for stage {stage}")
            continue
        digest = stage_receipt.get("output_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
        ):
            errors.append(f"invalid output digest for stage {stage}")
    return {
        "validator": "cache_formal_receipt_v1",
        "episode_id": episode.get("episode_id"),
        "pass": not errors,
        "required_stages": list(required),
        "errors": errors,
    }


def _decision_outcome(row: Mapping[str, Any]) -> str | None:
    decision = row.get("decision")
    value = decision.get("outcome") if isinstance(decision, Mapping) else row.get("outcome")
    return str(value) if value is not None else None


def compare_historical_oracle(
    episodes: Sequence[Mapping[str, Any]],
    decisions: Iterable[Mapping[str, Any]],
    frontier: Mapping[str, str] | None,
    expected_frontier: Mapping[str, str],
) -> dict[str, Any]:
    """Acceptance-only comparison; the returned oracle must never drive scheduling."""

    expected: dict[str, str | None] = {}
    for episode in episodes:
        golden = episode.get("golden")
        if not isinstance(golden, Mapping) or golden.get("scheduler_visible") is not False:
            raise ValueError("historical oracle is absent or scheduler-visible")
        outcome = _decision_outcome(golden)
        if outcome is None:
            raise ValueError("historical oracle decision outcome is missing")
        expected[str(episode["episode_id"])] = outcome
    if len(expected) != 35:
        raise ValueError("historical acceptance requires exactly 35 unique episodes")

    actual: dict[str, str | None] = {}
    duplicates: list[str] = []
    for row in decisions:
        episode_id = str(row.get("episode_id"))
        if episode_id in actual:
            duplicates.append(episode_id)
        actual[episode_id] = _decision_outcome(row)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatched = {
        episode_id: {"expected": expected[episode_id], "actual": actual[episode_id]}
        for episode_id in sorted(set(expected) & set(actual))
        if expected[episode_id] != actual[episode_id]
    }
    matched = sum(
        expected[episode_id] == actual.get(episode_id)
        for episode_id in expected
        if episode_id in actual
    )
    episode_agrees = not (duplicates or missing or unexpected or mismatched) and matched == 35

    expected_frontier = dict(expected_frontier)
    if set(expected_frontier) != {"kernel", "cache"} or any(
        not isinstance(value, str) or not value for value in expected_frontier.values()
    ):
        raise ValueError("historical frontier contract is invalid")
    actual_frontier = dict(frontier or {})
    frontier_agrees = actual_frontier == expected_frontier
    status = "PASS" if episode_agrees and frontier_agrees else "FAIL"
    return {
        "status": status,
        "acceptance_only": True,
        "scheduler_feedback_allowed": False,
        "performance_claim": False,
        "episode_agreement": {
            "agrees": episode_agrees,
            "matched": matched,
            "expected": 35,
            "duplicates": sorted(set(duplicates)),
            "missing": missing,
            "unexpected": unexpected,
            "mismatched": mismatched,
        },
        "frontier_agreement": {
            "agrees": frontier_agrees,
            "expected": dict(expected_frontier),
            "actual": actual_frontier,
        },
    }


class HistoricalOracleReplay:
    """CPU acceptance oracle replay, explicitly invalid for performance simulation."""

    def replay(
        self,
        episodes: Sequence[Mapping[str, Any]],
        expected_frontier: Mapping[str, str],
    ) -> dict[str, Any]:
        decisions: list[dict[str, Any]] = []
        for episode in episodes:
            golden = episode.get("golden")
            if not isinstance(golden, Mapping) or golden.get("scheduler_visible") is not False:
                raise ValueError("historical oracle replay requires frozen hidden golden records")
            decisions.append(
                {
                    "episode_id": str(episode["episode_id"]),
                    "outcome": _decision_outcome(golden),
                }
            )
        if len(decisions) != 35:
            raise ValueError("historical oracle replay requires exactly 35 episodes")
        frontier = dict(expected_frontier)
        if set(frontier) != {"kernel", "cache"} or any(
            not isinstance(value, str) or not value for value in frontier.values()
        ):
            raise ValueError("historical frontier contract is invalid")
        return {
            "synthetic_historical_oracle_replay": True,
            "acceptance_only": True,
            "cannot_be_used_as_performance_simulation": True,
            "performance_claim": False,
            "decisions": decisions,
            "frontier": frontier,
        }
