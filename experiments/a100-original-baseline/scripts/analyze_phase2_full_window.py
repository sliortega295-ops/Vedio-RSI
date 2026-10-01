#!/usr/bin/env python3
"""Integrate the Phase 2 eight-A100 window using remote-host monotonic time.

This supersedes the phase-1 sample-count approximation for Phase 2 reports.
Intervals longer than 1.5 s or touching an invalid observer row are UNKNOWN,
not silently treated as 1 s of utilization. This is an offline analysis only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


NS = 1_000_000_000
MAX_VALID_STEP_NS = 1_500_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> tuple[list[dict], int]:
    rows = []
    malformed = 0
    with path.open() as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                malformed += 1
    return rows, malformed


def job_identity(row: dict) -> tuple:
    if row["event"].startswith("gpu_import_preflight_"):
        return ("import", row.get("output"), row.get("attempt"), row.get("pid"))
    return ("inference", row.get("run_dir"), row.get("prompt_index"), row.get("pid"))


def overlap_ns(a: int, b: int, c: int, d: int) -> int:
    return max(0, min(b, d) - max(a, c))


def merged_intervals(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for lo, hi in sorted(pairs):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(hi, out[-1][1]))
        else:
            out.append((lo, hi))
    return out


def jobs_and_allocation(events: list[dict], begin: int, end: int,
                        uuids: list[str]) -> tuple[list[dict], dict, dict]:
    start_types = {"gpu_job_start", "gpu_import_preflight_job_start"}
    end_types = {"gpu_job_end", "gpu_import_preflight_job_end"}
    starts: dict[tuple, dict] = {}
    ends: dict[tuple, dict] = {}
    for row in events:
        event = row.get("event")
        if event not in start_types | end_types:
            continue
        if not isinstance(row.get("monotonic_ns"), int) or not isinstance(row.get("pid"), int):
            raise ValueError("GPU event is missing monotonic timestamp or PID")
        key = job_identity(row)
        if key[0] == "inference" and (not isinstance(key[1], str)
                                           or not isinstance(key[2], int)):
            raise ValueError("inference GPU event lacks run/prompt identity")
        target = starts if event in start_types else ends
        if key in target:
            raise ValueError(f"duplicate GPU event identity: {event} {key}")
        target[key] = row
    orphan = set(ends) - set(starts)
    if orphan:
        raise ValueError(f"orphan GPU end event identity: {sorted(orphan)[:3]}")
    jobs = []
    by_uuid: dict[str, list[tuple[int, int]]] = {uuid: [] for uuid in uuids}
    changes: dict[int, list[tuple[str, int]]] = defaultdict(list)
    changes[begin]
    changes[end]
    for key, start in starts.items():
        finish = ends.get(key)
        if finish and int(finish["monotonic_ns"]) <= int(start["monotonic_ns"]):
            raise ValueError(f"GPU end precedes start: {key}")
        if finish and finish.get("gpu_uuid") not in (None, start.get("gpu_uuid")):
            raise ValueError(f"GPU start/end UUID differs: {key}")
        lo = max(begin, int(start["monotonic_ns"]))
        hi = min(end, int(finish["monotonic_ns"]) if finish else end)
        if hi <= lo:
            continue
        uuid = start["gpu_uuid"]
        if uuid not in by_uuid:
            raise ValueError(f"event assigned unapproved GPU UUID: {uuid}")
        by_uuid[uuid].append((lo, hi))
        changes[lo].append((uuid, +1))
        changes[hi].append((uuid, -1))
        jobs.append({"kind": key[0], "run_dir": start.get("run_dir"),
                     "prompt_index": start.get("prompt_index"), "pid": start.get("pid"),
                     "gpu_uuid": uuid, "start_monotonic_ns": lo,
                     "end_monotonic_ns": hi, "process_gpu_s": (hi - lo) / NS,
                     "end_observed": finish is not None and int(finish["monotonic_ns"]) <= end})
    by_uuid = {uuid: merged_intervals(pairs) for uuid, pairs in by_uuid.items()}
    active_counts = defaultdict(int)
    concurrent_ns = defaultdict(int)
    allocated_ns = defaultdict(int)
    zero_windows = []
    previous = None
    for when in sorted(changes):
        if previous is not None and when > previous:
            duration = when - previous
            active = [uuid for uuid in uuids if active_counts[uuid] > 0]
            concurrent_ns[len(active)] += duration
            for uuid in active:
                allocated_ns[uuid] += duration
            if not active and duration >= 30 * NS:
                zero_windows.append({"start_monotonic_ns": previous,
                                     "end_monotonic_ns": when,
                                     "duration_s": duration / NS})
        for uuid, delta in changes[when]:
            active_counts[uuid] += delta
            if active_counts[uuid] < 0:
                raise ValueError("unpaired GPU process end event")
        previous = when
    if any(active_counts.values()):
        raise ValueError("allocation sweep did not close")
    return jobs, by_uuid, {"concurrent_s": {str(n): value / NS for n, value in sorted(concurrent_ns.items())},
                           "allocated_s": {uuid: allocated_ns[uuid] / NS for uuid in uuids},
                           "zero_windows": zero_windows}


def integrate_samples(samples: list[dict], begin: int, end: int,
                      uuids: list[str], allocations: dict) -> dict:
    samples = sorted((row for row in samples if isinstance(row.get("monotonic_ns"), int)),
                     key=lambda row: row["monotonic_ns"])
    if len(samples) < 2 or samples[0]["monotonic_ns"] > begin or samples[-1]["monotonic_ns"] < end:
        raise ValueError("observer does not bracket the exact accounting window")
    per_gpu = {uuid: {"uuid": uuid, "index": index,
                      "util_equivalent_s_estimate": 0.0,
                      "allocated_util_equivalent_s_estimate": 0.0,
                      "memory_gt100_mib_s_estimate": 0.0,
                      "app_present_s_estimate": 0.0,
                      "util_peak_pct_sampled": 0.0,
                      "memory_peak_mib_sampled": 0.0,
                      "app_pids_seen": set()}
               for index, uuid in enumerate(uuids)}
    covered_ns = long_or_invalid_ns = 0
    long_gap_count = invalid_sample_count = incomplete_count = sample_count = 0
    max_step_ns = 0
    observer_overhead_s = 0.0
    overhead_count = 0
    for left, right in zip(samples, samples[1:]):
        lo = max(begin, left["monotonic_ns"])
        hi = min(end, right["monotonic_ns"])
        if hi <= lo:
            continue
        sample_count += 1
        step = right["monotonic_ns"] - left["monotonic_ns"]
        max_step_ns = max(max_step_ns, step)
        left_uuids = [row.get("uuid") for row in left.get("gpus", [])]
        right_uuids = [row.get("uuid") for row in right.get("gpus", [])]
        invalid = "error" in left or "error" in right
        incomplete = left_uuids != uuids or right_uuids != uuids
        invalid_sample_count += int(invalid)
        incomplete_count += int(incomplete)
        long_gap_count += int(step > MAX_VALID_STEP_NS)
        if invalid or incomplete or step > MAX_VALID_STEP_NS:
            long_or_invalid_ns += hi - lo
            continue
        covered_ns += hi - lo
        if isinstance(left.get("sample_overhead_s"), (int, float)):
            observer_overhead_s += left["sample_overhead_s"]
            overhead_count += 1
        app_by_uuid: dict[str, set[int]] = defaultdict(set)
        for app in left.get("gpu_apps", []):
            if app.get("gpu_uuid") in per_gpu and isinstance(app.get("pid"), int):
                app_by_uuid[app["gpu_uuid"]].add(app["pid"])
        duration_s = (hi - lo) / NS
        for gpu in left["gpus"]:
            uuid = gpu["uuid"]
            item = per_gpu[uuid]
            util = float(gpu["gpu_util_pct"])
            memory = float(gpu["memory_used_mib"])
            item["util_equivalent_s_estimate"] += duration_s * util / 100
            item["memory_gt100_mib_s_estimate"] += duration_s if memory > 100 else 0
            item["app_present_s_estimate"] += duration_s if app_by_uuid[uuid] else 0
            item["util_peak_pct_sampled"] = max(item["util_peak_pct_sampled"], util)
            item["memory_peak_mib_sampled"] = max(item["memory_peak_mib_sampled"], memory)
            item["app_pids_seen"].update(app_by_uuid[uuid])
            assigned_ns = sum(overlap_ns(lo, hi, a, b) for a, b in allocations[uuid])
            item["allocated_util_equivalent_s_estimate"] += assigned_ns / NS * util / 100
    for item in per_gpu.values():
        item["app_pids_seen"] = sorted(item["app_pids_seen"])
    return {"covered_s": covered_ns / NS,
            "unobserved_s": (end - begin - covered_ns) / NS,
            "long_or_invalid_s": long_or_invalid_ns / NS,
            "sample_intervals": sample_count,
            "long_gap_count": long_gap_count,
            "invalid_interval_count": invalid_sample_count,
            "incomplete_uuid_interval_count": incomplete_count,
            "max_sample_start_interval_s": max_step_ns / NS,
            "observer_overhead_mean_s": observer_overhead_s / overhead_count if overhead_count else None,
            "observer_overhead_count": overhead_count,
            "per_gpu_uuid": per_gpu}


def analyze(budget: dict, observer: dict, events: list[dict], samples: list[dict],
            *, cap_at_budget: bool) -> dict:
    uuids = observer["gpu_uuids"]
    if len(uuids) != 8 or len(set(uuids)) != 8:
        raise ValueError("observer UUID identity is not the eight assigned GPUs")
    begin = int(budget["first_gpu_monotonic_ns"])
    sampled = [row["monotonic_ns"] for row in samples if isinstance(row.get("monotonic_ns"), int)]
    if not sampled:
        raise ValueError("observer has no monotonic samples")
    end = begin + int(budget["limit_s"] * NS) if cap_at_budget else max(sampled)
    if end <= begin:
        raise ValueError("empty accounting window")
    jobs, allocations, sweep = jobs_and_allocation(events, begin, end, uuids)
    observed = integrate_samples(samples, begin, end, uuids, allocations)
    for uuid in uuids:
        observed["per_gpu_uuid"][uuid]["allocated_gpu_s"] = sweep["allocated_s"][uuid]
    wall_s = (end - begin) / NS
    capacity_s = wall_s * len(uuids)
    allocated_s = sum(sweep["allocated_s"].values())
    util_s = sum(item["util_equivalent_s_estimate"] for item in observed["per_gpu_uuid"].values())
    unobserved_gpu_capacity_s = len(uuids) * observed["unobserved_s"]
    return {"schema_version": 1, "phase": "phase2",
            "measurement": "remote_monotonic_process_windows_and_interval_weighted_1Hz_NVML",
            "first_gpu_utc": budget["first_gpu_utc"],
            "first_gpu_monotonic_ns": begin, "accounting_end_monotonic_ns": end,
            "capped_at_exact_budget_deadline": cap_at_budget,
            "wall_s": wall_s, "eight_gpu_capacity_s": capacity_s,
            "allocated_distinct_uuid_gpu_s": allocated_s,
            "allocated_fraction_of_capacity": allocated_s / capacity_s,
            "unallocated_gpu_capacity_s": capacity_s - allocated_s,
            "observed_interval_util_equivalent_gpu_s_estimate": util_s,
            "unobserved_gpu_capacity_s": unobserved_gpu_capacity_s,
            "interval_hold_util_equivalent_gpu_s_lower_bound_estimate": util_s,
            "interval_hold_util_equivalent_gpu_s_upper_bound_estimate": util_s + unobserved_gpu_capacity_s,
            "interval_hold_util_fraction_lower_bound_estimate": util_s / capacity_s,
            "interval_hold_util_fraction_upper_bound_estimate":
                (util_s + unobserved_gpu_capacity_s) / capacity_s,
            "observed_fraction_of_wall": observed["covered_s"] / wall_s,
            "seconds_by_concurrent_distinct_allocated_uuids": sweep["concurrent_s"],
            "zero_allocated_intervals_at_least_30s": sweep["zero_windows"],
            "job_intervals": jobs, "open_job_count_at_snapshot": sum(not row["end_observed"] for row in jobs),
            "observer": {key: value for key, value in observed.items() if key != "per_gpu_uuid"},
            "per_gpu_uuid": observed["per_gpu_uuid"],
            "limitations": [
                "NVML utilization is a sampled device estimate, not exact SM/kernel busy time.",
                "Intervals above 1.5 s or touching invalid samples are unobserved, not imputed; reported lower/upper utilization is an interval-hold estimate range for those missing spans, not a physical SM-time bound.",
                "Allocated means an observed site runner GPU child process window, not actual GPU computation.",
                "Unallocated capacity is not automatically avoidable scheduler idle; Agent dependencies and gates require separate receipts.",
                "No local-host monotonic timestamp is subtracted from remote-host monotonic time.",
            ]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--observer-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cap-at-budget", action="store_true")
    args = parser.parse_args()
    budget = json.loads(args.budget.read_text())
    observer = json.loads(args.observer_state.read_text())
    events, malformed_events = read_jsonl(args.events)
    samples, malformed_samples = read_jsonl(args.samples)
    if malformed_events:
        raise ValueError(f"GPU event JSONL has {malformed_events} malformed rows")
    if malformed_samples:
        raise ValueError(f"observer JSONL has {malformed_samples} malformed rows")
    result = analyze(budget, observer, events, samples, cap_at_budget=args.cap_at_budget)
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["source"] = {"budget": str(args.budget), "budget_sha256": sha256(args.budget),
                        "events": str(args.events), "events_sha256": sha256(args.events),
                        "malformed_event_rows": malformed_events,
                        "samples": str(args.samples), "samples_sha256": sha256(args.samples),
                        "observer_state": str(args.observer_state),
                        "observer_state_sha256": sha256(args.observer_state)}
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError("refusing to overwrite an existing telemetry analysis")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"wall_s": result["wall_s"],
                      "covered_s": result["observer"]["covered_s"],
                      "allocated_gpu_s": result["allocated_distinct_uuid_gpu_s"],
                      "observed_util_equivalent_gpu_s_estimate": result["observed_interval_util_equivalent_gpu_s_estimate"],
                      "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
