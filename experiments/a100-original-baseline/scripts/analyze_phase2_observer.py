#!/usr/bin/env python3
"""Read-only, same-host-monotonic integration of one Phase 2 remote job window.

GPU figures are sample-and-hold estimates at ~1 Hz, not kernel/SM timing.
Intervals over 1.5 s or touching invalid samples are left unobserved.
Process CPU/I/O are sampled lower bounds for the exact worker PID lineage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path("/home/lyy/Experiments/SolAgent/20260913-a100-original-baseline")
sys.path.insert(0, str(PROJECT / "repo"))
from orchestration import site_remote  # noqa: E402


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_remote_window(start_ns: int, end_ns: int) -> tuple[list[dict], dict]:
    path = site_remote.REMOTE_PROJECT / "telemetry/phase2/observer/samples.jsonl"
    # This code reads only the project's own observer JSONL. Filtering on the
    # remote host keeps the large, still-growing raw stream in place.
    code = (
        "import json,sys; "
        "p,a,b=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]); "
        "f=open(p); "
        "[print(line,end='') for line in f if a-2000000000 <= "
        "json.loads(line)['monotonic_ns'] <= b+2000000000]"
    )
    result = site_remote.ssh("python3", "-c", code, str(path), str(start_ns),
                             str(end_ns), timeout=120)
    rows = [json.loads(line) for line in result.stdout.splitlines()
            if line.startswith("{")]
    stat = site_remote.ssh("stat", "-c", "%s %Y", str(path), timeout=20).stdout.strip()
    return rows, {"remote_samples": str(path), "remote_stat_size_mtime": stat,
                  "transport_selected_stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
                  "selected_row_count": len(rows), "read_at_utc": utc()}


def process_lineage(sample: dict, worker_pid: int) -> list[dict]:
    procs = {p["pid"]: p for p in sample.get("processes", [])}
    selected = {worker_pid}
    while True:
        old = len(selected)
        selected.update(pid for pid, proc in procs.items()
                        if proc.get("ppid") in selected)
        if len(selected) == old:
            break
    return [procs[pid] for pid in selected if pid in procs]


def analyze(rows: list[dict], start_ns: int, end_ns: int,
            worker_pid: int, ticks_per_s: int) -> dict:
    if not start_ns < end_ns:
        raise ValueError("invalid same-host monotonic job window")
    rows = sorted(rows, key=lambda row: row["monotonic_ns"])
    if len(rows) < 2:
        raise ValueError("observer has fewer than two boundary samples")
    uuids = site_remote.ssh(
        "python3", "-c",
        "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['gpu_uuids']))",
        str(site_remote.REMOTE_PROJECT / "telemetry/phase2/observer/state.json"),
        timeout=20).stdout.strip().split()
    if len(uuids) != 8 or len(set(uuids)) != 8:
        raise ValueError("observer state does not contain eight distinct UUIDs")
    gpu = {u: {"uuid": u, "index": i, "memory_gt100_mib_s_estimate": 0.0,
               "util_equivalent_s_estimate": 0.0, "nonzero_util_s_estimate": 0.0,
               "util_peak_pct_sampled": 0.0, "memory_peak_mib_sampled": 0.0,
               "power_w_s_sampled": 0.0, "app_pids_seen": set(),
               "app_present_s_estimate": 0.0} for i, u in enumerate(uuids)}
    covered_ns = gap_ns = 0
    error_rows = incomplete_rows = 0
    max_gap_s = 0.0
    first_mem_positive_ns = last_mem_positive_ns = None
    cpu_first_last: dict[tuple[int, int], list[dict]] = {}
    lineage_sample_count = 0
    rss_peak_bytes = 0
    for row in rows:
        if start_ns <= row["monotonic_ns"] < end_ns:
            error_rows += int("error" in row)
            incomplete_rows += int([g["uuid"] for g in row.get("gpus", [])] != uuids)
            lineage = process_lineage(row, worker_pid)
            lineage_sample_count += bool(lineage)
            rss_peak_bytes = max(rss_peak_bytes, sum(p.get("rss_bytes", 0) for p in lineage))
            for proc in lineage:
                ident = (proc["pid"], proc.get("start_ticks", -1))
                pair = cpu_first_last.setdefault(ident, [proc, proc])
                pair[1] = proc
    for left, right in zip(rows, rows[1:]):
        a = max(start_ns, left["monotonic_ns"])
        b = min(end_ns, right["monotonic_ns"])
        if b <= a:
            continue
        dt_ns = b - a
        step_s = (right["monotonic_ns"] - left["monotonic_ns"]) / 1e9
        max_gap_s = max(max_gap_s, step_s)
        if (step_s > 1.5 or "error" in left or "error" in right
                or [g["uuid"] for g in left.get("gpus", [])] != uuids
                or [g["uuid"] for g in right.get("gpus", [])] != uuids):
            gap_ns += dt_ns
            continue
        covered_ns += dt_ns
        dt = dt_ns / 1e9
        for device in left["gpus"]:
            item = gpu[device["uuid"]]
            util = float(device["gpu_util_pct"])
            mem = float(device["memory_used_mib"])
            item["memory_gt100_mib_s_estimate"] += dt if mem > 100 else 0.0
            item["util_equivalent_s_estimate"] += dt * util / 100.0
            item["nonzero_util_s_estimate"] += dt if util > 0 else 0.0
            item["util_peak_pct_sampled"] = max(item["util_peak_pct_sampled"], util)
            item["memory_peak_mib_sampled"] = max(item["memory_peak_mib_sampled"], mem)
            item["power_w_s_sampled"] += dt * float(device["power_w"])
            apps = [app for app in left.get("gpu_apps", []) if app["gpu_uuid"] == device["uuid"]]
            item["app_present_s_estimate"] += dt if apps else 0.0
            item["app_pids_seen"].update(app["pid"] for app in apps)
            if device["uuid"] == uuids[0] and mem > 100:
                first_mem_positive_ns = (left["monotonic_ns"] if first_mem_positive_ns is None
                                         else first_mem_positive_ns)
                last_mem_positive_ns = left["monotonic_ns"]
    wall_s = (end_ns - start_ns) / 1e9
    for item in gpu.values():
        item["app_pids_seen"] = sorted(item["app_pids_seen"])
        item["util_mean_pct_covered_estimate"] = (
            100 * item["util_equivalent_s_estimate"] / (covered_ns / 1e9)
            if covered_ns else None)
        item["power_mean_w_covered_estimate"] = (
            item.pop("power_w_s_sampled") / (covered_ns / 1e9)
            if covered_ns else None)
    user_s = system_s = read_bytes = write_bytes = 0.0
    per_process = []
    for ident, (first, last) in cpu_first_last.items():
        proc_user = max(0, last.get("user_ticks", 0) - first.get("user_ticks", 0)) / ticks_per_s
        proc_system = max(0, last.get("system_ticks", 0) - first.get("system_ticks", 0)) / ticks_per_s
        proc_read = max(0, last.get("io", {}).get("read_bytes", 0)
                        - first.get("io", {}).get("read_bytes", 0))
        proc_write = max(0, last.get("io", {}).get("write_bytes", 0)
                         - first.get("io", {}).get("write_bytes", 0))
        user_s += proc_user
        system_s += proc_system
        read_bytes += proc_read
        write_bytes += proc_write
        per_process.append({"pid": ident[0], "start_ticks": ident[1],
                            "cmdline": first.get("cmdline", "")[:300],
                            "cpu_user_s_lower_bound": proc_user,
                            "cpu_system_s_lower_bound": proc_system,
                            "io_read_bytes_lower_bound": proc_read,
                            "io_write_bytes_lower_bound": proc_write})
    per_process.sort(key=lambda proc: proc["cpu_user_s_lower_bound"]
                     + proc["cpu_system_s_lower_bound"], reverse=True)
    return {"job_start_monotonic_ns": start_ns, "job_end_monotonic_ns": end_ns,
            "job_wall_s": wall_s, "covered_s": covered_ns / 1e9,
            "unobserved_s": (end_ns - start_ns - covered_ns) / 1e9,
            "invalid_or_long_gap_s": gap_ns / 1e9,
            "max_sample_start_interval_s": max_gap_s,
            "window_error_samples": error_rows, "window_incomplete_uuid_samples": incomplete_rows,
            "window_sample_count": sum(start_ns <= r["monotonic_ns"] < end_ns for r in rows),
            "gpu_by_uuid": gpu, "worker_pid": worker_pid,
            "worker_lineage_sample_count": lineage_sample_count,
            "worker_lineage_pid_start_identities": [list(x) for x in sorted(cpu_first_last)],
            "worker_lineage_per_process_sampled_lower_bounds": per_process,
            "worker_lineage_cpu_user_s_lower_bound": user_s,
            "worker_lineage_cpu_system_s_lower_bound": system_s,
            "worker_lineage_rss_peak_bytes_sampled": rss_peak_bytes,
            "worker_lineage_io_read_bytes_lower_bound": int(read_bytes),
            "worker_lineage_io_write_bytes_lower_bound": int(write_bytes),
            "gpu0_first_gt100_sample_start_monotonic_ns": first_mem_positive_ns,
            "gpu0_last_gt100_sample_start_monotonic_ns": last_mem_positive_ns,
            "method": "left-sample-hold between adjacent same-host monotonic sample starts, clipped to exact trusted worker window; >1.5s intervals unobserved",
            "limitations": [
                "NVML utilization is a sampled device-level estimate, not exact SM busy time.",
                "GPU >100 MiB is sampled occupancy, not exact allocation or kernel time.",
                "NVML application PIDs can differ from visible /proc namespace PIDs; no direct PID join is assumed.",
                "Worker lineage CPU/I/O are lower bounds; sub-second processes may be missed.",
                "No local-host monotonic timestamp is subtracted from remote-host monotonic time.",
            ]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-ns", type=int, required=True)
    parser.add_argument("--end-ns", type=int, required=True)
    parser.add_argument("--worker-pid", type=int, required=True)
    parser.add_argument("--ticks-per-s", type=int, default=100)
    args = parser.parse_args()
    rows, source = read_remote_window(args.start_ns, args.end_ns)
    print(json.dumps({"schema_version": 1, "phase": "phase2", "generated_at_utc": utc(),
                      "source": source,
                      "result": analyze(rows, args.start_ns, args.end_ns,
                                        args.worker_pid, args.ticks_per_s)},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
