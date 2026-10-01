#!/usr/bin/env python3
"""Independent one-second GPU UUID and process CPU/I/O observer on the A100 node."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path("/share/project/shuchenweng/yongyan_liu/Experiments/SolAgent/20260913-a100-original-baseline")
OBS = PROJECT / "telemetry/observer"
STATE = OBS / "state.json"
SAMPLES = OBS / "samples.jsonl"
PHASE = "phase1"
GPUS = (
    "GPU-cad55364-e014-c35f-337e-db899afd1c6a",
    "GPU-a904c6f6-6410-dc0f-c7c3-efc068004617",
    "GPU-79ab3e5e-e382-1f9c-bede-1900666e9283",
    "GPU-bab43db0-a051-0419-39f7-e33effa9445b",
    "GPU-44bddf94-7411-b3ad-f195-fdbdd12fa73e",
    "GPU-af8dba1f-f74e-6a55-bb9e-6c8961edbd6e",
    "GPU-5edaa614-d5f8-8a3f-11ec-b128f8d029f3",
    "GPU-01fff32a-1dd6-93a7-1aae-60d378382961",
)
STOP = False


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return state != "Z"
    except (OSError, IndexError):
        return False


def parse_gpu_rows(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 6:
            continue
        index, uuid, gpu_util, mem_util, mem_mib, power_w = fields
        if uuid not in GPUS:
            continue
        rows.append({"index": int(index), "uuid": uuid,
                     "gpu_util_pct": float(gpu_util), "memory_util_pct": float(mem_util),
                     "memory_used_mib": float(mem_mib), "power_w": float(power_w)})
    if [row["uuid"] for row in rows] != list(GPUS):
        raise RuntimeError(f"authorized A100 UUID order changed: {rows}")
    return rows


def gpu_sample() -> tuple[list[dict], list[dict], float]:
    start = time.monotonic()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw",
                          "--format=csv,noheader,nounits"], text=True, capture_output=True,
                         check=True, timeout=15)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                           "--format=csv,noheader,nounits"], text=True, capture_output=True,
                          check=True, timeout=15)
    parsed_apps = []
    for line in apps.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 3 and fields[0] in GPUS:
            try:
                parsed_apps.append({"gpu_uuid": fields[0], "pid": int(fields[1]),
                                    "memory_mib": float(fields[2])})
            except ValueError:
                continue
    return parse_gpu_rows(gpu.stdout), parsed_apps, time.monotonic() - start


def proc_info(pid: int) -> dict | None:
    base = Path("/proc") / str(pid)
    try:
        raw = (base / "stat").read_text()
        fields = raw.rsplit(") ", 1)[1].split()
        cmd = (base / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")[:600]
        io_data = {}
        for line in (base / "io").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("read_bytes", "write_bytes", "rchar", "wchar"):
                io_data[key] = int(value.strip())
        return {"pid": pid, "ppid": int(fields[1]), "start_ticks": int(fields[19]),
                "user_ticks": int(fields[11]), "system_ticks": int(fields[12]),
                "rss_bytes": int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
                "io": io_data, "cmdline": cmd}
    except (OSError, ValueError, IndexError):
        return None


def process_sample(gpu_apps: list[dict]) -> list[dict]:
    processes = {}
    for item in Path("/proc").iterdir():
        if item.name.isdigit():
            info = proc_info(int(item.name))
            if info:
                processes[info["pid"]] = info
    selected = {row["pid"] for row in gpu_apps}
    selected.add(os.getpid())
    marker = str(PROJECT)
    for pid, info in processes.items():
        if marker in info["cmdline"]:
            selected.add(pid)
    changed = True
    while changed:
        prior = len(selected)
        for pid, info in processes.items():
            if info["ppid"] in selected:
                selected.add(pid)
        changed = len(selected) != prior
    return [processes[pid] for pid in sorted(selected) if pid in processes]


def handle_stop(signum: int, frame: object) -> None:
    global STOP
    STOP = True


def run() -> int:
    OBS.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    started_ns = time.monotonic_ns()
    receipt = {"schema_version": 1, "pid": os.getpid(), "host": socket.gethostname(),
               "phase": PHASE,
               "started_at_utc": utc(), "started_monotonic_ns": started_ns,
               "sample_period_s": 1.0, "gpu_uuids": list(GPUS),
               "samples": str(SAMPLES), "clock_ticks_per_s": os.sysconf("SC_CLK_TCK")}
    STATE.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    next_tick = time.monotonic()
    count = 0
    with SAMPLES.open("a", buffering=1) as stream:
        while not STOP:
            t0 = time.monotonic_ns()
            try:
                gpus, apps, gpu_query_s = gpu_sample()
                processes = process_sample(apps)
                row = {"schema_version": 1, "utc": utc(), "host": socket.gethostname(),
                       "monotonic_ns": t0, "gpu_query_s": gpu_query_s,
                       "gpus": gpus, "gpu_apps": apps, "processes": processes,
                       "observer_process_cpu_s": time.process_time(),
                       "sample_overhead_s": (time.monotonic_ns() - t0) / 1e9}
            except Exception as exc:
                row = {"schema_version": 1, "utc": utc(), "host": socket.gethostname(),
                       "monotonic_ns": t0, "error": f"{type(exc).__name__}: {exc}",
                       "sample_overhead_s": (time.monotonic_ns() - t0) / 1e9}
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
            next_tick += 1.0
            time.sleep(max(0.0, next_tick - time.monotonic()))
    receipt.update({"stopped_at_utc": utc(), "stopped_monotonic_ns": time.monotonic_ns(),
                    "sample_count": count, "observer_process_cpu_s": time.process_time()})
    STATE.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return 0


def start() -> dict:
    if STATE.exists():
        existing = json.loads(STATE.read_text())
        if "stopped_at_utc" not in existing and alive(int(existing["pid"])):
            return {"status": "already_running", **existing}
    OBS.mkdir(parents=True, exist_ok=True)
    log = (OBS / "observer.log").open("a")
    child = subprocess.Popen([sys.executable, __file__, "--phase", PHASE, "run"], cwd=PROJECT,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                             start_new_session=True)
    log.close()
    for _ in range(30):
        if STATE.exists():
            state = json.loads(STATE.read_text())
            if state.get("pid") == child.pid:
                return {"status": "started", **state}
        time.sleep(0.1)
    raise RuntimeError("observer did not publish its start receipt")


def status() -> dict:
    if not STATE.exists():
        return {"status": "not_started"}
    state = json.loads(STATE.read_text())
    return {"status": "running" if alive(int(state["pid"])) and "stopped_at_utc" not in state else "stopped",
            **state, "sample_bytes": SAMPLES.stat().st_size if SAMPLES.exists() else 0}


def audit() -> dict:
    current = status()
    if current["status"] == "not_started" or not SAMPLES.exists():
        return {"status": "not_started"}
    count = errors = incomplete = gaps = 0
    first = last = None
    maximum_gap_s = maximum_overhead_s = total_overhead_s = 0.0
    for line in SAMPLES.open():
        row = json.loads(line)
        count += 1
        if "error" in row:
            errors += 1
        if [gpu["uuid"] for gpu in row.get("gpus", [])] != list(GPUS):
            incomplete += 1
        if last is not None:
            interval_s = (row["monotonic_ns"] - last["monotonic_ns"]) / 1e9
            maximum_gap_s = max(maximum_gap_s, interval_s)
            if interval_s > 1.5:
                gaps += 1
        overhead_s = row.get("sample_overhead_s", 0.0)
        total_overhead_s += overhead_s
        maximum_overhead_s = max(maximum_overhead_s, overhead_s)
        if first is None:
            first = row
        last = row
    assert first is not None and last is not None
    span_s = (last["monotonic_ns"] - first["monotonic_ns"]) / 1e9
    observer_cpu_s = last.get("observer_process_cpu_s", 0.0) - first.get("observer_process_cpu_s", 0.0)
    return {"status": current["status"], "pid": current["pid"], "host": current["host"],
            "phase": PHASE,
            "first_sample_utc": first["utc"], "last_sample_utc": last["utc"],
            "sample_count": count, "span_s": span_s, "error_samples": errors,
            "incomplete_uuid_samples": incomplete, "gaps_gt_1p5s": gaps,
            "max_gap_s": maximum_gap_s, "last_sample_age_s": (time.monotonic_ns() - last["monotonic_ns"]) / 1e9,
            "observer_cpu_s": observer_cpu_s, "observer_cpu_fraction": observer_cpu_s / span_s if span_s else None,
            "mean_sample_overhead_s": total_overhead_s / count,
            "max_sample_overhead_s": maximum_overhead_s,
            "last_sample_error": last.get("error"),
            "last_sample_uuid_complete": [gpu["uuid"] for gpu in last.get("gpus", [])] == list(GPUS),
            "gpu_uuids": list(GPUS)}


def stop() -> dict:
    state = status()
    if state["status"] == "running":
        os.kill(int(state["pid"]), signal.SIGTERM)
    return status()


def main() -> int:
    global OBS, STATE, SAMPLES, PHASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("phase1", "phase2", "phase2_smoke"), default="phase1")
    parser.add_argument("command", choices=("start", "run", "status", "audit", "stop"))
    args = parser.parse_args()
    PHASE = args.phase
    if PHASE in ("phase2", "phase2_smoke"):
        OBS = PROJECT / ("telemetry/phase2/observer_smoke" if PHASE == "phase2_smoke"
                         else "telemetry/phase2/observer")
        STATE = OBS / "state.json"
        SAMPLES = OBS / "samples.jsonl"
    result = {"start": start, "status": status, "audit": audit, "stop": stop, "run": run}[args.command]()
    if isinstance(result, dict):
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
