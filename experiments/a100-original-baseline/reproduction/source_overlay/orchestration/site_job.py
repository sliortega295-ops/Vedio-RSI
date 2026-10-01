#!/usr/bin/env python3
"""Remote-node detached job shim for this project's SSH submission adapter."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib as tomli
except ModuleNotFoundError:  # The authorized A100 node runs Python 3.10.
    import tomli

PROJECT = Path("/share/project/shuchenweng/yongyan_liu/Experiments/SolAgent/20260913-a100-original-baseline")
PHASE2_AUTH = PROJECT / "state/phase2_authorization.json"
PHASE2_STATE = PROJECT / "state/phase2_first_gpu_job.json"
PHASE2_LOCK = PROJECT / "state/phase2_first_gpu_job.lock"
PHASE2_LIMIT_S = 6 * 3600
PHASE2_SUBMIT_RESERVE_S = 21 * 60
PHASE2_GPU_UUIDS = (
    "GPU-cad55364-e014-c35f-337e-db899afd1c6a",
    "GPU-a904c6f6-6410-dc0f-c7c3-efc068004617",
    "GPU-79ab3e5e-e382-1f9c-bede-1900666e9283",
    "GPU-bab43db0-a051-0419-39f7-e33effa9445b",
    "GPU-44bddf94-7411-b3ad-f195-fdbdd12fa73e",
    "GPU-af8dba1f-f74e-6a55-bb9e-6c8961edbd6e",
    "GPU-5edaa614-d5f8-8a3f-11ec-b128f8d029f3",
    "GPU-01fff32a-1dd6-93a7-1aae-60d378382961",
)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate(run_dir: Path, launch: Path | None = None) -> Path:
    run = run_dir.resolve()
    if not run.is_dir() or run_dir.is_symlink() or not run.is_relative_to(PROJECT / "repo"):
        raise ValueError("site job run dir must be a real path in the new remote repo")
    if launch is not None and (not launch.is_file() or launch.is_symlink() or launch.parent.resolve() != run):
        raise ValueError("site job launch script must belong to the run dir")
    return run


def write(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return state != "Z"
    except (OSError, IndexError):
        return False


def phase2_authorization() -> dict:
    if not PHASE2_AUTH.is_file() or PHASE2_AUTH.is_symlink():
        raise RuntimeError("phase2 GPU authorization is absent")
    auth = json.loads(PHASE2_AUTH.read_text())
    if (auth.get("schema_version") != 1 or auth.get("phase") != "phase2"
            or auth.get("status") != "armed" or auth.get("remote_project") != str(PROJECT)
            or auth.get("remote_host") != socket.gethostname()
            or auth.get("max_window_s") != PHASE2_LIMIT_S
            or auth.get("submit_reserve_s") != PHASE2_SUBMIT_RESERVE_S
            or auth.get("gpu_uuids") != list(PHASE2_GPU_UUIDS)
            or not isinstance(auth.get("excluded_run_names"), list)
            or not isinstance(auth.get("arm_expires_unix_ns"), int)):
        raise RuntimeError("phase2 GPU authorization identity or policy differs")
    return auth


def phase2_run_allowed(run: Path, auth: dict) -> None:
    repo = PROJECT / "repo"
    parents = {
        repo / "runs",
        repo / "output/experiments/sana2ba100-kernel_aw-0002/worktree/runs",
        repo / "output/experiments/sana2ba100-cache_ca-0002/worktree/runs",
    }
    if run.parent not in parents or run.name in auth["excluded_run_names"]:
        raise RuntimeError("phase2 run is outside allowed parents or predates authorization")
    manifest = run / "manifest.resolved.toml"
    if not manifest.is_file() or manifest.is_symlink():
        raise RuntimeError("phase2 resolved manifest is missing")
    with manifest.open("rb") as stream:
        parsed = tomli.load(stream)
    if parsed.get("resolved_profile", {}).get("env", {}).get("SOL_A100_BUDGET_PHASE") != "phase2":
        raise RuntimeError("phase2 run manifest lacks its exact budget tag")


def phase2_budget_status() -> dict:
    auth = phase2_authorization()
    now = time.time_ns()
    auth_sha = hashlib.sha256(PHASE2_AUTH.read_bytes()).hexdigest()
    if PHASE2_STATE.is_symlink():
        raise RuntimeError("phase2 first-GPU budget cannot be a symlink")
    if PHASE2_STATE.exists():
        state = json.loads(PHASE2_STATE.read_text())
        first = state.get("first_gpu_unix_ns")
        deadline = state.get("deadline_unix_ns")
        if (state.get("schema_version") != 1 or state.get("phase") != "phase2"
                or state.get("host") != socket.gethostname()
                or state.get("authorization_sha256") != auth_sha
                or not isinstance(first, int) or not isinstance(deadline, int)
                or deadline != first + PHASE2_LIMIT_S * 10**9):
            raise RuntimeError("phase2 first-GPU budget is malformed or mismatched")
        latest = deadline - PHASE2_SUBMIT_RESERVE_S * 10**9
    else:
        state = None
        latest = auth["arm_expires_unix_ns"]
        deadline = None
    return {
        "phase": "phase2", "authorized": True, "authorization_sha256": auth_sha,
        "first_gpu_budget_exists": state is not None, "first_gpu_budget": state,
        "checked_at_utc": utc(), "checked_at_unix_ns": now,
        "latest_submit_unix_ns": latest, "deadline_unix_ns": deadline,
        "new_submit_open": now < latest,
    }


def phase2_start_budget(run: Path) -> dict:
    auth = phase2_authorization()
    phase2_run_allowed(run, auth)
    PHASE2_LOCK.parent.mkdir(parents=True, exist_ok=True)
    if PHASE2_LOCK.is_symlink():
        raise RuntimeError("phase2 first-GPU lock cannot be a symlink")
    with PHASE2_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not PHASE2_STATE.exists():
            now = time.time_ns()
            if now >= auth["arm_expires_unix_ns"]:
                raise TimeoutError("phase2 GPU authorization expired before first job")
            state = {
                "schema_version": 1, "phase": "phase2", "first_gpu_utc": utc(),
                "first_gpu_unix_ns": now, "first_gpu_monotonic_ns": time.monotonic_ns(),
                "host": socket.gethostname(), "first_run_dir": str(run),
                "deadline_unix_ns": now + PHASE2_LIMIT_S * 10**9,
                "limit_s": PHASE2_LIMIT_S,
                "authorization_sha256": hashlib.sha256(PHASE2_AUTH.read_bytes()).hexdigest(),
            }
            temp = PHASE2_STATE.with_name(".phase2_first_gpu_job.tmp")
            temp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
            temp.replace(PHASE2_STATE)
        budget = phase2_budget_status()
        fcntl.flock(lock, fcntl.LOCK_UN)
    if not budget["new_submit_open"]:
        raise TimeoutError("phase2 first-GPU new-submit cutoff reached")
    return budget


def start(run_dir: Path, launch: Path, phase: str = "phase1") -> dict:
    run = validate(run_dir, launch)
    if (run / "site-job.json").exists():
        raise RuntimeError("site job already started; refusing duplicate launch")
    if phase not in ("phase1", "phase2"):
        raise ValueError("unknown site GPU budget phase")
    budget = phase2_start_budget(run) if phase == "phase2" else None
    worker = subprocess.Popen(
        [sys.executable, __file__, "worker", "--run-dir", str(run), "--launch", str(launch),
         "--phase", phase],
        cwd=run, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=True,
    )
    receipt = {"schema_version": 1, "status": "started", "pid": worker.pid, "phase": phase,
               "phase2_budget": budget,
               "host": socket.gethostname(), "utc": utc(), "monotonic_ns": time.monotonic_ns()}
    write(run / "site-job.json", receipt)
    return receipt


def worker(run_dir: Path, launch: Path, phase: str = "phase1") -> dict:
    run = validate(run_dir, launch)
    if phase == "phase2":
        phase2_run_allowed(run, phase2_authorization())
        budget = phase2_budget_status()
        if not budget["first_gpu_budget_exists"] or time.time_ns() >= budget["deadline_unix_ns"]:
            raise TimeoutError("phase2 worker budget missing or expired before launch")
    elif phase != "phase1":
        raise ValueError("unknown site GPU budget phase")
    started_ns = time.monotonic_ns()
    output = run / "outputs/run.log"
    output.parent.mkdir(parents=True, exist_ok=True)
    budget_path = PHASE2_STATE if phase == "phase2" else PROJECT / "state/first_gpu_job.json"
    with output.open("w") as log:
        log.write(f"SOL_SITE_JOB_START utc={utc()} monotonic_ns={started_ns} host={socket.gethostname()}\n")
        log.flush()
        proc = subprocess.Popen(["bash", str(launch)], cwd=run, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        budget_error = None
        while proc.poll() is None:
            if phase == "phase2":
                try:
                    current = phase2_budget_status()
                    if not current["first_gpu_budget_exists"]:
                        raise RuntimeError("phase2 budget disappeared during GPU job")
                    deadline_ns = current["deadline_unix_ns"]
                except Exception as exc:
                    budget_error = f"{type(exc).__name__}: {exc}"
                    deadline_ns = 0
                if time.time_ns() >= deadline_ns:
                    timed_out = True
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                    break
            elif budget_path.exists():
                try:
                    deadline_ns = json.loads(budget_path.read_text())["deadline_unix_ns"]
                except (OSError, ValueError, KeyError):
                    deadline_ns = None
                if isinstance(deadline_ns, int) and time.time_ns() >= deadline_ns:
                    timed_out = True
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                    break
            time.sleep(1)
        code = proc.wait()
        log.write(f"SOL_SITE_JOB_END utc={utc()} monotonic_ns={time.monotonic_ns()} exit_code={code} timed_out={timed_out}\n")
    result = {"schema_version": 1, "phase": phase, "finished": True, "exit_code": code,
              "timed_out_budget": timed_out, "finished_at_utc": utc(),
              "finished_monotonic_ns": time.monotonic_ns(), "worker_pid": os.getpid(),
              "duration_s": (time.monotonic_ns() - started_ns) / 1e9,
              "budget_error": budget_error}
    write(run / "job-finished.json", result)
    return result


def status(run_dir: Path) -> dict:
    run = validate(run_dir)
    job_path = run / "site-job.json"
    finished = run / "job-finished.json"
    if finished.exists():
        return json.loads(finished.read_text())
    if not job_path.exists():
        return {"status": "not_started", "finished": False}
    job = json.loads(job_path.read_text())
    return {"status": "running" if process_alive(job["pid"]) else "lost",
            "finished": False, "alive": process_alive(job["pid"]), "pid": job["pid"],
            "host": job.get("host"), "checked_at_utc": utc()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "worker", "status", "budget-status"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--launch", type=Path)
    parser.add_argument("--phase", choices=("phase1", "phase2"), default="phase1")
    args = parser.parse_args()
    if args.command in ("start", "worker") and args.launch is None:
        parser.error("start/worker requires --launch")
    if args.command != "budget-status" and args.run_dir is None:
        parser.error("start/worker/status requires --run-dir")
    result = (start(args.run_dir, args.launch, args.phase) if args.command == "start" else
              worker(args.run_dir, args.launch, args.phase) if args.command == "worker" else
              status(args.run_dir) if args.command == "status" else phase2_budget_status())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
