#!/usr/bin/env python3
"""One bounded real-A100 import preflight before the fresh dense control.

This site-only setup job consumes the first-GPU six-hour budget. It is not a
Harness candidate, a quality result, or a synthetic CPU import test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path("/share/project/shuchenweng/yongyan_liu/Experiments/SolAgent/20260913-a100-original-baseline")
REPO = PROJECT / "repo"
sys.path.insert(0, str(REPO))
from models.sana_video_2b_a100.baseline import gpu_infer  # noqa: E402
from orchestration import site_observer  # noqa: E402

PYTHON = "/tmp/distillschedule-helios-official-torch210/bin/python"
EXPECTED_TRANSFORMER_SHA256 = "6a4c853e8f443734db72befdca0910b4cbf2ccf4a83792c7be03427a36522332"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    if not Path(PYTHON).is_file() or not gpu_infer.MODEL.is_dir():
        raise RuntimeError("pinned A100 Python or read-only model snapshot absent")
    source = gpu_infer.source_root()
    transformer = source / "python/sglang/multimodal_gen/runtime/models/dits/sana_video.py"
    if gpu_infer.sha256(transformer) != EXPECTED_TRANSFORMER_SHA256:
        raise RuntimeError("dense SGLang transformer source differs before GPU import")
    observer = site_observer.audit()
    if (observer.get("status") != "running" or observer.get("error_samples") != 0
            or observer.get("incomplete_uuid_samples") != 0
            or observer.get("last_sample_age_s", 100) > 3):
        raise RuntimeError(f"independent observer not healthy before GPU import: {observer}")
    output = PROJECT / "artifacts/gpu_import_preflight" / f"attempt-{args.attempt:02d}"
    if output.exists():
        raise RuntimeError(f"GPU import attempt directory already exists: {output}")
    output.mkdir(parents=True)
    gpu_infer.event("gpu_import_preflight_queue_begin", attempt=args.attempt, output=str(output))
    slot, lock = gpu_infer.reserve_one(output)
    try:
        os.environ["SOL_A100_PYTHON"] = PYTHON
        env = gpu_infer.child_env(gpu_infer.GPU_UUIDS[slot], slot)
        env["SOL_A100_GPU_IMPORT_PREFLIGHT"] = "1"
        env.pop("SOL_A100_CPU_IMPORT_STUB", None)
        command = [PYTHON, str(REPO / "orchestration/site_import_preflight.py")]
        budget = gpu_infer.first_gpu_budget()  # immediately before first GPU child
        gpu_infer.event("gpu_import_preflight_launch_begin", attempt=args.attempt,
                        output=str(output), gpu_uuid=gpu_infer.GPU_UUIDS[slot], command=command)
        started_ns = time.monotonic_ns()
        log_path = output / "import.log"
        with log_path.open("w") as log:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            gpu_infer.event("gpu_import_preflight_job_start", attempt=args.attempt,
                            output=str(output), gpu_uuid=gpu_infer.GPU_UUIDS[slot],
                            gpu_index=slot, pid=child.pid)
            try:
                exit_code = child.wait(timeout=180)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                exit_code = 124
        finished_ns = time.monotonic_ns()
        gpu_infer.event("gpu_import_preflight_job_end", attempt=args.attempt,
                        output=str(output), gpu_uuid=gpu_infer.GPU_UUIDS[slot],
                        gpu_index=slot, pid=child.pid, exit_code=exit_code)
        gpu_infer.wait_gpu_clean(gpu_infer.GPU_UUIDS[slot])
        transcript = log_path.read_text(errors="replace")
        receipts = [json.loads(line) for line in transcript.splitlines() if line.startswith("{")]
        import_receipt = receipts[-1] if receipts else None
        passed = (exit_code == 0 and isinstance(import_receipt, dict)
                  and import_receipt.get("status") == "PASS"
                  and import_receipt.get("cuda_visible_devices") == gpu_infer.GPU_UUIDS[slot]
                  and import_receipt.get("import_only_stub") is None)
        result = {"schema_version": 1, "status": "PASS" if passed else "FAILED",
                  "attempt": args.attempt, "host": socket.gethostname(), "finished_utc": utc(),
                  "gpu_uuid": gpu_infer.GPU_UUIDS[slot], "gpu_index": slot, "pid": child.pid,
                  "exit_code": exit_code, "started_monotonic_ns": started_ns,
                  "finished_monotonic_ns": finished_ns,
                  "duration_s": (finished_ns - started_ns) / 1e9,
                  "log": str(log_path), "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
                  "import_receipt": import_receipt, "first_gpu_budget": budget}
        (output / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if passed else 1
    finally:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
