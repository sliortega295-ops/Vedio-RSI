#!/usr/bin/env python3
"""UNACTIVATED template for one future exact composed-run original CPU judge.

No composed run is approved or routed. This file is not installed remotely or
dispatched by the broker. After the archived Master creates and measures a real
composed run, site review may pin its exact identity, source, five videos,
80 aligned frames and predecessor hashes. It then invokes the unchanged
collector, LPIPS judge and plan_eval in the already-authorized offline env.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REMOTE_REPO = Path("/share/project/shuchenweng/yongyan_liu/Experiments/SolAgent/20260913-a100-original-baseline/repo")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from orchestration import site_quality_worker as quality  # noqa: E402


QUALITY_WORKER_SHA256 = "f7e5d3acdc9f1c82f7d4abbdb75a75aa99766a551e24a57353ce5ab5af22a3c7"
APPROVED_RUN: dict | None = None  # Must remain None until exact native run review.


def preflight() -> tuple[Path, dict]:
    if REPO != REMOTE_REPO:
        raise RuntimeError("integrated CPU judge requires the trusted remote coordinator")
    if (Path(quality.__file__).resolve() != REPO / "orchestration/site_quality_worker.py"
            or quality.sha256(Path(quality.__file__)) != QUALITY_WORKER_SHA256):
        raise RuntimeError("original quality helper source SHA changed")
    approved = APPROVED_RUN
    if not isinstance(approved, dict):
        raise RuntimeError("no exact composed run is approved")
    name = approved.get("name")
    if (not isinstance(name, str) or not name.startswith("20260914-")
            or "/" in name or ".." in name):
        raise RuntimeError("composed run identity is invalid")
    run = REPO / "runs" / name
    if run.is_symlink() or run.resolve().parent != REPO / "runs" or not run.is_dir():
        raise RuntimeError("composed run is not a real coordinator run")
    files = approved.get("pre_sha256")
    if (not isinstance(files, dict) or set(files) != {
            "launch.sh", "manifest.resolved.toml", "metadata.json",
            "outputs/benchmark.json", "outputs/run_config.json",
            "outputs/quality.json", "outputs/collection.json", "assess_verdict.json"}):
        raise RuntimeError("composed run predecessor set is incomplete")
    for relative, expected in files.items():
        path = run / relative
        if expected is None:
            if path.exists() or path.is_symlink():
                raise RuntimeError(f"composed predecessor unexpectedly exists: {relative}")
        elif (not path.is_file() or path.is_symlink()
              or quality.sha256(path) != expected):
            raise RuntimeError(f"composed predecessor SHA changed: {relative}")
    for path, expected in quality.FIXED_FILES.items():
        if not path.is_file() or path.is_symlink() or quality.sha256(path) != expected:
            raise RuntimeError(f"original judge dependency/source SHA changed: {path.name}")
    if (not quality.EVAL_PYTHON.is_symlink()
            or quality.EVAL_PYTHON.resolve() != quality.EVAL_PYTHON_REAL
            or quality.sha256(quality.EVAL_PYTHON_REAL) != quality.EVAL_PYTHON_SHA256):
        raise RuntimeError("fixed offline evaluation interpreter changed")
    quality.check_benchmark(run / "outputs/benchmark.json", float(approved["total_s"]))
    dense_digest, _ = quality.frames_digest(quality.DENSE / "outputs/frames")
    run_digest, _ = quality.frames_digest(run / "outputs/frames")
    if (dense_digest != quality.DENSE_FRAME_DIGEST
            or run_digest != approved.get("frames_digest")):
        raise RuntimeError("composed/dense 80-frame contents changed")
    videos = (run / "outputs/benchmark.json").read_text()
    entries = json.loads(videos)["export"]["videos"]
    if len(entries) != 5:
        raise RuntimeError("composed video list is not five prompts")
    for prompt, entry in enumerate(entries, 1):
        path = run / "outputs" / ("out.mp4" if prompt == 1 else f"out_{prompt}.mp4")
        if (entry.get("prompt_index") != prompt or not path.is_file()
                or path.is_symlink() or quality.sha256(path) != entry.get("sha256")):
            raise RuntimeError(f"composed p{prompt:02d} video SHA changed")
    return run, approved


def main() -> int:
    try:
        run, approved = preflight()
    except Exception as exc:
        print(json.dumps({"status": "blocked_preflight", "error_type": type(exc).__name__,
                          "error": str(exc)}, sort_keys=True))
        return 2
    started_ns, started_utc = time.monotonic_ns(), quality.utc()
    attempt = run / "phase3_integrated_quality_attempts" / (
        started_utc.replace(":", "").replace("-", "") + f"_{os.getpid()}")
    receipt = {"schema_version": 1, "status": "failed", "kind": "integrated_original_cpu_lpips",
               "run_dir": str(run), "approved_run": approved, "host": socket.gethostname(),
               "started_at_utc": started_utc, "gpu_authorization": "not_granted_by_cpu_judge"}
    try:
        attempt.mkdir(parents=True, exist_ok=False)
        receipt["pre_snapshot"] = quality.snapshot(run, attempt / "pre")
        env = quality.evaluation_env(attempt, "INTEGRATED")
        collect = [str(quality.EVAL_PYTHON), str(quality.COLLECTOR), str(run),
                   "--baseline-run-dir", str(quality.DENSE)]
        receipt["collect"] = quality.invoke("collect", collect, env, attempt, timeout=1800)
        if receipt["collect"]["exit_code"] != 0:
            raise RuntimeError("original integrated collector failed")
        if quality.frames_digest(run / "outputs/frames")[0] != approved["frames_digest"]:
            raise RuntimeError("original collector changed composed frames")
        quality.check_benchmark(run / "outputs/benchmark.json", float(approved["total_s"]))
        plan = [str(quality.EVAL_PYTHON), str(quality.PLAN_EVAL), "--model",
                "sana_video_2b_a100", "--no-gemini", "--no-refresh-collection",
                "--assess", str(run), "--baseline-frames",
                str(quality.DENSE / "outputs/frames"), "--out", str(run / "assess_verdict.json")]
        receipt["plan_eval"] = quality.invoke("plan_eval", plan, env, attempt, timeout=300)
        if receipt["plan_eval"]["exit_code"] != 0:
            raise RuntimeError("original integrated plan_eval failed")
        result = quality.json_read(run / "outputs/quality.json")
        receipt["lpips_pair_audit"] = quality.pair_audit(run, result)
        receipt["quality_status"] = result.get("status")
        receipt["quality_blockers"] = result.get("promotion_blockers")
        audit = attempt / "network_audit.jsonl"
        events = [json.loads(line) for line in audit.read_text().splitlines()] if audit.is_file() else []
        receipt["network_guard_loaded_pids"] = sorted({row["pid"] for row in events
                                                       if row.get("kind") == "guard_loaded"})
        receipt["blocked_network_attempts"] = [row for row in events
                                               if row.get("kind", "").startswith("blocked_")]
        if len(receipt["network_guard_loaded_pids"]) < 3 or receipt["blocked_network_attempts"]:
            raise RuntimeError("offline guard missing or network attempt occurred")
        receipt["status"] = "lpips_complete_visual_gate_pending"
    except Exception as exc:
        receipt["error_type"], receipt["error"] = type(exc).__name__, str(exc)
    finally:
        if attempt.is_dir():
            receipt["post_snapshot"] = quality.snapshot(run, attempt / "post")
            receipt["finished_at_utc"] = quality.utc()
            receipt["wall_s"] = (time.monotonic_ns() - started_ns) / 1e9
            path = attempt / "receipt.json"
            path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": receipt["status"], "run_dir": str(run),
                          "receipt_sha256": quality.sha256(path) if attempt.is_dir() else None,
                          "attempt_dir": str(attempt) if attempt.is_dir() else None}, sort_keys=True))
    return 0 if receipt["status"] == "lpips_complete_visual_gate_pending" else 2


if __name__ == "__main__":
    raise SystemExit(main())
