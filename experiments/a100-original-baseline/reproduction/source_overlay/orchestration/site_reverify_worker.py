#!/usr/bin/env python3
"""STAGED, not deployed: trusted remote original Cache reverify CPU worker.

Only the exact delivered R05/R08 predecessor pins are staged. Deployment still
requires separate review and a host bridge approval. This is site transport:
it runs the archived collector and plan_eval, with no new quality rule or GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EXPECTED_REPO = (Path("/share/project/shuchenweng/yongyan_liu/Experiments/SolAgent"
                      "/20260913-a100-original-baseline") / "repo")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from orchestration import site_quality_worker as quality  # noqa: E402


QUALITY_WORKER_SHA256 = "f7e5d3acdc9f1c82f7d4abbdb75a75aa99766a551e24a57353ce5ab5af22a3c7"
# Exact real Cache DELIVERY SHA 0940aca6... appeared at 09:58 UTC. These are
# staged pins only: this worker is not installed on the remote coordinator and
# the host transport still refuses execution until separately approved.
APPROVED_POINTS: dict[str, dict] = {
    "R05": {
        "launch.sh": "392f725844f3606360c56624126f169e98c5a376f0c5e17cd36f911d50ca487e",
        "manifest.resolved.toml": "3d1cddb42c2f04fb2c387c81fa00d9dee14b1ff6c3432b74c5ba37f61b585342",
        "metadata.json": "94244e3a396b7d02f75185f538435b887bba307f875a5fa39ba18181d0463244",
        "outputs/benchmark.json": "383803093299e559402537371946911b500f8aa55f8582649a05acadbda9783e",
        "outputs/quality.json": "63c03a607fd0133a676d4fad6a164e7cfc57ddd3fdcdd9f1cfd00ee48d4156ef",
        "outputs/collection.json": "22f333cdb203052d8e35c3500501f9366d778e14506c7c99e3b40840ed5e909c",
        "assess_verdict.json": "fd0de26e2b475d89823ec469e41e53e3495fa5efec471efbc9fc27d6d8c3a4d1",
        "original_judge_receipt_relative": "phase3_quality_attempts/20260914T062035.537017+0000_2975905/receipt.json",
        "original_judge_receipt_sha256": "636f7ff9b8d382045dfd41862c96ec9f79412dbf4e939f12a61a1948670d63d7",
        "remote_reverify_predecessor_sha256": "fd0de26e2b475d89823ec469e41e53e3495fa5efec471efbc9fc27d6d8c3a4d1",
    },
    "R08": {
        "launch.sh": "6c8f610c4e1f6a0085cde0bb56f7b80f0133f57725f8d6375d2f54644823271f",
        "manifest.resolved.toml": "7b77a2d868e1257da3635901b46c8ae552c762b3e10532d555e700ca39c1ece9",
        "metadata.json": "3a609fa4ac789375b4e33db8ca3d7dccfa9e9b37e2f0f4570db1806eb7f76d78",
        "outputs/benchmark.json": "870b56d54408b480881d663697e7be19960b4e4248c724a46f24761866aa3576",
        "outputs/quality.json": "f709caabd587b6b49155dc2511f00389a5a19393d2d75e704ebd2fd82ec48672",
        "outputs/collection.json": "d545aa8be651ea7149a4fb3c9f711352a29c8023bb769637db62d2afea600e32",
        "assess_verdict.json": "6313b3c8b9b186c7a21ba5857ec59976ea6a87778d56f349b0656ff0fab4da29",
        "original_judge_receipt_relative": "phase3_quality_attempts/20260914T091442.857077+0000_3058273/receipt.json",
        "original_judge_receipt_sha256": "e83d0c0b31e4b5d770154f6633b1e7212291c5df23945d2ca7c114d0e30caa19",
        "remote_reverify_predecessor_sha256": None,
    },
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def quality_preflight_spec(key: str, approved: dict) -> dict:
    """Pin the current post-quality bundle, not its original pre-quality metadata."""
    base = quality.RUNS[key]
    return {**base,
            "benchmark_before_sha256": approved["outputs/benchmark.json"],
            "remote_bundle_sha256": {
                relative: approved[relative]
                for relative in ("launch.sh", "manifest.resolved.toml", "metadata.json")}}


def preflight(key: str) -> tuple[Path, dict, dict]:
    if REPO != EXPECTED_REPO:
        raise RuntimeError("remote reverify worker is outside the trusted coordinator repo")
    if (Path(quality.__file__).resolve() != REPO / "orchestration/site_quality_worker.py"
            or quality.sha256(Path(quality.__file__)) != QUALITY_WORKER_SHA256):
        raise RuntimeError("remote original-quality helper source SHA changed")
    if key not in APPROVED_POINTS or key not in quality.RUNS:
        raise RuntimeError("remote reverify point is not explicitly approved")
    approved = APPROVED_POINTS[key]
    base = quality.RUNS[key]
    run = quality.CACHE / "runs" / base["name"]
    exact = quality_preflight_spec(key, approved)
    verified = quality.preflight(run, exact)
    for relative in ("outputs/quality.json", "outputs/collection.json",
                     "assess_verdict.json", "metadata.json", "launch.sh",
                     "manifest.resolved.toml"):
        path = run / relative
        if (not path.is_file() or path.is_symlink()
                or quality.sha256(path) != approved[relative]):
            raise RuntimeError(f"remote reverify predecessor SHA changed: {relative}")
    original_receipt = run / approved["original_judge_receipt_relative"]
    if (not original_receipt.is_file() or original_receipt.is_symlink()
            or not original_receipt.resolve().is_relative_to(run)
            or quality.sha256(original_receipt) != approved["original_judge_receipt_sha256"]):
        raise RuntimeError("remote original LPIPS receipt changed")
    existing = run / "reverify_verdict.json"
    expected_old = approved["remote_reverify_predecessor_sha256"]
    if existing.is_symlink() or existing.is_file() != (expected_old is not None):
        raise RuntimeError("remote reverify fixed output predecessor presence changed")
    if expected_old is not None and quality.sha256(existing) != expected_old:
        raise RuntimeError("remote reverify fixed output predecessor SHA changed")
    return run, approved, verified


def snapshot(run: Path, target: Path) -> dict:
    result = quality.snapshot(run, target)
    verdict = run / "reverify_verdict.json"
    if verdict.exists():
        if not verdict.is_file() or verdict.is_symlink():
            raise RuntimeError("reverify output is not a regular file")
        destination = target / "reverify_verdict.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(verdict, destination)
        digest = quality.sha256(verdict)
        if quality.sha256(destination) != digest:
            raise RuntimeError("reverify output snapshot SHA differs")
        result["reverify_verdict.json"] = {"sha256": digest,
                                           "size_bytes": verdict.stat().st_size}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_key", choices=tuple(quality.RUNS))
    args = parser.parse_args()
    key = args.run_key
    # No attempt directory, collector, or model dependency is touched until
    # the exact remote preflight (which currently always denies) succeeds.
    try:
        run, approved, verified = preflight(key)
    except Exception as exc:
        print(json.dumps({"status": "blocked_preflight", "run_key": key,
                          "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True))
        return 2
    started_ns, started_utc = time.monotonic_ns(), utc()
    attempt = run / "phase3_reverify_attempts" / (
        started_utc.replace(":", "").replace("-", "").replace("+00:00", "Z")
        + f"_{os.getpid()}")
    receipt: dict = {"schema_version": 1, "kind": "phase3_original_cpu_reverify",
                     "status": "failed", "run_key": key, "run_dir": str(run),
                     "host": socket.gethostname(), "started_at_utc": started_utc,
                     "started_monotonic_ns": started_ns, "preflight": verified,
                     "approved_predecessor": approved,
                     "site_transport_not_quality_decision": True}
    try:
        attempt.mkdir(parents=True, exist_ok=False)
        receipt["pre_snapshot"] = snapshot(run, attempt / "pre")
        old = run / "reverify_verdict.json"
        if old.is_file():
            archive = attempt / "prior_reverify_verdict.json"
            os.replace(old, archive)
            if quality.sha256(archive) != approved["remote_reverify_predecessor_sha256"]:
                raise RuntimeError("remote archived reverify predecessor SHA differs")
            receipt["prior_reverify_archive_sha256"] = quality.sha256(archive)
        if old.exists() or old.is_symlink():
            raise RuntimeError("remote fixed reverify output path remains occupied")
        env = quality.evaluation_env(attempt, key)
        collect = [str(quality.EVAL_PYTHON), str(quality.COLLECTOR), str(run),
                   "--baseline-run-dir", str(quality.DENSE)]
        receipt["collect"] = quality.invoke("collect", collect, env, attempt, timeout=1800)
        if receipt["collect"]["exit_code"] != 0:
            raise RuntimeError("original collector failed; inspect preserved logs")
        if quality.frames_digest(run / "outputs/frames")[0] != quality.RUNS[key]["frames_digest"]:
            raise RuntimeError("original collector changed frozen five-prompt frames")
        quality.check_benchmark(run / "outputs/benchmark.json", quality.RUNS[key]["total_s"])
        plan = [str(quality.EVAL_PYTHON), str(quality.PLAN_EVAL), "--model",
                "sana_video_2b_a100", "--no-gemini", "--no-refresh-collection",
                "--assess", str(run), "--baseline-frames",
                str(quality.DENSE / "outputs/frames"), "--out", str(old)]
        receipt["plan_eval"] = quality.invoke("plan_eval", plan, env, attempt, timeout=300)
        if receipt["plan_eval"]["exit_code"] != 0:
            raise RuntimeError("original no-Gemini plan_eval failed; inspect preserved logs")
        verdict = quality.json_read(old)
        if not math.isfinite(float(verdict.get("lpips_max"))):
            raise RuntimeError("original plan_eval did not produce finite lpips_max")
        receipt["reverify_verdict_sha256"] = quality.sha256(old)
        receipt["reverify_verdict"] = verdict
        q = quality.json_read(run / "outputs/quality.json")
        receipt["quality_status"] = q.get("status")
        receipt["quality_blockers"] = q.get("promotion_blockers")
        receipt["lpips_pair_audit"] = quality.pair_audit(run, q)
        audit = attempt / "network_audit.jsonl"
        events = [json.loads(line) for line in audit.read_text().splitlines()] if audit.is_file() else []
        receipt["network_guard_loaded_pids"] = sorted({row["pid"] for row in events
                                                       if row.get("kind") == "guard_loaded"})
        receipt["blocked_network_attempts"] = [row for row in events
                                                if row.get("kind", "").startswith("blocked_")]
        if len(receipt["network_guard_loaded_pids"]) < 3 or receipt["blocked_network_attempts"]:
            raise RuntimeError("offline guard missing or an outbound network attempt occurred")
        receipt["status"] = "original_remote_reverify_complete"
    except Exception as exc:
        receipt["error_type"] = type(exc).__name__
        receipt["error"] = str(exc)
    finally:
        if attempt.is_dir():
            try:
                receipt["post_snapshot"] = snapshot(run, attempt / "post")
            except Exception as exc:
                receipt["status"] = "failed"
                receipt["post_snapshot_error"] = f"{type(exc).__name__}: {exc}"
        receipt["finished_at_utc"] = utc()
        receipt["finished_monotonic_ns"] = time.monotonic_ns()
        receipt["wall_s"] = (receipt["finished_monotonic_ns"] - started_ns) / 1e9
        path = attempt / "receipt.json"
        if attempt.is_dir():
            path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": receipt["status"], "run_key": key,
                          "attempt_dir": str(attempt) if attempt.is_dir() else None,
                          "receipt_sha256": quality.sha256(path) if path.is_file() else None,
                          "wall_s": receipt["wall_s"], "error": receipt.get("error")}, sort_keys=True))
    return 0 if receipt["status"] == "original_remote_reverify_complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
