"""Fail-closed host gate for exact archived Master Cache re-verification.

The delivered R05/R08 pins are staged only. Each point still needs a pinned
pre-call stale-output quarantine receipt and a reviewed trusted worker SHA.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from orchestration import site_remote


COORD = Path("/home/lyy/Experiments/SolAgent/20260913-a100-original-baseline/repo")
PROJECT = COORD.parent
CACHE = COORD / "output/experiments/sana2ba100-cache_ca-0002/worktree"
DELIVERY = CACHE / "DELIVERY.json"
RUNS = CACHE / "runs"
MODEL_ID = "sana_video_2b_a100"
DELIVERY_SHA256 = "0940aca6c4e90502b808f5d3441f49a7b3d0d1ac1c2f10a4170e6e4ab2920e85"
APPROVED_POINTS: dict[str, dict] = {
    "20260914-054324-cache_ca_r05_taylorseer_t0180_w10_o1": {
        "run_key": "R05", "delivery_sha256": DELIVERY_SHA256, "total_s": 371.36,
        "predecessor_sha256": {
            "launch.sh": "392f725844f3606360c56624126f169e98c5a376f0c5e17cd36f911d50ca487e",
            "manifest.resolved.toml": "3d1cddb42c2f04fb2c387c81fa00d9dee14b1ff6c3432b74c5ba37f61b585342",
            "metadata.json": "598904fecc09e1d4d264f6787e99a8b8ccc778c1679792af289684ff8a5e387e",
            "outputs/run_config.json": "0c98e68c6aa33fbb2a6cc1f3d0041108206439efb915af97557aa92ebcfded51",
            "outputs/benchmark.json": "383803093299e559402537371946911b500f8aa55f8582649a05acadbda9783e",
            "outputs/quality.json": "63c03a607fd0133a676d4fad6a164e7cfc57ddd3fdcdd9f1cfd00ee48d4156ef",
            "outputs/collection.json": "22f333cdb203052d8e35c3500501f9366d778e14506c7c99e3b40840ed5e909c",
            "assess_verdict.json": "fd0de26e2b475d89823ec469e41e53e3495fa5efec471efbc9fc27d6d8c3a4d1",
        },
        "quality_receipt_path": RUNS / "20260914-054324-cache_ca_r05_taylorseer_t0180_w10_o1/phase3_quality_attempts/20260914T062035.537017+0000_2975905/receipt.json",
        "quality_receipt_sha256": "636f7ff9b8d382045dfd41862c96ec9f79412dbf4e939f12a61a1948670d63d7",
        "stale_reverify_sha256": "fd0de26e2b475d89823ec469e41e53e3495fa5efec471efbc9fc27d6d8c3a4d1",
        "guard_receipt_path": RUNS / "20260914-054324-cache_ca_r05_taylorseer_t0180_w10_o1/phase3_reverify_preflight/20260914T102319.216294Z_b316cb493f7444f1a3126ab13d5e5e43/receipt.json",
        "guard_receipt_sha256": "cd5d4a0156103c80ac8e451ab00f352ba266c1d60822410f3512f7bd9325c7d4",
    },
    "20260914-083326-cache_ca_r08_taylorseer_t0180_w30_o1": {
        "run_key": "R08", "delivery_sha256": DELIVERY_SHA256, "total_s": 480.96,
        "predecessor_sha256": {
            "launch.sh": "6c8f610c4e1f6a0085cde0bb56f7b80f0133f57725f8d6375d2f54644823271f",
            "manifest.resolved.toml": "7b77a2d868e1257da3635901b46c8ae552c762b3e10532d555e700ca39c1ece9",
            "metadata.json": "b9689f9e763827a732e9d014e57d5b9f23cd3ef25f0ad60a6c9438b36c706da5",
            "outputs/run_config.json": "2eac08b00536f71a6f0d5359f91114d198f7cd71c496bccad18a6d37c54d0989",
            "outputs/benchmark.json": "870b56d54408b480881d663697e7be19960b4e4248c724a46f24761866aa3576",
            "outputs/quality.json": "f709caabd587b6b49155dc2511f00389a5a19393d2d75e704ebd2fd82ec48672",
            "outputs/collection.json": "d545aa8be651ea7149a4fb3c9f711352a29c8023bb769637db62d2afea600e32",
            "assess_verdict.json": "6313b3c8b9b186c7a21ba5857ec59976ea6a87778d56f349b0656ff0fab4da29",
        },
        "quality_receipt_path": RUNS / "20260914-083326-cache_ca_r08_taylorseer_t0180_w30_o1/phase3_quality_attempts/20260914T091442.857077+0000_3058273/receipt.json",
        "quality_receipt_sha256": "e83d0c0b31e4b5d770154f6633b1e7212291c5df23945d2ca7c114d0e30caa19",
        "stale_reverify_sha256": "f1ed56e8c3cc11b4486d0d7fa2c3f107d4b223ee45cda16f29abb5f6507b34ca",
        "guard_receipt_path": RUNS / "20260914-083326-cache_ca_r08_taylorseer_t0180_w30_o1/phase3_reverify_preflight/20260914T100556.678685Z_59d08234ec79414ba3fd01029a65469e/receipt.json",
        "guard_receipt_sha256": "5df8fb34dad7f175758fb03fa602eecf7baa887a9ef5d16789e65622de82ff03",
    },
}
REMOTE_RUNS = site_remote.REMOTE_PROJECT / "repo/output/experiments/sana2ba100-cache_ca-0002/worktree/runs"
LOCAL_WORKER = COORD / "orchestration/site_reverify_worker.py"
REMOTE_WORKER = site_remote.REMOTE_PROJECT / "repo/orchestration/site_reverify_worker.py"
TRUSTED_WORKER_SHA256 = "6cc90d29303feee23a1a6aa3e3f99b7b46444c4a4708afbea78df7390fa61ff9"
REMOTE_PYTHON = "/tmp/distillschedule-helios-official-torch210/bin/python3.10"
TRUSTED_PYTHON_SHA256 = "7c9e72ea1f06138ed5f52f49894c62f29e51e302f0d02f8fb18490e90e994331"
SYNC_FILES = ("outputs/benchmark.json", "outputs/quality.json", "outputs/collection.json",
              "assess_verdict.json", "reverify_verdict.json")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_approved(run: Path) -> dict:
    if run.is_symlink() or run.resolve().parent != RUNS or run != run.resolve():
        raise ValueError("reverify run is not an exact real seq0002 Cache run")
    if not DELIVERY.is_file() or DELIVERY.is_symlink():
        raise RuntimeError("reverify denied: Cache DELIVERY absent")
    delivery = json.loads(DELIVERY.read_text())
    if (delivery.get("schema_version") != 2 or delivery.get("status") != "complete"
            or delivery.get("component") != "cache"
            or delivery.get("model_id") != MODEL_ID):
        raise RuntimeError("reverify denied: Cache DELIVERY schema/identity invalid")
    points = delivery.get("frontier_points")
    if not isinstance(points, list) or not points:
        raise RuntimeError("reverify denied: Cache DELIVERY has no frontier points")
    matched = [point for point in points if isinstance(point, dict)
               and point.get("run_dir") == f"runs/{run.name}"]
    if len(matched) != 1 or run.name not in APPROVED_POINTS:
        raise RuntimeError("reverify denied: exact DELIVERY frontier point not approved")
    spec = APPROVED_POINTS[run.name]
    if sha(DELIVERY) != spec.get("delivery_sha256"):
        raise RuntimeError("reverify denied: DELIVERY changed after approval")
    for relative, expected in spec.get("predecessor_sha256", {}).items():
        path = run / relative
        if not path.is_file() or path.is_symlink() or sha(path) != expected:
            raise RuntimeError(f"reverify denied: predecessor SHA mismatch: {relative}")
    if not spec.get("predecessor_sha256") or not spec.get("quality_receipt_sha256"):
        raise RuntimeError("reverify denied: quality predecessor receipt unpinned")
    guard_path = spec.get("guard_receipt_path")
    if not guard_path or not spec.get("guard_receipt_sha256"):
        raise RuntimeError("reverify denied: pre-call stale-output guard not approved")
    guard = Path(guard_path)
    if (not guard.is_file() or guard.is_symlink()
            or not guard.resolve().is_relative_to(run / "phase3_reverify_preflight")
            or sha(guard) != spec["guard_receipt_sha256"]):
        raise RuntimeError("reverify denied: pre-call guard receipt missing or changed")
    guard_data = json.loads(guard.read_text())
    if (guard_data.get("run_dir") != str(run)
            or guard_data.get("fixed_output_absent") is not True
            or guard_data.get("predecessor_sha256") != spec["stale_reverify_sha256"]
            or (run / "reverify_verdict.json").exists()):
        raise RuntimeError("reverify denied: pre-call guard semantics changed")
    receipt = Path(spec["quality_receipt_path"])
    if (not receipt.is_file() or receipt.is_symlink()
            or not receipt.resolve().is_relative_to(run)
            or sha(receipt) != spec["quality_receipt_sha256"]):
        raise RuntimeError("reverify denied: original judge receipt absent or changed")
    return {"run_dir": str(run), "delivery_sha256": spec["delivery_sha256"],
            "quality_receipt_sha256": spec["quality_receipt_sha256"]}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def trusted_remote_worker() -> dict:
    if (not LOCAL_WORKER.is_file() or LOCAL_WORKER.is_symlink()
            or sha(LOCAL_WORKER) != TRUSTED_WORKER_SHA256):
        raise RuntimeError("host trusted reverify worker SHA mismatch")
    worker_sha = site_remote.ssh("sha256sum", str(REMOTE_WORKER), timeout=20).stdout.split()[0]
    python_sha = site_remote.ssh("sha256sum", REMOTE_PYTHON, timeout=20).stdout.split()[0]
    if worker_sha != TRUSTED_WORKER_SHA256 or python_sha != TRUSTED_PYTHON_SHA256:
        raise RuntimeError("remote trusted reverify worker/interpreter SHA mismatch")
    return {"worker_sha256": worker_sha, "interpreter_sha256": python_sha}


def snapshot_local(run: Path) -> tuple[Path, dict]:
    marker = utc().replace(":", "").replace("-", "").replace("+00:00", "Z") + f"_{os.getpid()}"
    target = PROJECT / "artifacts/phase3/pre_reverify_snapshots/host" / run.name / marker
    target.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for relative in ("metadata.json", "launch.sh", "manifest.resolved.toml",
                     "outputs/run_config.json", *SYNC_FILES):
        source = run / relative
        if not source.exists():
            continue
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"host reverify snapshot source unsafe: {relative}")
        destination = target / "pre" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha(destination) != sha(source):
            raise RuntimeError(f"host reverify snapshot differs: {relative}")
        hashes[relative] = sha(source)
    return target, hashes


def remote_run(key: str) -> tuple[dict, int]:
    try:
        result = site_remote.ssh(REMOTE_PYTHON, str(REMOTE_WORKER), key, timeout=1800)
        raw, code = result.stdout, result.returncode
    except subprocess.CalledProcessError as exc:
        raw, code = exc.stdout or "", exc.returncode
    return site_remote._json_receipt(raw), code


def pull_attempt(run: Path, key: str, summary: dict) -> Path:
    remote = Path(summary.get("attempt_dir") or "")
    expected_parent = REMOTE_RUNS / run.name / "phase3_reverify_attempts"
    if (remote.parent != expected_parent
            or not re.fullmatch(r"[0-9]{8}T[0-9]{6}\.[0-9]{6}\+0000_[0-9]+", remote.name)):
        raise RuntimeError("remote reverify attempt path outside exact run")
    local = run / "phase3_reverify_attempts" / remote.name
    if local.exists() or local.is_symlink():
        raise RuntimeError("reverify attempt local destination already exists")
    site_remote.pull_tree(remote, local)
    receipt = local / "receipt.json"
    if not receipt.is_file() or sha(receipt) != summary.get("receipt_sha256"):
        raise RuntimeError("pulled reverify receipt SHA mismatch")
    return local


def sync_small_post(run: Path, attempt: Path, pre_hashes: dict,
                    post_snapshot: dict, expected_total: float) -> dict:
    sources = attempt / "post"
    for relative in SYNC_FILES:
        source, target = sources / relative, run / relative
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"remote reverify post JSON missing: {relative}")
        if sha(source) != (post_snapshot.get(relative) or {}).get("sha256"):
            raise RuntimeError(f"pulled reverify post differs from trusted receipt: {relative}")
        old_sha = pre_hashes.get(relative)
        if target.is_symlink() or (old_sha is None and target.exists()) or (
                old_sha is not None and (not target.is_file() or sha(target) != old_sha)):
            raise RuntimeError(f"local reverify predecessor changed concurrently: {relative}")
    bench = json.loads((sources / "outputs/benchmark.json").read_text())
    rows = bench.get("per_prompt")
    export = bench.get("export") or {}
    if (not isinstance(rows, list) or len(rows) != 5
            or bench.get("timing_scope") !=
            "five_prompt_sum_sglang_request_metrics_excluding_model_load_internal_one_step_warmup_and_video_export"
            or export.get("frame_count") != 80 or export.get("per_prompt_frames") != 16
            or len(export.get("videos") or []) != 5
            or abs(float(bench.get("total_s")) - expected_total) > 1e-3
            or abs(sum(float(row["request_s"]) for row in rows) - expected_total) > 1e-3):
        raise RuntimeError("reverify post changed frozen five-request timing")
    verdict = json.loads((sources / "reverify_verdict.json").read_text())
    if not math.isfinite(float(verdict.get("lpips_max"))):
        raise RuntimeError("reverify post lacks finite original lpips_max")
    copied = {}
    for relative in SYNC_FILES:
        source, target = sources / relative, run / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha(target) != sha(source):
            raise RuntimeError(f"reverify post sync SHA mismatch: {relative}")
        copied[relative] = sha(target)
    return copied


def assess(run: Path) -> dict:
    assert_approved(run)
    # Empty approval map and unpinned worker SHA make this staged-only. A real
    # DELIVERY point must be reviewed before either can be activated.
    spec = APPROVED_POINTS[run.name]
    if (run / "reverify_verdict.json").exists() or (run / "reverify_verdict.json").is_symlink():
        raise RuntimeError("reverify fixed output not cleared by approved pre-call guard")
    source = trusted_remote_worker()
    attempt_host, pre_hashes = snapshot_local(run)
    for relative, expected in spec["predecessor_sha256"].items():
        if pre_hashes.get(relative) != expected:
            raise RuntimeError(f"reverify host pre-snapshot differs from approval: {relative}")
    started, start_ns = utc(), time.monotonic_ns()
    receipt = {"status": "transport_failed", "run_dir": str(run),
               "started_at_utc": started, "trusted_source": source,
               "host_pre_sha256": pre_hashes, "delivery_sha256": spec["delivery_sha256"],
               "gpu_authorization": "not_granted_by_reverify_route"}
    try:
        summary, code = remote_run(spec["run_key"])
        receipt["remote_exit_code"] = code
        receipt["remote_summary"] = summary
        pulled = pull_attempt(run, spec["run_key"], summary)
        receipt["local_attempt_dir"] = str(pulled)
        remote_receipt = json.loads((pulled / "receipt.json").read_text())
        if (code != 0 or summary.get("status") != "original_remote_reverify_complete"
                or remote_receipt.get("status") != "original_remote_reverify_complete"
                or remote_receipt.get("run_key") != spec["run_key"]
                or remote_receipt.get("blocked_network_attempts")
                or int(remote_receipt.get("plan_eval", {}).get("exit_code", -1)) != 0
                or int(remote_receipt.get("collect", {}).get("exit_code", -1)) != 0):
            raise RuntimeError("trusted remote original reverify failed")
        receipt["host_post_sha256"] = sync_small_post(run, pulled, pre_hashes,
                                                      remote_receipt.get("post_snapshot") or {},
                                                      float(spec["total_s"]))
        receipt["original_plan_eval_exit_code"] = int(remote_receipt["plan_eval"]["exit_code"])
        stdout = pulled / "plan_eval.stdout.log"
        if not stdout.is_file() or stdout.is_symlink():
            raise RuntimeError("original plan_eval stdout receipt missing")
        receipt["original_plan_eval_stdout"] = stdout.read_text()
        receipt["status"] = "original_remote_reverify_complete"
    except Exception as exc:
        receipt["error_type"] = type(exc).__name__
        receipt["error"] = str(exc)
    receipt["finished_at_utc"] = utc()
    receipt["host_wall_s"] = (time.monotonic_ns() - start_ns) / 1e9
    path = attempt_host / "transport_receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt["host_receipt"] = str(path)
    receipt["host_receipt_sha256"] = sha(path)
    if receipt["status"] != "original_remote_reverify_complete":
        raise RuntimeError(f"typed CPU reverify failed; see {path}")
    return receipt
