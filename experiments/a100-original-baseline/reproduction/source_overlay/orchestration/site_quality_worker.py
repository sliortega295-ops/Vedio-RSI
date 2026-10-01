#!/usr/bin/env python3
"""Trusted remote CPU transport for the archived Cache quality tools.

Only the eight exact reviewed seq0002 Cache runs are admitted. This shim supplies an
offline evaluation interpreter and records timing/provenance; the collector,
LPIPS judge, plan_eval, and Master/Executor quality decisions remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import shutil
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path("/share/project/shuchenweng/yongyan_liu/Experiments/SolAgent/20260913-a100-original-baseline")
REPO = PROJECT / "repo"
CACHE = REPO / "output/experiments/sana2ba100-cache_ca-0002/worktree"
DENSE = REPO / "runs/20260913-091006-sana_video_2b_a100_dense"
EVAL_PYTHON = Path("/tmp/distillschedule-helios-official-torch210/bin/python")
EVAL_PYTHON_REAL = Path("/tmp/distillschedule-helios-official-torch210/bin/python3.10")
EVAL_PYTHON_SHA256 = "7c9e72ea1f06138ed5f52f49894c62f29e51e302f0d02f8fb18490e90e994331"
COLLECTOR = REPO / "scripts/collect_run.py"
PLAN_EVAL = REPO / "search/plan_eval.py"
JUDGE = REPO / "tools/vision/lpips_judge.py"
GUARD = REPO / "orchestration/site_quality_guard/sitecustomize.py"
LPIPS_TARGET = PROJECT / "deps/phase3/lpips_target"
TORCH_HOME = PROJECT / "deps/phase3/torch_home"
CHECKPOINT = TORCH_HOME / "hub/checkpoints/alexnet-owt-7be5be79.pth"
LINEAR = LPIPS_TARGET / "lpips/weights/v0.1/alex.pth"
WHEEL = PROJECT / "deps/phase3/lpips_wheel/lpips-0.1.4-py3-none-any.whl"
FRAME_SCOPE = "five_prompt_sum_sglang_request_metrics_excluding_model_load_internal_one_step_warmup_and_video_export"
DENSE_FRAME_DIGEST = "728ddacbaa5c9656cf7952e60c835192de70f64a5db8c903f192fd148ed4c964"
FIXED_FILES = {
    COLLECTOR: "34d85bf04eb5ee603c6c05964d0137a426c6710e9cd828ae04b54b66c05e158f",
    PLAN_EVAL: "a06d9e8f845cd269cd80f3b085c5c03675129f145801ed77e9fe1e5d48ad4840",
    JUDGE: "f68f7c72f31ec8018176e7b9be7047eba97524b73053842d99ba69ed3e1570bc",
    GUARD: "f2d1a27030f0456380f410d4bb7e118e628d543730718f74c038ddf292e71fe2",
    CHECKPOINT: "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
    LINEAR: "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0",
    WHEEL: "fd537af5828b69d2e6ffc0a397bd506dbc28ca183543617690844c08e102ec5e",
}
RUNS = {
    "R01": {
        "name": "20260913-095257-cache_ca_r01_easycache_t010",
        "benchmark_before_sha256": "ecf0598168343475536d7929042dcc2ae123b1df708ed66fb8b7d5a918353f1f",
        "run_config_sha256": "f55df5ed499e978867c2f0398028a6e73765d5fdd9df24adb15f7bb6439fda73",
        "frames_digest": "ebebb661eed466d9668fd1a39e3ffc8f855bcb3bd70743f1f7701d21a7a4828c",
        "total_s": 375.18,
    },
    "R02": {
        "name": "20260913-102721-cache_ca_r02_teacache_t010",
        "benchmark_before_sha256": "e7cf36c4ac5b5b3880a991aa653ab8a8dddf90a2c75cacff5e3314b5933a1579",
        "run_config_sha256": "09b3395fd9ca0e0a351e33c9327d0bed2a96ab70f04613fefbb57503cbc4438c",
        "frames_digest": "e427746e6aaf5d3d512ba4a7cc73789b3161025dc5890d3791fd1beefe45df37",
        "total_s": 251.95,
    },
    "R03": {
        "name": "20260914-045221-cache_ca_r03_teacache_t0016",
        "benchmark_before_sha256": "f181fa6493ccebf46d22b65748a9284061d29b6c2d14a6ecf536c800f617926c",
        "run_config_sha256": "d6c90681de8062cf937b468d6db6d6f9ee221a77f2b4b6914a185ba31cad8e90",
        "frames_digest": "d833d5685e177d565937409d90416d90e733d89b68f3b678617c303712a80f52",
        "total_s": 370.52,
    },
    "R04": {
        "name": "20260914-051802-cache_ca_r04_taylorseer_t0040_o1",
        "benchmark_before_sha256": "cb36ffc54c66296eabee6a7edc482ceb26101113161d644fc1ae8622feb9bc78",
        "run_config_sha256": "1a3f75560f3c178358e3cb1859b893120e66b510907fd9275400745e5149dfd1",
        "frames_digest": "f28e2c0f3e63df316b354afd0d318172730024ff09b49fec100b336d1ddefa39",
        "total_s": 382.27,
    },
    "R05": {
        "name": "20260914-054324-cache_ca_r05_taylorseer_t0180_w10_o1",
        "benchmark_before_sha256": "e88373b0bd6d438f3efe453b0b4a605e5fe9ceb944fa41a4665ff5d30ff3d323",
        "run_config_sha256": "0c98e68c6aa33fbb2a6cc1f3d0041108206439efb915af97557aa92ebcfded51",
        "frames_digest": "36b9df309d8b3ab3ca47c3e82b42bf3c2da002af493336ac01702d83b6f1e3d0",
        "total_s": 371.36,
    },
    "R06": {
        "name": "20260914-063907-cache_ca_r06_easycache_t010_w30",
        "benchmark_before_sha256": "2dfedb8ad1a2fb4ed68729a8fe3c42adcc1cb66309d5c3c69e5d5a98a5787972",
        "run_config_sha256": "6f32cda5b35e547e6d66e39b1b46e892526940aaa5f1619fa4b44d4de7e02880",
        "frames_digest": "4c9f96ea9e749688ea335e705731b3b41c7460313902edae800824fdd670717c",
        "total_s": 480.37,
    },
    "R07": {
        "name": "20260914-072228-cache_ca_r07_teacache_t00087",
        "benchmark_before_sha256": "3f582161803def77d253506d8381dc1f0a6ed5c966de254d4dd0a5247dbac691",
        "run_config_sha256": "cdf42836a2de2044f42409fbe94fe9ea1990e73eeb6050d67a8dbad19a93a3e8",
        "frames_digest": "46faed6d9430a46aad6c7986cd882c55673d7b07e0fd81c018204e3b91cc4b54",
        "total_s": 480.18,
    },
    "R08": {
        "name": "20260914-083326-cache_ca_r08_taylorseer_t0180_w30_o1",
        "benchmark_before_sha256": "ffd8260d2ee47f0ad9b573f4ffa51bd96867f12aa0b36a1fa20bc78775d672d8",
        "run_config_sha256": "2eac08b00536f71a6f0d5359f91114d198f7cd71c496bccad18a6d37c54d0989",
        "frames_digest": "586be1f2c9c97bbb2f87ebd43e8d7bae9e2889b7db6680f6bc62ab9b85bab88c",
        "remote_bundle_sha256": {
            "launch.sh": "6c8f610c4e1f6a0085cde0bb56f7b80f0133f57725f8d6375d2f54644823271f",
            "manifest.resolved.toml": "7b77a2d868e1257da3635901b46c8ae552c762b3e10532d555e700ca39c1ece9",
            "metadata.json": "81d36ecd5df527ce05abec4301983217dc418e01751eb41413b4999a0b09a461",
        },
        "total_s": 480.96,
    },
}
SNAPSHOT_FILES = (
    "metadata.json", "outputs/benchmark.json", "outputs/quality.json",
    "outputs/collection.json", "assess_verdict.json",
)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def json_read(path: Path) -> dict:
    return json.loads(path.read_text())


def frames_digest(directory: Path) -> tuple[str, list[Path]]:
    frames = sorted(directory.glob("f_*.png"))
    expected = [f"f_{index:03d}.png" for index in range(1, 81)]
    if [path.name for path in frames] != expected or any(path.is_symlink() for path in frames):
        raise RuntimeError(f"five-prompt frame set is not exactly f_001..f_080: {directory}")
    ordered_hashes = "".join(sha256(frame) + "\n" for frame in frames)
    return hashlib.sha256(ordered_hashes.encode()).hexdigest(), frames


def check_benchmark(path: Path, expected_total: float) -> dict:
    bench = json_read(path)
    prompts = bench.get("per_prompt")
    export = bench.get("export") or {}
    if (not isinstance(prompts, list) or len(prompts) != 5
            or bench.get("timing_scope") != FRAME_SCOPE
            or export.get("frame_count") != 80 or export.get("per_prompt_frames") != 16
            or len(export.get("videos") or []) != 5):
        raise RuntimeError("five-prompt benchmark/export contract mismatch")
    total = float(bench.get("total_s"))
    summed = sum(float(row["request_s"]) for row in prompts)
    if abs(total - summed) > 1e-5 or abs(total - expected_total) > 1e-3:
        raise RuntimeError(f"benchmark total/per_prompt mismatch: {total}, {summed}")
    return {"total_s": total, "sum_request_s": summed, "timing_scope": bench["timing_scope"],
            "prompt_count": len(prompts), "per_prompt_frames": export["per_prompt_frames"]}


def preflight(run: Path, spec: dict) -> dict:
    if run.is_symlink() or run.resolve().parent != CACHE / "runs" or run.name != spec["name"]:
        raise RuntimeError("quality run path is not an exact seq0002 Cache run")
    files = {}
    for path, expected in FIXED_FILES.items():
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            raise RuntimeError(f"trusted source/dependency SHA mismatch: {path}")
        files[str(path)] = {"sha256": expected, "size_bytes": path.stat().st_size}
    if (not EVAL_PYTHON.is_file() or not EVAL_PYTHON.is_symlink()
            or os.readlink(EVAL_PYTHON) != "python3.10"
            or EVAL_PYTHON.resolve() != EVAL_PYTHON_REAL
            or not EVAL_PYTHON_REAL.is_file() or EVAL_PYTHON_REAL.is_symlink()
            or sha256(EVAL_PYTHON_REAL) != EVAL_PYTHON_SHA256):
        raise RuntimeError("fixed evaluation Python is unavailable")
    files[str(EVAL_PYTHON_REAL)] = {"sha256": EVAL_PYTHON_SHA256,
                                   "size_bytes": EVAL_PYTHON_REAL.stat().st_size,
                                   "venv_link": str(EVAL_PYTHON)}
    for relative, expected in (("outputs/benchmark.json", spec["benchmark_before_sha256"]),
                               ("outputs/run_config.json", spec["run_config_sha256"])):
        path = run / relative
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            raise RuntimeError(f"preexisting real run provenance changed: {relative}")
        files[str(path)] = {"sha256": expected, "size_bytes": path.stat().st_size}
    metadata = json_read(run / "metadata.json")
    if metadata.get("config_id") != spec["name"].split("-", 2)[-1]:
        raise RuntimeError("real run metadata config ID mismatch")
    for relative, expected in spec.get("remote_bundle_sha256", {}).items():
        path = run / relative
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            raise RuntimeError(f"real remote run bundle SHA mismatch: {relative}")
        files[str(path)] = {"sha256": expected, "size_bytes": path.stat().st_size}
    bench = check_benchmark(run / "outputs/benchmark.json", spec["total_s"])
    dense_digest, dense_frames = frames_digest(DENSE / "outputs/frames")
    candidate_digest, candidate_frames = frames_digest(run / "outputs/frames")
    if dense_digest != DENSE_FRAME_DIGEST or candidate_digest != spec["frames_digest"]:
        raise RuntimeError("aligned baseline/config frame contents changed")
    for entry in json_read(run / "outputs/benchmark.json")["export"]["videos"]:
        idx = entry.get("prompt_index")
        if idx not in range(1, 6):
            raise RuntimeError("export video prompt index invalid")
        video = run / "outputs" / ("out.mp4" if idx == 1 else f"out_{idx}.mp4")
        if not video.is_file() or video.is_symlink() or sha256(video) != entry.get("sha256"):
            raise RuntimeError(f"real run video SHA mismatch: p{idx:02d}")
    return {"files": files, "benchmark": bench,
            "dense_frames_digest": dense_digest, "candidate_frames_digest": candidate_digest,
            "baseline_frame_count": len(dense_frames), "candidate_frame_count": len(candidate_frames)}


def snapshot(run: Path, destination: Path) -> dict:
    hashes: dict[str, dict] = {}
    for relative in SNAPSHOT_FILES:
        source = run / relative
        if not source.is_file():
            continue
        if source.is_symlink():
            raise RuntimeError(f"quality snapshot source is a symlink: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        before, after = sha256(source), sha256(target)
        if before != after:
            raise RuntimeError(f"quality snapshot copy differs: {relative}")
        hashes[relative] = {"sha256": before, "size_bytes": source.stat().st_size}
    return hashes


def child_usage() -> tuple[float, float]:
    value = resource.getrusage(resource.RUSAGE_CHILDREN)
    return value.ru_utime, value.ru_stime


def invoke(name: str, argv: list[str], env: dict, attempt: Path, timeout: int) -> dict:
    started_utc, start_ns = utc(), time.monotonic_ns()
    before_user, before_system = child_usage()
    timed_out = False
    try:
        result = subprocess.run(argv, cwd=REPO, env=env, capture_output=True,
                                text=True, timeout=timeout, check=False)
        stdout, stderr, returncode = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        returncode = 124
    after_user, after_system = child_usage()
    (attempt / f"{name}.stdout.log").write_text(stdout)
    (attempt / f"{name}.stderr.log").write_text(stderr)
    return {"command": argv, "started_at_utc": started_utc, "finished_at_utc": utc(),
            "wall_s": (time.monotonic_ns() - start_ns) / 1e9,
            "child_user_s_lower_bound": after_user - before_user,
            "child_system_s_lower_bound": after_system - before_system,
            "exit_code": returncode, "timed_out": timed_out,
            "stdout_sha256": sha256(attempt / f"{name}.stdout.log"),
            "stderr_sha256": sha256(attempt / f"{name}.stderr.log")}


def pair_audit(run: Path, quality: dict) -> dict:
    spec = importlib.util.spec_from_file_location("phase3_trusted_collector", COLLECTOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the pinned archived collector pair selector")
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    baseline = sorted((DENSE / "outputs/frames").glob("f_*.png"))
    config = sorted((run / "outputs/frames").glob("f_*.png"))
    pairs = collector.select_stratified_and_worst_pairs(
        collector.paired_frames(config, [str(path) for path in baseline]),
        stratified_limit=collector.LPIPS_STRATIFIED_PAIRS,
        worst_case_limit=collector.LPIPS_WORST_CASE_PAIRS,
        total_limit=collector.LPIPS_MAX_PAIRS,
    )
    judge = ((quality.get("judges") or {}).get("lpips") or {})
    payload = judge.get("result") or {}
    scores = payload.get("per_frame") or []
    if (judge.get("status") != "complete" or payload.get("status") != "ok"
            or len(scores) != len(pairs) or payload.get("n") != len(pairs)):
        raise RuntimeError("original collector LPIPS result/pair count mismatch")
    rows = []
    by_prompt: dict[int, list[float]] = {index: [] for index in range(1, 6)}
    for (base, cand), value in zip(pairs, scores):
        base_index = int(base.stem.split("_")[-1])
        cand_index = int(cand.stem.split("_")[-1])
        if base_index != cand_index:
            raise RuntimeError("LPIPS pair crossed prompt/time index")
        prompt = (cand_index - 1) // 16 + 1
        score = float(value)
        if not math.isfinite(score):
            raise RuntimeError("LPIPS returned a nonfinite score")
        rows.append({"frame_index": cand_index, "prompt_index": prompt,
                     "baseline": str(base), "config": str(cand), "lpips": score})
        by_prompt[prompt].append(score)
    if any(not values for values in by_prompt.values()):
        raise RuntimeError("stratified LPIPS did not cover all five prompts")
    return {"available_aligned_pairs": 80, "selected_pair_count": len(rows),
            "selection": "archived select_stratified_and_worst_pairs(32,16,48)",
            "selected_pairs": rows,
            "per_prompt": {f"p{idx:02d}": {"n": len(values), "mean": statistics.fmean(values),
                                       "max": max(values)} for idx, values in by_prompt.items()}}


def evaluation_env(attempt: Path, key: str) -> dict:
    env = os.environ.copy()
    for name in list(env):
        if ("PROXY" in name.upper() or name.upper().endswith("API_KEY")
                or name.upper().startswith("PIP_INDEX")):
            env.pop(name, None)
    temp = Path("/tmp/lyy-experiments/SolAgent") / f"phase3-quality-{key.lower()}"
    temp.mkdir(parents=True, exist_ok=True)
    env.update({
        "PYTHONPATH": os.pathsep.join((str(GUARD.parent), str(LPIPS_TARGET))),
        "TORCH_HOME": str(TORCH_HOME), "TMPDIR": str(temp),
        "SOL_PHASE3_NETWORK_AUDIT": str(attempt / "network_audit.jsonl"),
        "CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "PIP_NO_INDEX": "1", "PIP_CONFIG_FILE": "/dev/null",
    })
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_key", choices=tuple(RUNS))
    args = parser.parse_args()
    key, expected = args.run_key, RUNS[args.run_key]
    run = CACHE / "runs" / expected["name"]
    start_ns, start_cpu_ns, started_utc = time.monotonic_ns(), time.process_time_ns(), utc()
    attempt = run / "phase3_quality_attempts" / (started_utc.replace(":", "").replace("-", "")
                                                   .replace("+00:00", "Z") + f"_{os.getpid()}")
    receipt: dict[str, object] = {"schema_version": 1, "kind": "phase3_typed_cpu_quality_transport",
                                  "run_key": key, "run_dir": str(run), "host": socket.gethostname(),
                                  "started_at_utc": started_utc, "started_monotonic_ns": start_ns,
                                  "status": "failed", "site_transport_not_quality_decision": True}
    try:
        started = time.monotonic_ns()
        receipt["preflight"] = preflight(run, expected)
        receipt["preflight_s"] = (time.monotonic_ns() - started) / 1e9
        attempt.mkdir(parents=True, exist_ok=False)
        receipt["pre_snapshot"] = snapshot(run, attempt / "pre")
        env = evaluation_env(attempt, key)
        collector_cmd = [str(EVAL_PYTHON), str(COLLECTOR), str(run),
                         "--baseline-run-dir", str(DENSE)]
        receipt["collect"] = invoke("collect", collector_cmd, env, attempt, timeout=1800)
        if receipt["collect"]["exit_code"] != 0:
            raise RuntimeError("archived collector failed; see preserved stdout/stderr")
        if frames_digest(run / "outputs/frames")[0] != expected["frames_digest"]:
            raise RuntimeError("archived collector changed the five-prompt frame set")
        check_benchmark(run / "outputs/benchmark.json", expected["total_s"])
        plan_cmd = [str(EVAL_PYTHON), str(PLAN_EVAL), "--model", "sana_video_2b_a100",
                    "--no-gemini", "--no-refresh-collection", "--assess", str(run),
                    "--baseline-frames", str(DENSE / "outputs/frames"),
                    "--out", str(run / "assess_verdict.json")]
        receipt["plan_eval"] = invoke("plan_eval", plan_cmd, env, attempt, timeout=300)
        if receipt["plan_eval"]["exit_code"] != 0:
            raise RuntimeError("archived no-Gemini plan_eval failed; see preserved stdout/stderr")
        quality = json_read(run / "outputs/quality.json")
        receipt["quality_status"] = quality.get("status")
        receipt["quality_blockers"] = quality.get("promotion_blockers")
        receipt["lpips_pair_audit"] = pair_audit(run, quality)
        receipt["assess_verdict"] = json_read(run / "assess_verdict.json")
        audit_path = attempt / "network_audit.jsonl"
        events = [json.loads(line) for line in audit_path.read_text().splitlines()] if audit_path.is_file() else []
        receipt["network_guard_loaded_pids"] = sorted({row["pid"] for row in events
                                                       if row.get("kind") == "guard_loaded"})
        receipt["blocked_network_attempts"] = [row for row in events
                                               if row.get("kind", "").startswith("blocked_")]
        if len(receipt["network_guard_loaded_pids"]) < 3 or receipt["blocked_network_attempts"]:
            raise RuntimeError("offline guard missing from a subprocess or a network attempt occurred")
        receipt["status"] = "lpips_complete_visual_and_other_quality_gates_pending"
    except Exception as exc:
        receipt["error_type"] = type(exc).__name__
        receipt["error"] = str(exc)
    finally:
        if attempt.is_dir():
            try:
                receipt["post_snapshot"] = snapshot(run, attempt / "post")
            except Exception as exc:
                receipt["status"] = "failed"
                receipt["post_snapshot_error_type"] = type(exc).__name__
                receipt["post_snapshot_error"] = str(exc)
        receipt["finished_at_utc"] = utc()
        receipt["finished_monotonic_ns"] = time.monotonic_ns()
        receipt["wall_s"] = (receipt["finished_monotonic_ns"] - start_ns) / 1e9
        receipt["worker_process_cpu_s"] = (time.process_time_ns() - start_cpu_ns) / 1e9
        if attempt.is_dir():
            path = attempt / "receipt.json"
            try:
                path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
                receipt_sha = sha256(path)
            except Exception as exc:
                receipt["status"] = "failed"
                receipt["receipt_write_error"] = f"{type(exc).__name__}: {exc}"
                receipt_sha = None
        else:
            receipt_sha = None
        print(json.dumps({"status": receipt["status"], "run_key": key,
                          "attempt_dir": str(attempt) if attempt.is_dir() else None,
                          "receipt_sha256": receipt_sha, "wall_s": receipt["wall_s"],
                          "quality_status": receipt.get("quality_status"),
                          "quality_blockers": receipt.get("quality_blockers"),
                          "lpips_selected_n": (receipt.get("lpips_pair_audit") or {}).get("selected_pair_count"),
                          "error": receipt.get("error")}, sort_keys=True))
    return 0 if receipt["status"] == "lpips_complete_visual_and_other_quality_gates_pending" else 2


if __name__ == "__main__":
    raise SystemExit(main())
