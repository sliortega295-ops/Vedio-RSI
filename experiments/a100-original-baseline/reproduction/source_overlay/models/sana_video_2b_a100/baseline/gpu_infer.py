#!/usr/bin/env python3
"""Site-only five-prompt SGLang runner for the archived Sol-Agent Harness.

This is a transport/workload adapter, not an optimization recipe. Each of the
five unchanged prompts is generated sequentially on one leased GPU. The
official Harness still owns candidate selection, collection, quality gates,
and Master integration.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PROJECT = Path("/share/project/shuchenweng/yongyan_liu/Experiments/SolAgent/20260913-a100-original-baseline")
MODEL = Path("/share/project/shuchenweng/yongyan_liu/Experiments/SolRolloutBench/20260912-a100-eight/models/sana-video-2b/db5f398b13ca086d09a50ce156c20527773841b1")
OVERLAY = Path("/share/project/shuchenweng/yongyan_liu/Experiments/SolRolloutBench/20260912-a100-eight/envs/sglang-overlay")
PROMPTS = REPO / "models/sana_video_2b_a100/prompts/validation_first5.txt"
PROMPTS_SHA256 = "de249238cd19dd5b77de61eb69fd114228d4813a0335a1c794e366bbcd00e7e0"
MODEL_REVISION = "db5f398b13ca086d09a50ce156c20527773841b1"
TIMING_SCOPE = "five_prompt_sum_sglang_request_metrics_excluding_model_load_internal_one_step_warmup_and_video_export"
GPU_UUIDS = (
    "GPU-cad55364-e014-c35f-337e-db899afd1c6a",
    "GPU-a904c6f6-6410-dc0f-c7c3-efc068004617",
    "GPU-79ab3e5e-e382-1f9c-bede-1900666e9283",
    "GPU-bab43db0-a051-0419-39f7-e33effa9445b",
    "GPU-44bddf94-7411-b3ad-f195-fdbdd12fa73e",
    "GPU-af8dba1f-f74e-6a55-bb9e-6c8961edbd6e",
    "GPU-5edaa614-d5f8-8a3f-11ec-b128f8d029f3",
    "GPU-01fff32a-1dd6-93a7-1aae-60d378382961",
)
EVENT_LOCK = threading.Lock()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def event(name: str, **fields: object) -> None:
    phase = os.environ.get("SOL_A100_BUDGET_PHASE", "phase1")
    if phase not in ("phase1", "phase2"):
        raise RuntimeError("unknown site GPU budget phase")
    path = PROJECT / ("telemetry/phase2/events.jsonl" if phase == "phase2"
                      else "telemetry/events.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "event": name, "utc": utc(), "host": socket.gethostname(),
        "monotonic_ns": time.monotonic_ns(), **fields,
    }
    with EVENT_LOCK, path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        fcntl.flock(stream, fcntl.LOCK_UN)


def check_workload(out_dir: Path) -> list[str]:
    if REPO.resolve() == Path("/home/lyy/Experiments/SolAgent/20260913-a100-original-baseline/repo"):
        raise RuntimeError("SGLang runner must execute on the authorized remote A100 node")
    if not out_dir.is_dir() or out_dir.is_symlink() or not out_dir.resolve().is_relative_to(PROJECT / "repo"):
        raise RuntimeError("OUT_DIR must be a real directory in the new remote project")
    if not MODEL.is_dir() or MODEL.is_symlink() or not OVERLAY.is_dir() or OVERLAY.is_symlink():
        raise RuntimeError("pinned read-only model snapshot or dependency overlay missing")
    if MODEL.name != MODEL_REVISION or sha256(PROMPTS) != PROMPTS_SHA256:
        raise RuntimeError("model revision or five-prompt file differs from frozen workload")
    prompts = [line.strip() for line in PROMPTS.read_text().splitlines() if line.strip()]
    if len(prompts) != 5 or os.environ.get("SOL_A100_SEED") != "42":
        raise RuntimeError("five-prompt / seed=42 workload changed")
    expected_env = {
        "SOL_A100_MODEL_PATH": str(MODEL),
        "SOL_A100_DEP_OVERLAY": str(OVERLAY),
        "SOL_A100_PROMPTS_FILE": "models/sana_video_2b_a100/prompts/validation_first5.txt",
        "SOL_A100_EXPECTED_PROMPTS": "5",
        "SOL_A100_SGL_KERNEL_BACKEND": "torch_fallback",
    }
    if any(os.environ.get(key) != value for key, value in expected_env.items()):
        raise RuntimeError("site model/dependency/prompt contract env differs")
    if os.environ.get("AUTOVIDEO_CONFIG_ID") == "sana_video_2b_a100_dense":
        off = {
            "SGLANG_SANA_LINATTN_BF16": "0",
            "SGLANG_SANA_QKV_MERGE": "0",
            "SGLANG_SANA_EASYCACHE_THRESH": "0",
        }
        if any(os.environ.get(key) != value for key, value in off.items()) or os.environ.get("SOL_A100_COMPILE") == "1":
            raise RuntimeError("dense baseline has an optimization enabled")
    return prompts


def gpu_state() -> tuple[list[str], set[str]]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True, capture_output=True, check=True, timeout=15,
    )
    observed = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if observed != list(GPU_UUIDS):
        raise RuntimeError(f"A100 UUID mapping changed: {observed}")
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True, capture_output=True, check=True, timeout=15,
    )
    busy = {line.split(",", 1)[0].strip() for line in apps.stdout.splitlines() if line.strip()}
    return observed, busy


def reserve_one(out_dir: Path) -> tuple[int, object]:
    lock_dir = PROJECT / "state/gpu"
    lock_dir.mkdir(parents=True, exist_ok=True)
    begin = time.monotonic()
    event("gpu_queue_begin", run_dir=str(out_dir.parent), requested=1)
    while time.monotonic() - begin < 1800:
        _, busy = gpu_state()
        for index, uuid in enumerate(GPU_UUIDS):
            if uuid in busy:
                continue
            lock = (lock_dir / f"slot-{index}.lock").open("a+")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            _, fresh_busy = gpu_state()
            if uuid not in fresh_busy:
                event("gpu_queue_end", run_dir=str(out_dir.parent), wait_s=time.monotonic() - begin,
                      slots=[index], uuids=[uuid])
                return index, lock
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
        time.sleep(2)
    raise TimeoutError("no idle authorized A100 GPU available within 30 minutes")


def first_gpu_budget() -> dict:
    phase = os.environ.get("SOL_A100_BUDGET_PHASE", "phase1")
    if phase == "phase2":
        path = PROJECT / "state/phase2_first_gpu_job.json"
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("trusted phase2 first-GPU budget receipt is absent")
        budget = json.loads(path.read_text())
        first = budget.get("first_gpu_unix_ns")
        deadline = budget.get("deadline_unix_ns")
        if (budget.get("schema_version") != 1 or budget.get("phase") != "phase2"
                or budget.get("host") != socket.gethostname()
                or not isinstance(first, int) or not isinstance(deadline, int)
                or deadline != first + 6 * 3600 * 10**9):
            raise RuntimeError("trusted phase2 first-GPU budget receipt differs")
        if time.time_ns() >= deadline:
            raise TimeoutError("phase2 six-hour GPU budget expired")
        return budget
    if phase != "phase1":
        raise RuntimeError("unknown site GPU budget phase")
    path = PROJECT / "state/first_gpu_job.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = PROJECT / "state/first_gpu_job.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            budget = json.loads(path.read_text())
        else:
            now = time.time_ns()
            budget = {
                "schema_version": 1, "first_gpu_utc": utc(),
                "first_gpu_unix_ns": now, "first_gpu_monotonic_ns": time.monotonic_ns(),
                "host": socket.gethostname(), "deadline_unix_ns": now + 6 * 3600 * 10**9,
                "limit_s": 6 * 3600,
            }
            write_json(path, budget)
        fcntl.flock(lock, fcntl.LOCK_UN)
    if time.time_ns() >= budget["deadline_unix_ns"]:
        raise TimeoutError("first-execution six-hour GPU budget expired")
    return budget


def child_env(uuid: str, index: int) -> dict[str, str]:
    env = os.environ.copy()
    cache = PROJECT / "cache"
    for item in (cache / "huggingface/hub", cache / "xdg", cache / "triton", cache / "torch_inductor", cache / "tmp"):
        item.mkdir(parents=True, exist_ok=True)
    env.update({
        "CUDA_VISIBLE_DEVICES": uuid, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "SOL_A100_GPU_OPT_IN": "1", "SOL_A100_PINNED_GPU_INDEX": str(index),
        "SOL_A100_SGL_KERNEL_BACKEND": "torch_fallback", "SGLANG_SANA_MINIMAL_IMPORT": "1",
        "PYTHONPATH": ":".join((str(REPO / "models/sana_video_2b_a100/compat"),
                                  str(source_root() / "python"), str(OVERLAY))),
        "HF_HOME": str(cache / "huggingface"), "HF_HUB_CACHE": str(cache / "huggingface/hub"),
        "XDG_CACHE_HOME": str(cache / "xdg"), "TRITON_CACHE_DIR": str(cache / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "torch_inductor"), "TMPDIR": str(cache / "tmp"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    python_lib = str(Path(os.environ["SOL_A100_PYTHON"]).parent.parent / "lib")
    env["LD_LIBRARY_PATH"] = python_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    return env


def command(prompt: str, prompt_index: int, slot: int, prompt_dir: Path) -> list[str]:
    base = 41000 + 1000 * slot
    runner = source_root() / "scripts/sana/sana_video_sglang_run.py"
    cmd = [os.environ["SOL_A100_PYTHON"], str(Path(__file__).with_name("a100_exec.py")),
           "--target", str(runner), "--output-dir", str(prompt_dir), "--gpu-index", str(slot),
           "--port", str(base + 500), "--master-port", str(base + 800),
           "--scheduler-port", str(base), "--nccl-port", str(base + 100), "--",
           "--model", str(MODEL), "--prompt", prompt, "--frames", "81", "--steps", "50",
           "--height", "480", "--width", "832", "--guidance-scale", "6.0",
           "--seed", "42", "--output", "out"]
    if os.environ.get("SOL_A100_COMPILE") == "1":
        cmd.append("--compile")
    if os.environ.get("SGLANG_SANA_LINATTN_BF16") == "1":
        cmd.append("--linattn-bf16")
    if os.environ.get("SGLANG_SANA_QKV_MERGE") == "1":
        cmd.append("--qkv-merge")
    threshold = float(os.environ.get("SGLANG_SANA_EASYCACHE_THRESH", "0"))
    if threshold > 0:
        cmd.extend(["--easycache", str(threshold), "--ec-warmup",
                    os.environ.get("SGLANG_SANA_EASYCACHE_WARMUP", "3"),
                    "--ec-subsample", os.environ.get("SGLANG_SANA_EASYCACHE_SUBSAMPLE", "8")])
    return cmd


def inspect_video(video: Path) -> dict:
    output = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(video)],
        text=True, capture_output=True, check=True, timeout=90,
    )
    stream = json.loads(output.stdout)["streams"][0]
    valid = (int(stream["width"]) == 832 and int(stream["height"]) == 480
             and int(stream["nb_read_frames"]) == 81
             and Fraction(stream["r_frame_rate"]) == Fraction(16, 1))
    if not valid or video.stat().st_size <= 0:
        raise RuntimeError(f"video structural check failed: {stream}")
    return {"bytes": video.stat().st_size, "sha256": sha256(video), "ffprobe_stream": stream}


def run_one(prompt_index: int, prompt: str, slot: int, out_dir: Path) -> dict:
    prompt_dir = out_dir / "prompts" / f"p{prompt_index:02d}"
    prompt_dir.mkdir(parents=True, exist_ok=False)
    cmd = command(prompt, prompt_index, slot, prompt_dir)
    log_path = prompt_dir / "run.log"
    event("gpu_job_launch_begin", run_dir=str(out_dir.parent), prompt_index=prompt_index,
          gpu_uuid=GPU_UUIDS[slot], gpu_index=slot, command=cmd)
    start = time.monotonic()
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=REPO, env=child_env(GPU_UUIDS[slot], slot),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        event("gpu_job_start", run_dir=str(out_dir.parent), prompt_index=prompt_index,
              gpu_uuid=GPU_UUIDS[slot], gpu_index=slot, pid=proc.pid)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if ("STAGE " in line or "Warmed-up request processed" in line
                    or "GENERATE_OK" in line or "finished in" in line):
                event("runtime_log_marker", run_dir=str(out_dir.parent),
                      prompt_index=prompt_index, gpu_uuid=GPU_UUIDS[slot],
                      pid=proc.pid, line=line.strip()[:500])
        rc = proc.wait()
    elapsed = time.monotonic() - start
    event("gpu_job_end", run_dir=str(out_dir.parent), prompt_index=prompt_index,
          gpu_uuid=GPU_UUIDS[slot], gpu_index=slot, pid=proc.pid,
          elapsed_s=elapsed, exit_code=rc)
    wait_gpu_clean(GPU_UUIDS[slot])
    transcript = log_path.read_text(errors="replace")
    request = re.findall(r"Warmed-up request processed in .*?([0-9]+\.[0-9]+).*? seconds \(with warmup excluded\)", transcript)
    generation = re.findall(r"GENERATE_OK in ([0-9.]+)s", transcript)
    stages = {name: float(seconds) for name, seconds in
              re.findall(r"\[([A-Za-z][A-Za-z0-9_]+)\] finished in ([0-9.]+) seconds", transcript)}
    if rc != 0 or len(request) != 1 or len(generation) != 1 or "RUN_COMPLETE_MARKER" not in transcript:
        raise RuntimeError(f"prompt {prompt_index} failed: rc={rc}, request={request}, generate={generation}; see {log_path}")
    videos = sorted(prompt_dir.rglob("*.mp4"))
    if len(videos) != 1:
        raise RuntimeError(f"prompt {prompt_index} produced {len(videos)} MP4s, expected 1: {videos}")
    video = videos[0]
    info = inspect_video(video)
    return {"prompt_index": prompt_index, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "seed": 42, "gpu_index": slot, "gpu_uuid": GPU_UUIDS[slot], "pid": proc.pid,
            "process_wall_s": elapsed, "request_s": float(request[0]),
            "generate_call_s": float(generation[0]), "runtime_stage_s": stages,
            "video": str(video), "video_info": info, "log": str(log_path), "command": cmd}


def source_root() -> Path:
    editable = REPO / "external/sol_runtime"
    if editable.is_dir():
        return editable
    if REPO.resolve() == (PROJECT / "repo").resolve():
        return PROJECT / "source/sol_runtime"
    raise RuntimeError("executor worktree lacks its isolated editable SGLang source")


def wait_gpu_clean(uuid: str) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        _, busy = gpu_state()
        if uuid not in busy:
            return
        time.sleep(1)
    raise RuntimeError(f"GPU still has a compute process after the prompt returned: {uuid}")


def export_five(rows: list[dict], out_dir: Path) -> dict:
    event("export_begin", run_dir=str(out_dir.parent))
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=False)
    exported = []
    for row in rows:
        index = row["prompt_index"]
        video = Path(row["video"])
        dest = out_dir / ("out.mp4" if index == 1 else f"out_{index}.mp4")
        shutil.copy2(video, dest)
        select = "+".join(f"eq(n\\,{round(i * 80 / 15)})" for i in range(16))
        template = frames_dir / f"p{index:02d}_%02d.png"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf",
                        f"select={select}", "-vsync", "0", str(template)],
                       check=True, timeout=180)
        frames = sorted(frames_dir.glob(f"p{index:02d}_*.png"))
        if len(frames) != 16:
            raise RuntimeError(f"prompt {index} yielded {len(frames)} selected frames, expected 16")
        for offset, frame in enumerate(frames, 1):
            frame.rename(frames_dir / f"f_{(index - 1) * 16 + offset:03d}.png")
        exported.append({"prompt_index": index, "video": str(dest), "sha256": sha256(dest)})
    event("export_end", run_dir=str(out_dir.parent), videos=5, frames=80)
    return {"videos": exported, "frame_count": 80, "per_prompt_frames": 16}


def main() -> int:
    out_dir = Path(os.environ["OUT_DIR"]).resolve()
    prompts = check_workload(out_dir)
    event("run_start", run_dir=str(out_dir.parent), config_id=os.environ.get("AUTOVIDEO_CONFIG_ID"))
    started = time.monotonic()
    budget = first_gpu_budget()
    slot, lock = reserve_one(out_dir)
    source = source_root()
    source_hashes = {
        name: sha256(source / name) for name in (
            "scripts/sana/sana_video_sglang_run.py",
            "python/sglang/multimodal_gen/runtime/models/dits/sana_video.py",
            "python/sglang/multimodal_gen/runtime/pipelines/sana_video.py",
        )
    }
    if os.environ.get("AUTOVIDEO_CONFIG_ID") == "sana_video_2b_a100_dense" and source_hashes[
        "python/sglang/multimodal_gen/runtime/models/dits/sana_video.py"
    ] != "6a4c853e8f443734db72befdca0910b4cbf2ccf4a83792c7be03427a36522332":
        raise RuntimeError("dense baseline SGLang transformer differs from clean source")
    run_config = {
        "schema_version": 1, "model_id": "sana_video_2b_a100", "model_path": str(MODEL),
        "model_revision": MODEL_REVISION, "model_manifest_sha256": sha256(Path(__file__).with_name("model_manifest.json")),
        "source_commit": "16af862a135e27d55bf8e574d786db595a8e0654", "source_hashes": source_hashes,
        "workload": {"width": 832, "height": 480, "num_frames": 81, "fps": 16,
                     "num_inference_steps": 50, "guidance_scale": 6.0,
                     "prompt_count": 5, "prompt_sha256": PROMPTS_SHA256, "seed": 42},
        "world_size": 1, "allocated_gpus": 1, "available_gpus": 8,
        "gpu_slots": [slot], "gpu_uuids": [GPU_UUIDS[slot]],
        "support_kernel_backend": "torch_fallback",
        "optimizations": {"compile": os.environ.get("SOL_A100_COMPILE") == "1",
                          "linattn_bf16": os.environ.get("SGLANG_SANA_LINATTN_BF16") == "1",
                          "qkv_merge": os.environ.get("SGLANG_SANA_QKV_MERGE") == "1",
                          "easycache_threshold": float(os.environ.get("SGLANG_SANA_EASYCACHE_THRESH", "0"))},
        "timing_scope": TIMING_SCOPE,
    }
    write_json(out_dir / "run_config.json", run_config)
    try:
        rows = []
        for index, prompt in enumerate(prompts, 1):
            rows.append(run_one(index, prompt, slot, out_dir))
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    rows.sort(key=lambda row: row["prompt_index"])
    exported = export_five(rows, out_dir)
    total_s = sum(row["request_s"] for row in rows)
    denoise_s = sum(value for row in rows for name, value in row["runtime_stage_s"].items()
                    if "denois" in name.lower()) or None
    decode_s = sum(value for row in rows for name, value in row["runtime_stage_s"].items()
                   if "decod" in name.lower() or "vae" in name.lower()) or None
    benchmark = {
        "schema_version": 1, "total_s": total_s, "denoise_s": denoise_s,
        "decode_s": decode_s, "timing_scope": TIMING_SCOPE, "warm_steady_state": True,
        "warmup_requests": 5, "includes_model_load": False,
        "includes_video_write": False, "wall_total_s": time.monotonic() - started,
        "sample_mean_s": total_s / 5, "per_prompt": rows,
        "source_phase_subset_s": {"runtime_stage_durations_are_source_reported_not_independent": True},
        "export": exported, "first_gpu_budget": budget,
        "config": {"world_size": 1, "num_gpus": 1, "available_gpus": 8,
                   "prompt_count": 5, "seed": 42, "num_frames": 81, "steps": 50},
    }
    write_json(out_dir / "benchmark.json", benchmark)
    event("run_end", run_dir=str(out_dir.parent), status="PASS", total_s=total_s,
          wall_s=benchmark["wall_total_s"])
    print("SOL_SITE_FIVE_PROMPT_PASS " + json.dumps({"total_s": total_s, "gpu_slots": [slot]}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        event("run_end", run_dir=os.environ.get("OUT_DIR", ""), status="FAILED",
              failure=f"{type(exc).__name__}: {exc}")
        raise
