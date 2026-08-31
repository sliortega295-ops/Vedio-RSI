#!/usr/bin/env python3
"""Locked one-H100 SANA-Video 2B runner with durable performance receipts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from gpu_guard import atomic_write_json, locked_idle_lease, query_compute_apps, query_gpu


NEGATIVE_PROMPT = (
    "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, "
    "sudden disappearance, jump cuts, jerky movements, rapid shot changes, "
    "frames out of sync, inconsistent character shapes, temporal artifacts, "
    "jitter, and ghosting effects, creating a disorienting visual experience."
)
WORKLOAD = {
    "width": 832,
    "height": 480,
    "frames": 81,
    "fps": 16,
    "steps": 50,
    "guidance_scale": 6.0,
    "seed": 42,
    "motion_score": 30,
    "vae_precision": "fp32",
    "transformer_precision": "bf16",
    "text_encoder_precision": "bf16",
    "flow_shift": 8.0,
}
TECHNIQUE_ENV = (
    "SGLANG_SANA_LINATTN_BF16",
    "SGLANG_SANA_QKV_MERGE",
    "SGLANG_SANA_EASYCACHE_THRESH",
    "SGLANG_SANA_EASYCACHE_WARMUP",
    "SGLANG_SANA_EASYCACHE_SUBSAMPLE",
    "SGLANG_SANA_EASYCACHE_DEBUG",
    "SGLANG_SANA_PROFILE",
    "SGLANG_TORCH_COMPILE_MODE",
    "TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC",
)
CRITICAL_RUNTIME_FILES = (
    "scripts/sana/sana_video_sglang_run.py",
    "python/sglang/multimodal_gen/registry_sana.py",
    "python/sglang/multimodal_gen/configs/models/dits/sana_video.py",
    "python/sglang/multimodal_gen/configs/pipeline_configs/sana_video.py",
    "python/sglang/multimodal_gen/configs/sample/sana_video.py",
    "python/sglang/multimodal_gen/runtime/models/dits/sana_video.py",
    "python/sglang/jit_kernel/diffusion/triton/sana_rope.py",
    "python/sglang/multimodal_gen/runtime/pipelines/sana_video.py",
)


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required environment variable is empty: {name}")
    return value


def _runtime_visible_device(gpu_uuid: str, gpu: dict[str, object]) -> str:
    """Resolve a UUID lease to the verified host index expected by old SGLang."""
    if gpu.get("uuid") != gpu_uuid:
        raise RuntimeError(
            f"leased GPU UUID {gpu_uuid} does not match live GPU receipt {gpu.get('uuid')}"
        )
    index = gpu.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise RuntimeError(f"live GPU receipt has invalid host index: {index!r}")
    return str(index)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    if raw in {"1", "true", "True", "yes", "on"}:
        return True
    if raw in {"0", "false", "False", "no", "off", ""}:
        return False
    raise RuntimeError(f"{name} must be a boolean, got {raw!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _optimization_knobs() -> dict[str, object]:
    easycache = float(os.environ.get("SANA_EASYCACHE_THRESH", "0"))
    if easycache < 0:
        raise RuntimeError("SANA_EASYCACHE_THRESH must be non-negative")
    return {
        "torch_compile": _bool_env("SANA_ENABLE_COMPILE"),
        "max_autotune": _bool_env("SANA_ENABLE_MAX_AUTOTUNE"),
        "linear_attention_bf16": _bool_env("SANA_ENABLE_LINATTN_BF16"),
        "qkv_merge": _bool_env("SANA_ENABLE_QKV_MERGE"),
        "paired_rope": _bool_env("SGLANG_SANA_PAIRED_ROPE"),
        "easycache_threshold": easycache,
        "warmup_disabled": _bool_env("SANA_DISABLE_WARMUP"),
    }


def _assert_dense_control(config_id: str, knobs: dict[str, object]) -> None:
    if config_id != "sana_video_2b_h100_dense_baseline":
        return
    enabled = {
        key: value
        for key, value in knobs.items()
        if (isinstance(value, bool) and value)
        or (key == "easycache_threshold" and float(value) != 0.0)
    }
    if enabled:
        raise RuntimeError(f"dense baseline has optimization knobs enabled: {enabled}")


def _command(
    python_bin: Path,
    runtime_root: Path,
    model_path: Path,
    prompt_file: Path,
    output_basename: str,
    knobs: dict[str, object],
) -> list[str]:
    command = [
        str(python_bin),
        str(runtime_root / "scripts/sana/sana_video_sglang_run.py"),
        "--model",
        str(model_path),
        "--prompt-file",
        str(prompt_file),
        "--frames",
        str(WORKLOAD["frames"]),
        "--steps",
        str(WORKLOAD["steps"]),
        "--height",
        str(WORKLOAD["height"]),
        "--width",
        str(WORKLOAD["width"]),
        "--guidance-scale",
        str(WORKLOAD["guidance_scale"]),
        "--seed",
        str(WORKLOAD["seed"]),
        "--output",
        output_basename,
    ]
    if knobs["torch_compile"]:
        command.extend(["--compile", "--compile-mode", os.environ.get("SANA_COMPILE_MODE", "default")])
    if knobs["max_autotune"]:
        if not knobs["torch_compile"]:
            raise RuntimeError("max autotune requires torch compile")
        command.append("--max-autotune")
    if knobs["linear_attention_bf16"]:
        command.append("--linattn-bf16")
    if knobs["qkv_merge"]:
        command.append("--qkv-merge")
    if float(knobs["easycache_threshold"]) > 0:
        command.extend(
            [
                "--easycache",
                str(knobs["easycache_threshold"]),
                "--ec-warmup",
                os.environ.get("SANA_EC_WARMUP", "3"),
                "--ec-subsample",
                os.environ.get("SANA_EC_SUBSAMPLE", "8"),
            ]
        )
    if knobs["warmup_disabled"]:
        command.append("--no-warmup")
    return command


def _monitor_gpu(gpu_uuid: str, stop: threading.Event, samples: list[dict[str, object]]) -> None:
    while not stop.is_set():
        try:
            sample = query_gpu(gpu_uuid)
            sample["sampled_at_utc"] = datetime.now(timezone.utc).isoformat()
            samples.append(sample)
        except Exception as exc:  # telemetry failure must remain visible in the receipt
            samples.append(
                {
                    "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error": repr(exc),
                }
            )
        stop.wait(0.5)


def _run_logged(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    gpu_uuid: str,
) -> tuple[int, float, list[dict[str, object]], str]:
    samples: list[dict[str, object]] = []
    stop = threading.Event()
    started = time.perf_counter()
    transcript: list[str] = []
    with log_path.open("w") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        monitor = threading.Thread(
            target=_monitor_gpu,
            args=(gpu_uuid, stop, samples),
            daemon=True,
        )
        monitor.start()
        assert process.stdout is not None
        for line in process.stdout:
            transcript.append(line)
            log_handle.write(line)
            log_handle.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        returncode = process.wait()
        stop.set()
        monitor.join(timeout=5)
    return returncode, time.perf_counter() - started, samples, "".join(transcript)


def _ffprobe(video: Path) -> dict[str, object]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_read_frames,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise RuntimeError(f"expected one video stream, got {len(streams)}")
    stream = streams[0]
    frames_raw = stream.get("nb_read_frames") or stream.get("nb_frames")
    fps = float(Fraction(str(stream["r_frame_rate"])))
    duration = float(stream.get("duration") or payload.get("format", {}).get("duration"))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(frames_raw),
        "fps": fps,
        "duration_s": duration,
        "raw": payload,
    }


def _validate_video(video: Path, frames_dir: Path) -> dict[str, object]:
    probe = _ffprobe(video)
    expected_duration = WORKLOAD["frames"] / WORKLOAD["fps"]
    checks = {
        "width": probe["width"] == WORKLOAD["width"],
        "height": probe["height"] == WORKLOAD["height"],
        "frames": probe["frames"] == WORKLOAD["frames"],
        "fps": abs(float(probe["fps"]) - WORKLOAD["fps"]) < 1e-6,
        "duration": abs(float(probe["duration_s"]) - expected_duration) <= 0.25,
        "nonempty": video.stat().st_size > 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"video validity checks failed: {checks}, probe={probe}")
    frames_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            "select=eq(n\\,0)+eq(n\\,40)+eq(n\\,80)",
            "-vsync",
            "0",
            str(frames_dir / "frame_%02d.png"),
        ],
        check=True,
    )
    frame_receipts = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(frames_dir.glob("frame_*.png"))
    ]
    if len(frame_receipts) != 3:
        raise RuntimeError(f"expected 3 decoded validity frames, got {len(frame_receipts)}")
    return {"status": "VALIDATED", "checks": checks, "ffprobe": probe, "frames": frame_receipts}


def _wait_for_no_compute_apps(gpu_uuid: str, timeout_s: float = 15.0) -> list[dict[str, object]]:
    deadline = time.monotonic() + timeout_s
    while True:
        apps = query_compute_apps(gpu_uuid)
        if not apps or time.monotonic() >= deadline:
            return apps
        time.sleep(0.5)


def main() -> int:
    out_dir = Path(_required_env("OUT_DIR")).resolve()
    repo_root = Path(_required_env("AUTOVIDEO_REPO_ROOT")).resolve()
    runtime_root = Path(
        os.environ.get("SANA_RUNTIME_ROOT", str(repo_root / "external/sol_runtime"))
    ).resolve()
    python_bin = Path(_required_env("SANA_PYTHON_BIN")).resolve()
    model_path = Path(_required_env("SANA_MODEL_PATH")).resolve()
    prompt_file_raw = Path(_required_env("SANA_PROMPT_FILE"))
    prompt_file = (
        prompt_file_raw if prompt_file_raw.is_absolute() else repo_root / prompt_file_raw
    ).resolve()
    lease_file = Path(_required_env("SANA_GPU_LEASE_FILE")).resolve()
    config_id = _required_env("AUTOVIDEO_CONFIG_ID")

    required_paths = [
        runtime_root / "scripts/sana/sana_video_sglang_run.py",
        python_bin,
        model_path / "model_index.json",
        prompt_file,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"required runtime paths are missing: {missing}")
    prompt = prompt_file.read_text().strip()
    if not prompt.endswith("motion score: 30."):
        raise RuntimeError("formal prompt must end with the exact motion-score suffix")

    knobs = _optimization_knobs()
    _assert_dense_control(config_id, knobs)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = hashlib.sha256(str(out_dir).encode()).hexdigest()[:12]
    output_basename = f"sana2b_{slug}"
    runtime_video = runtime_root / "outputs" / f"{output_basename}.mp4"
    final_video = out_dir / "out.mp4"
    if runtime_video.exists() or final_video.exists():
        raise RuntimeError("refusing to overwrite an existing output video")

    command = _command(
        python_bin, runtime_root, model_path, prompt_file, output_basename, knobs
    )
    critical_hashes = {
        rel: _sha256(runtime_root / rel) for rel in CRITICAL_RUNTIME_FILES
    }
    workload_receipt = {
        **WORKLOAD,
        "prompt": prompt,
        "prompt_sha256": _sha256(prompt_file),
        "negative_prompt": NEGATIVE_PROMPT,
    }
    child_env = dict(os.environ)
    for name in TECHNIQUE_ENV:
        child_env.pop(name, None)
    child_env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "SGLANG_SANA_MINIMAL_IMPORT": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )

    with locked_idle_lease(lease_file) as (lease, gpu_before):
        # CUDA itself accepts a UUID here, but the archived SGLang CUDA platform
        # parses CUDA_VISIBLE_DEVICES with int().  Keep the lease and every
        # ownership check UUID-scoped, then expose only that verified host index
        # to the child process for compatibility with the archived runtime.
        child_env["CUDA_VISIBLE_DEVICES"] = _runtime_visible_device(
            lease.gpu_uuid, gpu_before
        )
        run_config: dict[str, object] = {
            "schema_version": 1,
            "config_id": config_id,
            "world_size": 1,
            "num_gpus": 1,
            "nproc": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "cwd": str(runtime_root),
            "gpu_lease": {
                "gpu_uuid": lease.gpu_uuid,
                "lease_file": str(lease.lease_file),
                "lock_path": str(lease.lock_path),
                "host": lease.host,
                "owner": lease.owner,
                "leased_at_utc": lease.leased_at_utc,
                "cuda_visible_devices": child_env["CUDA_VISIBLE_DEVICES"],
                "visibility_resolution": "uuid_lease_to_verified_host_index",
            },
            "gpu_before": gpu_before,
            "workload": workload_receipt,
            "workload_fingerprint": _fingerprint(workload_receipt),
            "optimization_knobs": knobs,
            "source": {
                "harness_archival_parent": _required_env("SANA_HARNESS_ARCHIVE_SHA"),
                "runtime_authority_sha": _required_env("SANA_RUNTIME_AUTHORITY_SHA"),
                "runtime_compat_sha": _required_env("SANA_RUNTIME_COMPAT_SHA"),
                "runtime_root": str(runtime_root),
                "critical_file_sha256": critical_hashes,
            },
            "model": {
                "repo": _required_env("SANA_MODEL_REPO"),
                "revision": _required_env("SANA_MODEL_REVISION"),
                "path": str(model_path),
            },
            "environment": {
                "python": str(python_bin),
                "python_sha256": _sha256(python_bin),
                "dependency_overlay": _required_env("SANA_DEPENDENCY_OVERLAY"),
                "kernel_staging": _required_env("SANA_KERNEL_STAGING"),
                "pythonpath": child_env.get("PYTHONPATH", ""),
                "hf_hub_offline": child_env["HF_HUB_OFFLINE"],
            },
        }
        atomic_write_json(out_dir / "run_config.json", run_config)
        (out_dir / "command.txt").write_text(" ".join(command) + "\n")

        returncode, wall_s, samples, transcript = _run_logged(
            command,
            cwd=runtime_root,
            env=child_env,
            log_path=out_dir / "run.log",
            gpu_uuid=lease.gpu_uuid,
        )
        residual = _wait_for_no_compute_apps(lease.gpu_uuid)
        generation_match = re.search(r"GENERATE_OK in ([0-9.]+)s", transcript)
        runtime_peak_match = re.search(r"Max peak: ([0-9.]+) MB", transcript)
        telemetry_peak = max(
            (
                int(sample["memory_used_mib"])
                for sample in samples
                if "memory_used_mib" in sample
            ),
            default=0,
        )
        generation_s = float(generation_match.group(1)) if generation_match else None
        runtime_peak_memory_mb = (
            float(runtime_peak_match.group(1)) if runtime_peak_match else None
        )
        authoritative_peak_mib = runtime_peak_memory_mb or float(telemetry_peak)
        benchmark: dict[str, object] = {
            "schema_version": 1,
            "status": "FAILED" if returncode else "PARTIAL",
            "returncode": returncode,
            "generation_s": generation_s,
            "total_s": generation_s,
            "denoise_s": None,
            "timing_scope": (
                "warm_single_prompt_gen.generate_including_text_encoder_denoise_"
                "vae_decode_and_video_write_excluding_model_load_and_one_step_warmup"
            ),
            "process_wall_s": wall_s,
            "runtime_peak_memory_mb": runtime_peak_memory_mb,
            "nvidia_smi_peak_memory_mib": telemetry_peak,
            "max_device_memory_used_mib": authoritative_peak_mib,
            "memory": {"max_device_memory_used_mib": authoritative_peak_mib},
            "world_size": 1,
            "num_gpus": 1,
            "nproc": 1,
            "prompt_count": 1,
            "steps_per_prompt": WORKLOAD["steps"],
            "warmup_policy": "one_step_model_warmup_before_measured_generation",
            "nvidia_smi_sample_count": len(samples),
            "nvidia_smi_samples": samples,
            "gpu_before": gpu_before,
            "gpu_after": query_gpu(lease.gpu_uuid),
            "residual_compute_apps": residual,
        }
        atomic_write_json(out_dir / "benchmark.json", benchmark)
        if returncode != 0:
            raise RuntimeError(f"SANA runtime exited with status {returncode}")
        if residual:
            raise RuntimeError(f"runtime left compute apps on leased GPU: {residual}")
        if generation_s is None:
            raise RuntimeError("runtime succeeded without a parseable GENERATE_OK latency")
        if not runtime_video.exists():
            raise RuntimeError(f"runtime reported success but video is missing: {runtime_video}")
        shutil.move(runtime_video, final_video)
        validity = _validate_video(final_video, out_dir / "frames")
        atomic_write_json(out_dir / "quality.json", validity)
        benchmark.update(
            {
                "status": "VALIDATED",
                "video": {
                    "path": str(final_video),
                    "bytes": final_video.stat().st_size,
                    "sha256": _sha256(final_video),
                },
                "validity": validity,
            }
        )
        atomic_write_json(out_dir / "benchmark.json", benchmark)
        atomic_write_json(
            out_dir / "collection.json",
            {
                "schema_version": 1,
                "status": "VALIDATED",
                "artifacts": {
                    path.name: {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                    for path in (
                        final_video,
                        out_dir / "run.log",
                        out_dir / "run_config.json",
                        out_dir / "benchmark.json",
                        out_dir / "quality.json",
                    )
                },
            },
        )
    print(json.dumps(benchmark, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
