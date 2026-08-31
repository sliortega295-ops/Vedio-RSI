#!/usr/bin/env python3
"""Run the minimal cu128/H100 SANA import and BF16 kernel gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from gpu_guard import atomic_write_json, locked_idle_lease, query_compute_apps


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required environment variable is empty: {name}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_runtime_shims(runtime_root: Path) -> None:
    runner = runtime_root / "scripts/sana/sana_video_sglang_run.py"
    spec = importlib.util.spec_from_file_location("sana_video_sglang_run", runner)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runtime shim installer: {runner}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._install_sana_minimal_package_shims()


def main() -> int:
    lease_file = Path(_required_env("SANA_GPU_LEASE_FILE"))
    runtime_root = Path(_required_env("SANA_RUNTIME_ROOT")).resolve()
    model_path = Path(_required_env("SANA_MODEL_PATH")).resolve()
    output = Path(_required_env("SANA_PROBE_OUTPUT")).resolve()
    started = datetime.now(timezone.utc).isoformat()

    with locked_idle_lease(lease_file) as (lease, gpu_before):
        os.environ["CUDA_VISIBLE_DEVICES"] = lease.gpu_uuid
        os.environ["SGLANG_SANA_MINIMAL_IMPORT"] = "1"
        _install_runtime_shims(runtime_root)

        import torch
        from sgl_kernel.elementwise import rmsnorm
        from sglang.multimodal_gen.configs.pipeline_configs.sana_video import (
            SanaVideoPipelineConfig,
        )
        from sglang.multimodal_gen.configs.sample.sana_video import (
            SanaVideoSamplingParams,
        )
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
            DiffGenerator,
        )
        from sglang.multimodal_gen.runtime.models.dits.sana_video import (
            SanaVideoTransformer3DModel,
        )
        from sglang.multimodal_gen.runtime.pipelines.sana_video import SanaVideoPipeline

        torch.cuda.init()
        if torch.cuda.device_count() != 1:
            raise RuntimeError(f"expected one logical CUDA device, got {torch.cuda.device_count()}")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != lease.gpu_uuid:
            raise RuntimeError("logical CUDA visibility does not match persistent lease")

        torch.manual_seed(42)
        x = torch.randn((32, 1152), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((1152,), device="cuda", dtype=torch.bfloat16)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        output_tensor = None
        for _ in range(20):
            output_tensor = rmsnorm(x, weight)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert output_tensor is not None
        reference = (
            x.float()
            * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            * weight.float()
        )
        max_abs_error = float((output_tensor.float() - reference).abs().max().item())
        allclose = bool(torch.allclose(output_tensor.float(), reference, rtol=0.03, atol=0.03))
        if not allclose:
            raise RuntimeError(f"BF16 RMSNorm gate failed, max_abs_error={max_abs_error}")

        scheduler = json.loads((model_path / "scheduler/scheduler_config.json").read_text())
        transformer = json.loads((model_path / "transformer/config.json").read_text())
        report: dict[str, object] = {
            "schema_version": 1,
            "status": "VALIDATED",
            "started_at_utc": started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "gpu": gpu_before,
            "gpu_uuid": lease.gpu_uuid,
            "lease_file": str(lease.lease_file),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "triton_version": __import__("triton").__version__,
            "device_name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "runtime_root": str(runtime_root),
            "runtime_runner_sha256": _sha256(
                runtime_root / "scripts/sana/sana_video_sglang_run.py"
            ),
            "model_path": str(model_path),
            "model_revision": _required_env("SANA_MODEL_REVISION"),
            "scheduler_flow_shift": scheduler.get("flow_shift"),
            "attention_head_dim": transformer.get("attention_head_dim"),
            "pipeline_precision": SanaVideoPipelineConfig().precision,
            "pipeline_vae_precision": SanaVideoPipelineConfig().vae_precision,
            "imports": [
                SanaVideoTransformer3DModel.__name__,
                SanaVideoPipelineConfig.__name__,
                SanaVideoSamplingParams.__name__,
                SanaVideoPipeline.__name__,
                DiffGenerator.__name__,
            ],
            "rmsnorm": {
                "dtype": str(x.dtype),
                "shape": list(x.shape),
                "calls": 20,
                "elapsed_ms": elapsed_ms,
                "max_abs_error": max_abs_error,
                "allclose": allclose,
            },
            "memory_allocated_peak_bytes": torch.cuda.max_memory_allocated(),
            "memory_reserved_peak_bytes": torch.cuda.max_memory_reserved(),
        }
        del output_tensor, reference, x, weight
        torch.cuda.empty_cache()
        atomic_write_json(output, report)

    residual = query_compute_apps(lease.gpu_uuid)
    if residual:
        raise RuntimeError(f"probe left compute apps on leased GPU: {residual}")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
