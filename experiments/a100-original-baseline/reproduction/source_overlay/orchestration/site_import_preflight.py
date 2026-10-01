#!/usr/bin/env python3
"""Pinned SANA SGLang A100 import check; CPU-stub and real-GPU modes differ."""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path("/share/project/shuchenweng/yongyan_liu/Experiments/SolAgent/20260913-a100-original-baseline")
SOURCE = ROOT / "source/sol_runtime"
OVERLAY = Path("/share/project/shuchenweng/yongyan_liu/Experiments/SolRolloutBench/20260912-a100-eight/envs/sglang-overlay")
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


def main() -> int:
    import sys
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_mode = os.environ.get("SOL_A100_GPU_IMPORT_PREFLIGHT") == "1"
    if (gpu_mode and visible not in GPU_UUIDS) or (not gpu_mode and visible != ""):
        raise RuntimeError("GPU preflight requires one authorized UUID; CPU preflight requires none")
    for path in (ROOT / "repo/models/sana_video_2b_a100/compat", SOURCE / "python", OVERLAY):
        sys.path.insert(0, str(path))
    os.environ["SOL_A100_SGL_KERNEL_BACKEND"] = "torch_fallback"
    os.environ["SGLANG_SANA_MINIMAL_IMPORT"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.monotonic()
    runner = SOURCE / "scripts/sana/sana_video_sglang_run.py"
    spec = importlib.util.spec_from_file_location("sana_runner_import_preflight", runner)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._install_sana_minimal_package_shims()
    from bootstrap import (install_bf16_quantization_import_shim,
                           install_srt_quantization_type_shim,
                           install_torch_sgl_kernel_fallback)
    install_bf16_quantization_import_shim(SOURCE)
    install_srt_quantization_type_shim(SOURCE)
    install_torch_sgl_kernel_fallback()
    # CPU mode only: upstream Triton asks torch for device properties at import.
    # The synthetic property is never used in the real-GPU mode or inference.
    import torch
    if gpu_mode:
        if os.environ.get("SOL_A100_CPU_IMPORT_STUB"):
            raise RuntimeError("GPU preflight forbids CPU import stub")
        torch.cuda.init()
        properties = torch.cuda.get_device_properties(0)
        if torch.cuda.device_count() != 1 or properties.major != 8 or properties.minor != 0:
            raise RuntimeError("expected exactly one visible A100 / sm_80 GPU")
        import_stub = None
    else:
        if os.environ.get("SOL_A100_CPU_IMPORT_STUB") != "1":
            raise RuntimeError("CPU preflight requires an explicit import-only device-property stub")
        cpu_module = torch.get_device_module()

        class ImportOnlyDeviceModule:
            def current_device(self) -> int:
                return 0

            def get_device_properties(self, device: int) -> SimpleNamespace:
                if device != 0:
                    raise RuntimeError("unexpected synthetic CPU import device")
                return SimpleNamespace(warp_size=32)

            def __getattr__(self, name: str):
                return getattr(cpu_module, name)

        torch.get_device_module = lambda: ImportOnlyDeviceModule()
        import_stub = "A100_warp_size_32_no_GPU_runtime_claim"
    from sglang.multimodal_gen.runtime.models.dits.sana_video import SanaVideoTransformer3DModel
    from sglang.multimodal_gen.runtime.pipelines.sana_video import SanaVideoPipeline
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
    import sgl_kernel
    print(json.dumps({"status": "PASS", "classes": [SanaVideoTransformer3DModel.__name__,
                                                     SanaVideoPipeline.__name__, DiffGenerator.__name__],
                      "support_kernel_backend": sgl_kernel.__sol_a100_backend__,
                      "import_only_stub": import_stub,
                      "cuda_device_name": properties.name if gpu_mode else None,
                      "cuda_capability": [properties.major, properties.minor] if gpu_mode else None,
                      "cuda_visible_devices": visible,
                      "duration_s": time.monotonic() - started}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
