#!/usr/bin/env python3
"""A100-only process bootstrap around the clean pre-search SGLang runner.

The historical DiffGenerator starts a worker with multiprocessing ``spawn``.
Therefore import shims are installed at module execution time in both the
parent (``__main__``) and spawned worker (``__mp_main__``), before unpickling
imports SGLang. No CPU geometry mock is installed in a GPU process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path("/share/project/shuchenweng/yongyan_liu/Experiments/SolAgent/20260913-a100-original-baseline")
RUNTIME_ROOT = (REPO_ROOT / "external/sol_runtime" if (REPO_ROOT / "external/sol_runtime").is_dir()
                else PROJECT_ROOT / "source/sol_runtime")
FROZEN_RUNNER = RUNTIME_ROOT / "scripts/sana/sana_video_sglang_run.py"
FROZEN_RUNNER_SHA256 = "b9e707bd131534d9096a570ab3e4d0c892db2dd6c5d1ac05cee8d1a72a03d964"
COMPAT_DIR = REPO_ROOT / "models/sana_video_2b_a100/compat"
PORT_NAMES = ("port", "master_port", "scheduler_port", "nccl_port")
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
_FROZEN_MODULE: Any = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap_imports() -> Any:
    """Install the smallest explicit BF16/A100 import path, also under spawn."""
    global _FROZEN_MODULE
    if _FROZEN_MODULE is not None:
        return _FROZEN_MODULE
    if os.environ.get("SOL_A100_GPU_OPT_IN") != "1":
        raise RuntimeError("clean A100 wrapper requires explicit GPU opt-in")
    if os.environ.get("SOL_A100_SGL_KERNEL_BACKEND") != "torch_fallback":
        raise RuntimeError("A100 SGLang GPU path requires explicit torch_fallback opt-in")
    if not FROZEN_RUNNER.is_file() or _sha256(FROZEN_RUNNER) != FROZEN_RUNNER_SHA256:
        raise RuntimeError("clean SGLang runner source differs from pinned SHA-256")
    for path in (str(COMPAT_DIR), str(RUNTIME_ROOT / "python")):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ["SGLANG_SANA_MINIMAL_IMPORT"] = "1"
    spec = importlib.util.spec_from_file_location("sol_a100_frozen_sana_runner", FROZEN_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen SANA runner module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._install_sana_minimal_package_shims()
    from bootstrap import (
        install_bf16_quantization_import_shim,
        install_srt_quantization_type_shim,
        install_torch_sgl_kernel_fallback,
    )

    install_bf16_quantization_import_shim(RUNTIME_ROOT)
    install_srt_quantization_type_shim(RUNTIME_ROOT)
    install_torch_sgl_kernel_fallback()
    import sgl_kernel

    if getattr(sgl_kernel, "__sol_a100_backend__", None) != "torch_fallback":
        raise RuntimeError("A100 Torch kernel fallback did not install")
    from sglang.multimodal_gen.runtime.platforms import cuda as cuda_platform

    _install_cuda_uuid_mapper(cuda_platform)
    # The historical main calls this again; it was already executed exactly
    # once before quantization and fallback setup. Keep its original position
    # for all other behavior while avoiding a duplicate package installation.
    module._install_sana_minimal_package_shims = lambda: None
    _FROZEN_MODULE = module
    return module


def _slot_ports(index: int) -> dict[str, int]:
    if index not in range(8):
        raise ValueError("A100 SGLang port slot must be 0..7")
    base = 41000 + 1000 * index
    return {"port": base + 500, "master_port": base + 800,
            "scheduler_port": base, "nccl_port": base + 100}


def _verify_visible_gpu_slot(index: int, visible_uuid: str) -> int:
    """Check the child UUID against the same frozen mapping used by the parent."""
    if index not in range(8):
        raise ValueError("A100 SGLang GPU slot must be 0..7")
    expected_uuid = GPU_UUIDS[index]
    if visible_uuid != expected_uuid:
        raise RuntimeError("CUDA_VISIBLE_DEVICES GPU UUID does not match the pinned A100 slot")
    return index


def _install_cuda_uuid_mapper(cuda_module: Any) -> None:
    """Adapt the frozen NVML helper to one UUID-pinned A100 CUDA device.

    The frozen helper parses CUDA_VISIBLE_DEVICES as an integer. Keep UUID
    visibility and require NVML to resolve it to the independently pinned
    physical slot; this process-local patch is reinstalled under spawn.
    """
    slot_text = os.environ.get("SOL_A100_PINNED_GPU_INDEX", "")
    if slot_text not in {str(index) for index in range(8)}:
        raise RuntimeError("missing or invalid pinned A100 GPU index for SGLang")
    slot = int(slot_text)
    visible_uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _verify_visible_gpu_slot(slot, visible_uuid)
    nvml = cuda_module.pynvml

    def uuid_to_physical_device_id(device_id: int) -> int:
        if type(device_id) is not int or device_id != 0:
            raise RuntimeError("UUID-pinned A100 SGLang permits only logical device 0")
        if (os.environ.get("CUDA_VISIBLE_DEVICES") != visible_uuid
                or os.environ.get("SOL_A100_PINNED_GPU_INDEX") != slot_text):
            raise RuntimeError("A100 GPU visibility or pinned slot changed after bootstrap")
        try:
            handle = nvml.nvmlDeviceGetHandleByUUID(visible_uuid)
            actual_uuid = nvml.nvmlDeviceGetUUID(handle)
            physical_index = nvml.nvmlDeviceGetIndex(handle)
        except Exception as exc:
            raise RuntimeError("NVML could not resolve the pinned A100 GPU UUID") from exc
        if isinstance(actual_uuid, bytes):
            actual_uuid = actual_uuid.decode("ascii")
        if actual_uuid != visible_uuid or type(physical_index) is not int:
            raise RuntimeError("NVML returned a mismatched pinned A100 GPU UUID")
        if physical_index != slot:
            raise RuntimeError("NVML physical GPU index differs from the pinned A100 slot")
        return physical_index

    cuda_module.device_id_to_physical_device_id = uuid_to_physical_device_id


def _reserve_port_check(ports: dict[str, int]) -> dict[str, Any]:
    if set(ports) != set(PORT_NAMES) or len(set(ports.values())) != 4:
        raise RuntimeError("A100 SGLang port set is malformed or overlapping")
    reservations: list[socket.socket] = []
    try:
        for name in PORT_NAMES:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("", ports[name]))
                sock.listen(1)
            except OSError as exc:
                sock.close()
                raise RuntimeError(f"A100 SGLang port unavailable: {name}={ports[name]}") from exc
            reservations.append(sock)
    finally:
        for sock in reservations:
            sock.close()
    return {"status": "AVAILABLE", "checked_ports": [ports[name] for name in PORT_NAMES],
            "check_scope": "simultaneous_ipv4_bind_before_runtime_launch"}


def _effective_ports(generator: Any, ports: dict[str, int], output_dir: Path) -> dict[str, Any]:
    server = getattr(generator, "server_args", None)
    port_args = getattr(generator, "port_args", None)
    expected = {**ports, "strict_ports": True, "output_path": str(output_dir)}
    observed = {name: getattr(server, name, None) for name in expected}
    if observed != expected or getattr(port_args, "master_port", None) != ports["master_port"] or getattr(port_args, "nccl_port", None) != ports["nccl_port"]:
        shutdown = getattr(generator, "shutdown", None)
        if callable(shutdown):
            shutdown()
        raise RuntimeError("effective SGLang ports or output path mismatch")
    return {"status": "VERIFIED", "ports": ports, "strict_ports": True,
            "output_path": str(output_dir)}


def _install_port_hook(generator_class: Any, ports: dict[str, int], output_dir: Path) -> None:
    original = generator_class.from_pretrained

    @classmethod
    def wrapped(cls: Any, local_mode: bool = True, **kwargs: Any) -> Any:
        if any(name in kwargs for name in (*PORT_NAMES, "strict_ports", "output_path", "server_args")):
            raise RuntimeError("frozen runner tried to override managed A100 ports or output path")
        generator = original(local_mode=local_mode, **kwargs, **ports,
                             strict_ports=True, output_path=str(output_dir))
        receipt = _effective_ports(generator, ports, output_dir)
        print("SOL_A100_SGLANG_PORTS " + json.dumps(receipt, sort_keys=True), flush=True)
        return generator

    generator_class.from_pretrained = wrapped


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu-index", required=True, type=int)
    for name in PORT_NAMES:
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=int)
    parser.add_argument("target_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if os.environ.get("SOL_A100_GPU_OPT_IN") != "1":
        raise RuntimeError("clean A100 wrapper requires explicit GPU opt-in")
    if args.target.resolve() != FROZEN_RUNNER.resolve():
        raise RuntimeError("A100 wrapper target must be the frozen historical SANA runner")
    output_dir = args.output_dir.resolve()
    allowed_root = (PROJECT_ROOT / "repo").resolve()
    if (not args.output_dir.is_dir() or args.output_dir.is_symlink()
            or not output_dir.is_relative_to(allowed_root)):
        raise RuntimeError("A100 wrapper output must be a real project-local run directory")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    index = _verify_visible_gpu_slot(args.gpu_index, visible)
    ports = {name: getattr(args, name) for name in PORT_NAMES}
    if ports != _slot_ports(index):
        raise RuntimeError("A100 wrapper ports do not match the pinned GPU slot")
    port_preflight = _reserve_port_check(ports)
    os.environ["SOL_A100_PINNED_GPU_INDEX"] = str(index)
    module = _bootstrap_imports()
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

    _install_port_hook(DiffGenerator, ports, output_dir)
    import sgl_kernel

    print("SOL_A100_SGLANG_BACKEND " + json.dumps({
        "support_kernel_backend": sgl_kernel.__sol_a100_backend__,
        "original_runner_sha256": FROZEN_RUNNER_SHA256,
        "a100_wrapper_sha256": _sha256(Path(__file__)),
        "a100_bootstrap_sha256": _sha256(COMPAT_DIR / "bootstrap.py"),
        "port_preflight": port_preflight,
        "gpu_visible_index": index,
        "gpu_visible_uuid": visible,
        "sm80_geometry_mock": False,
    }, sort_keys=True), flush=True)
    forwarded = list(args.target_args)
    if forwarded[:1] == ["--"]:
        forwarded.pop(0)
    sys.argv = [str(FROZEN_RUNNER), *forwarded]
    module.main()
    return 0


if __name__ == "__mp_main__":
    _bootstrap_imports()
elif __name__ == "__main__":
    raise SystemExit(main())
