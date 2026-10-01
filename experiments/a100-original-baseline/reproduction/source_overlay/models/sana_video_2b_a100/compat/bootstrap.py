"""Opt-in, BF16-only import bootstrap for the A100 SANA replay.

Only the diffusion quantization package initializer is bypassed: that eager
initializer imports every FP8/FP4 backend and the unrelated SGLang LLM runtime.
The original quantization base class is loaded unchanged from the frozen
source. Any actual request for a quantized model fails closed.

This is an import compatibility shim, not proof of numerical equivalence or
SM80 kernel support. It must run before any diffusion quantization import.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import os
import sys
import types
from pathlib import Path


PACKAGE_NAME = "sglang.multimodal_gen.runtime.layers.quantization"
SRT_QUANT_PACKAGE_NAME = "sglang.srt.layers.quantization"
KERNEL_BACKEND_ENV = "SOL_A100_SGL_KERNEL_BACKEND"


def install_bf16_quantization_import_shim(runtime_root: Path) -> None:
    if PACKAGE_NAME in sys.modules:
        raise RuntimeError("A100 BF16 quantization shim must precede SGLang imports")

    package_path = (
        runtime_root
        / "python/sglang/multimodal_gen/runtime/layers/quantization"
    ).resolve()
    base_config = package_path / "configs/base_config.py"
    if not base_config.is_file():
        raise FileNotFoundError(f"frozen quantization base config missing: {base_config}")

    module = types.ModuleType(PACKAGE_NAME)
    module.__package__ = PACKAGE_NAME
    module.__path__ = [str(package_path)]
    module.__spec__ = importlib.machinery.ModuleSpec(
        PACKAGE_NAME, loader=None, is_package=True
    )
    module.__spec__.submodule_search_locations = [str(package_path)]
    sys.modules[PACKAGE_NAME] = module

    original_base = importlib.import_module(
        f"{PACKAGE_NAME}.configs.base_config"
    )
    module.QuantizationConfig = original_base.QuantizationConfig

    def _unsupported_quantization(method: str):
        raise RuntimeError(
            "A100 SANA BF16-only import shim rejects quantization method: "
            + str(method)
        )

    module.get_quantization_config = _unsupported_quantization
    module.QUANTIZATION_METHODS = []


def install_srt_quantization_type_shim(runtime_root: Path) -> None:
    """Expose the original SRT QuantizationConfig type without eager FP8 imports.

    The SANA transformer loader imports this only for type annotations. Its
    package initializer otherwise loads every LLM quantizer and the server.
    """
    if SRT_QUANT_PACKAGE_NAME in sys.modules:
        raise RuntimeError("A100 SRT quantization shim must precede SRT imports")
    package_path = (
        runtime_root / "python/sglang/srt/layers/quantization"
    ).resolve()
    base_config = package_path / "base_config.py"
    if not base_config.is_file():
        raise FileNotFoundError(f"frozen SRT quantization base config missing: {base_config}")

    module = types.ModuleType(SRT_QUANT_PACKAGE_NAME)
    module.__package__ = SRT_QUANT_PACKAGE_NAME
    module.__path__ = [str(package_path)]
    module.__spec__ = importlib.machinery.ModuleSpec(
        SRT_QUANT_PACKAGE_NAME, loader=None, is_package=True
    )
    module.__spec__.submodule_search_locations = [str(package_path)]
    sys.modules[SRT_QUANT_PACKAGE_NAME] = module
    original_base = importlib.import_module(f"{SRT_QUANT_PACKAGE_NAME}.base_config")
    module.QuantizationConfig = original_base.QuantizationConfig


def install_torch_sgl_kernel_fallback() -> None:
    """Install only the three SANA-used sgl_kernel APIs, by explicit opt-in.

    This substitutes ordinary Torch operations for a binary wheel that is not
    ABI-compatible with the reused A100 Torch. It is not a performance-equivalent
    or bitwise-equivalent implementation of the historical CUDA kernels.
    """
    if os.environ.get(KERNEL_BACKEND_ENV) != "torch_fallback":
        raise RuntimeError(
            f"Torch sgl_kernel fallback requires {KERNEL_BACKEND_ENV}=torch_fallback"
        )
    if "sgl_kernel" in sys.modules:
        raise RuntimeError("A100 sgl_kernel fallback must precede sgl_kernel imports")

    import torch
    import torch.nn.functional as F

    def _validate_norm(
        input: torch.Tensor, weight: torch.Tensor, eps: float, out: torch.Tensor | None
    ) -> None:
        if input.ndim < 1 or weight.ndim != 1 or input.shape[-1] != weight.shape[0]:
            raise ValueError("RMSNorm input/weight hidden dimensions do not match")
        if not input.is_floating_point() or not weight.is_floating_point():
            raise TypeError("RMSNorm fallback requires floating point tensors")
        if weight.device != input.device or (out is not None and out.device != input.device):
            raise ValueError("RMSNorm fallback requires tensors on the same device")
        if eps <= 0:
            raise ValueError("RMSNorm epsilon must be positive")
        if out is not None and (out.shape != input.shape or out.dtype != input.dtype):
            raise ValueError("RMSNorm output shape/dtype must match input")

    def _norm_value(
        input: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> torch.Tensor:
        value = input.float()
        scale = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
        return (value * scale * weight.float()).to(input.dtype)

    def rmsnorm(
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        out: torch.Tensor | None = None,
        enable_pdl: bool | None = None,
    ) -> torch.Tensor:
        if enable_pdl is True:
            raise ValueError("Torch fallback cannot honor enable_pdl=True")
        _validate_norm(input, weight, eps, out)
        value = _norm_value(input, weight, eps)
        if out is not None:
            out.copy_(value)
            return out
        return value

    def fused_add_rmsnorm(
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        enable_pdl: bool | None = None,
    ) -> None:
        if enable_pdl is True:
            raise ValueError("Torch fallback cannot honor enable_pdl=True")
        _validate_norm(input, weight, eps, None)
        if residual.shape != input.shape or residual.dtype != input.dtype:
            raise ValueError("fused RMSNorm residual shape/dtype must match input")
        if residual.device != input.device:
            raise ValueError("fused RMSNorm residual device must match input")
        if input.data_ptr() == residual.data_ptr():
            raise ValueError("fused RMSNorm input and residual cannot alias")
        # Match the historical native reference's FP32 sum and dtype-rounded
        # residual. The CUDA kernel's internal rounding still needs a GPU gate.
        summed = input.float() + residual.float()
        residual.copy_(summed.to(input.dtype))
        input.copy_(_norm_value(summed, weight, eps))

    def silu_and_mul(
        input: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        if input.ndim < 1 or input.shape[-1] % 2:
            raise ValueError("SiluAndMul requires an even final dimension")
        if not input.is_floating_point():
            raise TypeError("SiluAndMul fallback requires a floating point input")
        expected_shape = (*input.shape[:-1], input.shape[-1] // 2)
        if out is not None and (
            out.shape != expected_shape or out.dtype != input.dtype or out.device != input.device
        ):
            raise ValueError("SiluAndMul output shape/dtype/device mismatch")
        left, right = input.chunk(2, dim=-1)
        value = (F.silu(left.float()) * right.float()).to(input.dtype)
        if out is not None:
            out.copy_(value)
            return out
        return value

    module = types.ModuleType("sgl_kernel")
    module.__sol_a100_backend__ = "torch_fallback"
    module.rmsnorm = rmsnorm
    module.fused_add_rmsnorm = fused_add_rmsnorm
    module.silu_and_mul = silu_and_mul

    def _unsupported_kernel(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        raise RuntimeError("A100 Torch sgl_kernel fallback rejects unsupported API: " + name)

    module.__getattr__ = _unsupported_kernel
    sys.modules["sgl_kernel"] = module
