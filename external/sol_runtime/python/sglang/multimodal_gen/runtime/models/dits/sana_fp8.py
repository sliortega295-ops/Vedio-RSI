"""Selective H100 FP8 execution for SANA-Video pointwise FFN projections.

SANA-Video's ``GLUMBTempConv`` FFN is convolutional, unlike the linear FFN in
LTX.  Its ``conv_inverted`` and ``conv_point`` operators are nevertheless 1x1
convolutions and are exactly expressible as last-dimension matrix
multiplications.  This module replaces only those operators with online-packed
W8A8 E4M3 GEMMs while retaining the depthwise and temporal convolutions in
BF16.

The implementation deliberately reuses SGLang's established FP8 quantizer and
``apply_fp8_linear`` dispatcher.  OFF performs no replacement.  Unsupported
hardware, missing kernels, and incompatible module shapes are reported rather
than silently counted as FP8 execution.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import torch
import torch.nn as nn


_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off", ""})
_SUPPORTED_SCOPES = frozenset({"ffn_1x1"})


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    raise RuntimeError(f"{name} must be boolean, got {raw!r}")


@dataclass(frozen=True)
class SanaFP8Policy:
    """Environment-controlled FP8 installation policy.

    ``block_end`` is inclusive.  ``-1`` resolves to the final transformer
    block.  Dense guards are represented by narrowing ``block_start`` and
    ``block_end``; excluded blocks remain the original BF16 modules.
    """

    enabled: bool = False
    scope: str = "ffn_1x1"
    block_start: int = 0
    block_end: int = -1
    strict: bool = True

    @classmethod
    def from_env(cls) -> "SanaFP8Policy":
        return cls(
            enabled=_bool_env("SGLANG_SANA_FP8", False),
            scope=os.environ.get("SGLANG_SANA_FP8_SCOPE", "ffn_1x1").strip(),
            block_start=int(os.environ.get("SGLANG_SANA_FP8_BLOCK_START", "0")),
            block_end=int(os.environ.get("SGLANG_SANA_FP8_BLOCK_END", "-1")),
            strict=_bool_env("SGLANG_SANA_FP8_STRICT", True),
        )

    def validate(self, block_count: int) -> tuple[int, int]:
        if self.scope not in _SUPPORTED_SCOPES:
            raise RuntimeError(
                f"unsupported SANA FP8 scope {self.scope!r}; "
                f"supported={sorted(_SUPPORTED_SCOPES)}"
            )
        if block_count <= 0:
            raise RuntimeError("SANA FP8 requires at least one transformer block")
        end = block_count - 1 if self.block_end == -1 else self.block_end
        if self.block_start < 0 or end < self.block_start or end >= block_count:
            raise RuntimeError(
                "invalid SANA FP8 block range "
                f"[{self.block_start}, {self.block_end}] for {block_count} blocks"
            )
        return self.block_start, end


class FP8Backend(Protocol):
    name: str

    def preflight(self, device: torch.device, dtype: torch.dtype) -> dict[str, Any]: ...

    def pack_weight(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...

    def linear(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor: ...


class SGLangH100FP8Backend:
    """SGLang dynamic-activation/per-channel-weight FP8 backend for SM90+."""

    name = "sglang_w8a8_e4m3"

    def __init__(self) -> None:
        # Delay these imports until FP8 is actually requested.  This keeps the
        # dense OFF path independent of optional staged CUDA extensions.
        from sglang.srt.layers.quantization.fp8_kernel import (
            per_token_group_quant_fp8,
        )
        from sglang.srt.layers.quantization.fp8_utils import (
            apply_fp8_linear,
            cutlass_fp8_supported,
        )

        self._quantize = per_token_group_quant_fp8
        self._apply = apply_fp8_linear
        self._cutlass_supported = bool(cutlass_fp8_supported())

    def preflight(self, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(f"SANA FP8 requires CUDA weights, got device={device}")
        capability = torch.cuda.get_device_capability(device)
        if capability < (8, 9):
            raise RuntimeError(
                "SANA FP8 requires native FP8 tensor cores (SM89+); "
                f"got capability={capability}"
            )
        if dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError(f"SANA FP8 expects BF16/FP16 source weights, got {dtype}")
        if not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "_scaled_mm"):
            raise RuntimeError("installed PyTorch lacks E4M3 or torch._scaled_mm")
        return {
            "backend": self.name,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "compute_capability": list(capability),
            "source_dtype": str(dtype),
            "fp8_dtype": str(torch.float8_e4m3fn),
            "cutlass_fp8_supported": self._cutlass_supported,
        }

    @torch.no_grad()
    def pack_weight(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if weight.ndim != 2:
            raise RuntimeError(f"FP8 weight must be 2D, got shape={tuple(weight.shape)}")
        # Match SGLang's online Fp8LinearMethod: quantize each output row over
        # the complete K dimension, then transpose to the [K, N] layout expected
        # by apply_fp8_linear / scaled_mm.
        qweight, weight_scale = self._quantize(weight, weight.shape[-1])
        return qweight.t(), weight_scale.t().contiguous()

    def linear(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return self._apply(
            input=x,
            weight=weight,
            weight_scale=weight_scale,
            input_scale=None,
            bias=bias,
            cutlass_fp8_supported=self._cutlass_supported,
            use_per_token_if_dynamic=False,
        )


class SanaFP8PointwiseConv2d(nn.Module):
    """A 1x1 Conv2d executed as an FP8 last-dimension GEMM."""

    def __init__(
        self,
        *,
        module_name: str,
        in_channels: int,
        out_channels: int,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        backend: FP8Backend,
    ) -> None:
        super().__init__()
        self.module_name = module_name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (1, 1)
        self.stride = (1, 1)
        self.padding = (0, 0)
        self.dilation = (1, 1)
        self.groups = 1
        self.backend_name = backend.name
        self._backend = backend
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale = nn.Parameter(weight_scale, requires_grad=False)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach(), requires_grad=False)
        self.fp8_calls = 0
        self._activation_reported = False

    @classmethod
    def from_conv(
        cls,
        conv: nn.Conv2d,
        *,
        module_name: str,
        backend: FP8Backend,
    ) -> "SanaFP8PointwiseConv2d":
        if tuple(conv.kernel_size) != (1, 1):
            raise RuntimeError(f"{module_name}: only 1x1 Conv2d is FP8 eligible")
        if tuple(conv.stride) != (1, 1) or tuple(conv.padding) != (0, 0):
            raise RuntimeError(f"{module_name}: stride/padding are not GEMM-equivalent")
        if tuple(conv.dilation) != (1, 1) or conv.groups != 1:
            raise RuntimeError(f"{module_name}: dilation/groups are not GEMM-equivalent")
        source_weight = conv.weight.detach().squeeze(-1).squeeze(-1)
        qweight, weight_scale = backend.pack_weight(source_weight)
        return cls(
            module_name=module_name,
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            weight=qweight,
            weight_scale=weight_scale,
            bias=conv.bias,
            backend=backend,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise RuntimeError(
                f"{self.module_name}: expected NCHW with C={self.in_channels}, "
                f"got shape={tuple(x.shape)}"
            )
        # The SANA FFN enters Conv2d in channels-last memory format.  In that
        # common path this permutation is a contiguous view; ``contiguous`` is
        # a no-op.  It remains correct for an unusual contiguous-NCHW caller.
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        y_nhwc = self._backend.linear(
            x_nhwc, self.weight, self.weight_scale, self.bias
        )
        self.fp8_calls += 1
        if not self._activation_reported:
            if not torch.isfinite(y_nhwc).all():
                raise RuntimeError(f"{self.module_name}: non-finite FP8 output")
            print(
                "SANA_FP8_MODULE_ACTIVE "
                + json.dumps(
                    {
                        "module": self.module_name,
                        "backend": self.backend_name,
                        "input_shape": list(x.shape),
                        "input_dtype": str(x.dtype),
                        "weight_dtype": str(self.weight.dtype),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            self._activation_reported = True
        return y_nhwc.permute(0, 3, 1, 2)


@dataclass
class SanaFP8InstallReport:
    enabled: bool
    status: str
    policy: dict[str, Any]
    backend: dict[str, Any] = field(default_factory=dict)
    converted_modules: list[str] = field(default_factory=list)
    skipped_modules: list[dict[str, str]] = field(default_factory=list)
    source_weight_bytes: int = 0
    fp8_weight_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def install_sana_fp8(
    transformer: nn.Module,
    policy: SanaFP8Policy | None = None,
    backend: FP8Backend | None = None,
) -> SanaFP8InstallReport:
    """Install selective FP8 modules after the model's BF16 post-load setup."""

    policy = policy or SanaFP8Policy.from_env()
    report = SanaFP8InstallReport(
        enabled=policy.enabled,
        status="off" if not policy.enabled else "preflight",
        policy=asdict(policy),
    )
    if not policy.enabled:
        return report

    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        raise RuntimeError("SANA FP8 target has no transformer_blocks")
    block_start, block_end = policy.validate(len(blocks))
    selected = list(range(block_start, block_end + 1))
    if not selected:
        raise RuntimeError("SANA FP8 policy selected no transformer blocks")

    first_ff = getattr(blocks[selected[0]], "ff", None)
    first_conv = getattr(first_ff, "conv_inverted", None)
    if not isinstance(first_conv, nn.Conv2d):
        raise RuntimeError("SANA FP8 could not locate the first FFN 1x1 projection")

    backend = backend or SGLangH100FP8Backend()
    report.backend = backend.preflight(first_conv.weight.device, first_conv.weight.dtype)

    for index, block in enumerate(blocks):
        ff = getattr(block, "ff", None)
        if ff is None:
            report.skipped_modules.append(
                {"module": f"transformer_blocks.{index}.ff", "reason": "missing_ff"}
            )
            continue
        if index not in selected:
            for leaf in ("conv_inverted", "conv_point"):
                report.skipped_modules.append(
                    {
                        "module": f"transformer_blocks.{index}.ff.{leaf}",
                        "reason": "dense_block_guard",
                    }
                )
            continue
        if getattr(ff, "_fused_gate", False) and not getattr(
            ff, "_fused_gate_ready", False
        ):
            raise RuntimeError(
                f"transformer_blocks.{index}.ff must build its fused gate before FP8 packing"
            )
        for leaf in ("conv_inverted", "conv_point"):
            name = f"transformer_blocks.{index}.ff.{leaf}"
            original = getattr(ff, leaf, None)
            if not isinstance(original, nn.Conv2d):
                message = f"expected Conv2d, got {type(original).__name__}"
                if policy.strict:
                    raise RuntimeError(f"{name}: {message}")
                report.skipped_modules.append({"module": name, "reason": message})
                continue
            source_bytes = original.weight.numel() * original.weight.element_size()
            converted = SanaFP8PointwiseConv2d.from_conv(
                original, module_name=name, backend=backend
            )
            setattr(ff, leaf, converted)
            report.converted_modules.append(name)
            report.source_weight_bytes += source_bytes
            report.fp8_weight_bytes += (
                converted.weight.numel() * converted.weight.element_size()
                + converted.weight_scale.numel()
                * converted.weight_scale.element_size()
            )

    expected = len(selected) * 2
    if policy.strict and len(report.converted_modules) != expected:
        raise RuntimeError(
            "SANA FP8 strict installation converted "
            f"{len(report.converted_modules)} modules, expected {expected}"
        )
    report.status = "installed" if report.converted_modules else "fallback_only"
    if report.status != "installed" and policy.strict:
        raise RuntimeError("SANA FP8 strict installation produced no converted modules")
    print("SANA_FP8_INSTALL " + json.dumps(report.to_dict(), sort_keys=True), flush=True)
    return report


def collect_sana_fp8_runtime(transformer: nn.Module) -> dict[str, Any]:
    modules = [
        module
        for module in transformer.modules()
        if isinstance(module, SanaFP8PointwiseConv2d)
    ]
    return {
        "converted_module_count": len(modules),
        "active_module_count": sum(module.fp8_calls > 0 for module in modules),
        "total_fp8_calls": sum(module.fp8_calls for module in modules),
        "modules": [
            {
                "module": module.module_name,
                "backend": module.backend_name,
                "calls": module.fp8_calls,
                "weight_dtype": str(module.weight.dtype),
            }
            for module in modules
        ],
    }
