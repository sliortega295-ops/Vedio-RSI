# Copyright 2025 SGLang authors
"""H100 FP8 FFN load-time transform.

This mirrors :mod:`nvfp4_ffn`: the transform owns the FFN-precision seam and
declares the policy through environment variables consumed during model
post-load.  SANA-Video supplies the model-specific 1x1-convolution adapter.
"""

from __future__ import annotations

from sglang.multimodal_gen.runtime.efficiency.registry import register_transform
from sglang.multimodal_gen.runtime.efficiency.technique import Seam
from sglang.multimodal_gen.runtime.efficiency.transform import (
    ModelTransform,
    TransformContext,
    TransformPhase,
)


@register_transform("fp8_ffn")
class FP8FFN(ModelTransform):
    """Enable selective W8A8 E4M3 FFN projections on native FP8 GPUs."""

    name = "fp8_ffn"
    phase = TransformPhase.LOAD
    writes = frozenset({Seam.FFN_PRECISION})

    def __init__(
        self,
        scope: str = "ffn_1x1",
        block_start: int = 0,
        block_end: int = -1,
        strict: bool = True,
    ) -> None:
        self.scope = scope
        self.block_start = block_start
        self.block_end = block_end
        self.strict = strict

    def applies_to(self, spec) -> bool:
        return getattr(spec, "name", "") == "SanaVideo"

    def set_env(self, ctx: TransformContext) -> None:
        ctx.env.update(
            {
                "SGLANG_SANA_FP8": "1",
                "SGLANG_SANA_FP8_SCOPE": self.scope,
                "SGLANG_SANA_FP8_BLOCK_START": str(self.block_start),
                "SGLANG_SANA_FP8_BLOCK_END": str(self.block_end),
                "SGLANG_SANA_FP8_STRICT": "1" if self.strict else "0",
            }
        )
