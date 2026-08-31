# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""Pipeline stages for diffusion models."""

import os as _os

from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.encoding import EncodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageEncodingStage,
    ImageVAEEncodingStage,
    LTX2ImageEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
    LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation import (
    TimestepPreparationStage,
)

_SANA_MINIMAL_IMPORT = _os.environ.get("SGLANG_SANA_MINIMAL_IMPORT") == "1"

__all__ = [
    "PipelineStage",
    "InputValidationStage",
    "TimestepPreparationStage",
    "LatentPreparationStage",
    "DenoisingStage",
    "EncodingStage",
    "DecodingStage",
    "ImageEncodingStage",
    "ImageVAEEncodingStage",
    "LTX2ImageEncodingStage",
    "TextEncodingStage",
]

if not _SANA_MINIMAL_IMPORT:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import (
        CausalDMDDenoisingStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.comfyui_latent_preparation import (
        ComfyUILatentPreparationStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding_av import (
        LTX2AVDecodingStage,
        LTX2Stage1ExportStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising_av import (
        LTX2AVDenoisingStage,
        LTX2RefinementStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising_dmd import (
        DmdDenoisingStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.hunyuan3d_paint import (
        Hunyuan3DPaintPostprocessStage,
        Hunyuan3DPaintPreprocessStage,
        Hunyuan3DPaintTexGenStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.hunyuan3d_shape import (
        Hunyuan3DShapeBeforeDenoisingStage,
        Hunyuan3DShapeDenoisingStage,
        Hunyuan3DShapeExportStage,
        Hunyuan3DShapeSaveStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation_av import (
        LTX2AVLatentPreparationStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.ltx_2_denoising import (
        LTX2DenoisingStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.text_connector import (
        LTX2TextConnectorStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.upsampling import (
        LTX2HalveResolutionStage,
        LTX2LoRASwitchStage,
        LTX2UpsampleStage,
    )

    __all__.extend(
        [
            "ComfyUILatentPreparationStage",
            "LTX2AVLatentPreparationStage",
            "DmdDenoisingStage",
            "LTX2DenoisingStage",
            "LTX2AVDenoisingStage",
            "CausalDMDDenoisingStage",
            "LTX2AVDecodingStage",
            "LTX2Stage1ExportStage",
            "LTX2TextConnectorStage",
            "Hunyuan3DShapeBeforeDenoisingStage",
            "Hunyuan3DShapeDenoisingStage",
            "Hunyuan3DShapeExportStage",
            "Hunyuan3DShapeSaveStage",
            "Hunyuan3DPaintPreprocessStage",
            "Hunyuan3DPaintTexGenStage",
            "Hunyuan3DPaintPostprocessStage",
            "LTX2RefinementStage",
            "LTX2HalveResolutionStage",
            "LTX2LoRASwitchStage",
            "LTX2UpsampleStage",
        ]
    )

del _os
