# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
import os as _os
import sys as _sys

if _os.environ.get("SGLANG_SANA_MINIMAL_IMPORT") == "1":
    # multiprocessing uses a fresh interpreter with the spawn start method.
    # Install the same SANA-only registry alias there before pipelines_core
    # imports the broad optional-model registry.
    from sglang.multimodal_gen import registry_sana as _sana_registry

    _sys.modules["sglang.multimodal_gen.registry"] = _sana_registry

from sglang.multimodal_gen.configs.pipeline_configs import PipelineConfig
from sglang.multimodal_gen.configs.sample import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

__all__ = ["DiffGenerator", "PipelineConfig", "SamplingParams"]

del _os, _sys
if "_sana_registry" in globals():
    del _sana_registry

# Trigger multimodal CI tests
