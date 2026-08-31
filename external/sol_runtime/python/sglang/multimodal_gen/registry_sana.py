"""Opt-in SANA-Video-only registry for the CUDA 12.8 baseline adapter.

This module only narrows discovery. It reuses the existing SANA config,
sampling, and pipeline classes without changing their implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_PIPELINE_REGISTRY: dict[str, type[Any]] = {}
_PIPELINE_CONFIG_REGISTRY: dict[str, tuple[type[Any], type[Any]]] = {}


@dataclass
class ConfigInfo:
    sampling_param_cls: type[Any]
    pipeline_config_cls: type[Any]


@dataclass
class ModelInfo:
    pipeline_cls: type[Any]
    sampling_param_cls: type[Any]
    pipeline_config_cls: type[Any]


def _normalized_model_path(model_path: str) -> str:
    return str(model_path).lower().replace("_", "-")


def _is_sana_video(model_path: str) -> bool:
    normalized = _normalized_model_path(model_path)
    return "sana-video" in normalized or "sana--video" in normalized


def _is_720p(model_path: str) -> bool:
    return "720p" in _normalized_model_path(model_path)


def _get_sana_config_classes(model_path: str):
    from sglang.multimodal_gen.configs.pipeline_configs.sana_video import (
        SanaVideo720PPipelineConfig,
        SanaVideoPipelineConfig,
    )
    from sglang.multimodal_gen.configs.sample.sana_video import (
        SanaVideoSamplingParams,
    )

    pipeline_config_cls = (
        SanaVideo720PPipelineConfig if _is_720p(model_path) else SanaVideoPipelineConfig
    )
    return pipeline_config_cls, SanaVideoSamplingParams


def _discover_and_register_pipelines() -> None:
    if _PIPELINE_REGISTRY:
        return

    from sglang.multimodal_gen.runtime.pipelines.sana_video import SanaVideoPipeline

    _PIPELINE_REGISTRY["SanaVideoPipeline"] = SanaVideoPipeline
    config_480p, sampling_cls = _get_sana_config_classes(
        "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"
    )
    config_720p, _ = _get_sana_config_classes(
        "Efficient-Large-Model/SANA-Video_2B_720p_diffusers"
    )
    _PIPELINE_CONFIG_REGISTRY["SanaVideoPipeline"] = (config_480p, sampling_cls)
    _PIPELINE_CONFIG_REGISTRY["SanaVideo720PPipeline"] = (config_720p, sampling_cls)


def get_pipeline_config_classes(
    pipeline_class_name: str,
) -> tuple[type[Any], type[Any]] | None:
    _discover_and_register_pipelines()
    return _PIPELINE_CONFIG_REGISTRY.get(pipeline_class_name)


def _get_config_info(
    model_path: str, model_id: str | None = None
) -> ConfigInfo | None:
    del model_id
    if not _is_sana_video(model_path):
        return None
    pipeline_config_cls, sampling_param_cls = _get_sana_config_classes(model_path)
    return ConfigInfo(
        sampling_param_cls=sampling_param_cls,
        pipeline_config_cls=pipeline_config_cls,
    )


def get_model_info(
    model_path: str,
    backend: Any = None,
    model_id: str | None = None,
) -> ModelInfo | None:
    backend_value = getattr(backend, "value", backend)
    if backend_value not in (None, "auto", "sglang"):
        raise ValueError(
            "SANA minimal registry only supports the native sglang backend"
        )

    config_info = _get_config_info(model_path, model_id=model_id)
    if config_info is None:
        return None

    _discover_and_register_pipelines()
    return ModelInfo(
        pipeline_cls=_PIPELINE_REGISTRY["SanaVideoPipeline"],
        sampling_param_cls=config_info.sampling_param_cls,
        pipeline_config_cls=config_info.pipeline_config_cls,
    )


def has_registered_diffusion_model_path(model_path: str) -> bool:
    return _is_sana_video(model_path)


def get_non_diffusers_pipeline_name(model_path: str) -> None:
    del model_path
    return None


__all__ = [
    "ConfigInfo",
    "ModelInfo",
    "_PIPELINE_CONFIG_REGISTRY",
    "_PIPELINE_REGISTRY",
    "_discover_and_register_pipelines",
    "_get_config_info",
    "get_model_info",
    "get_pipeline_config_classes",
    "get_non_diffusers_pipeline_name",
    "has_registered_diffusion_model_path",
]
