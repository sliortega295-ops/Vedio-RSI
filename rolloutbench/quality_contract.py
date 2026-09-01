from __future__ import annotations

from types import MappingProxyType


VBENCH_REF = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
FORMAL_CACHE_IDS = ("C02", "C03", "C04", "C06", "C07", "C09", "C10", "C11", "C12")
EXCLUDED_CACHE_IDS = MappingProxyType(
    {
        "C01": "provenance_failed",
        "C05": "calibration_only",
        "C08": "calibration_only",
    }
)
QUALITY_SEEDS = (42, 12345)
QUALITY_DIMENSIONS = (
    "subject_consistency",
    "motion_smoothness",
    "background_consistency",
    "temporal_flickering",
    "aesthetic_quality",
    "imaging_quality",
    "overall_consistency",
)
QUALITY_METRICS_BY_SUITE = MappingProxyType(
    {
        "subject_consistency": ("subject_consistency", "motion_smoothness"),
        "scene": ("background_consistency",),
        "temporal_flickering": ("temporal_flickering",),
        "overall_consistency": (
            "aesthetic_quality",
            "imaging_quality",
            "overall_consistency",
        ),
    }
)
MAX_MEAN_RELATIVE_DROP = 0.005
MAX_SINGLE_DIMENSION_DROP = 0.02
