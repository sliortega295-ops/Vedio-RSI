from __future__ import annotations

from types import MappingProxyType


VBENCH_REF = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
DENSE_REFERENCE_ID = "DENSE"
DENSE_RUNTIME_REF = "8bd01c6898f920c140a9c74197676debbcaff1fe"
DENSE_RUNTIME_PARENT_REF = "2c6bdf06db4dd3be507720b25512217e1f3ae5e9"
DENSE_CONFIG_PATH = "config/sana_video_2b_h100/baseline.toml"
DENSE_CONFIG_SHA256 = "6b61041dd4441ed3b84bad3756a2b656ba4e49d7cba3be66e9c7abda4c2f083e"
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

K22_FAILURE_CONTRACT = MappingProxyType(
    {
        "kind": "real_fail_closed_layout_mismatch",
        "stage": "generate",
        "deterministic": True,
        "episode_id": "K22",
        "failure_code": "SANA_K22_NONCONTIGUOUS_SDPA_LAYOUT",
        "expected_log_marker": (
            "SANA cross-attention output layout requires contiguous input"
        ),
        "child_returncode": 4,
        "config_id": "sana_video_2b_h100_kernel_r22_cross_output_layout",
        "config_sha256": (
            "8e659124c573beb8feeb0e532960f3ed15241588ed9c525b5b2afd1cfbb70b57"
        ),
        "runtime_ref": "ef4dcd633c811ad76d0d997db08972f6bb525c31",
    }
)
