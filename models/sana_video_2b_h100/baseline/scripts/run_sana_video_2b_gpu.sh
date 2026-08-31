#!/usr/bin/env bash
set -euo pipefail

: "${OUT_DIR:?OUT_DIR must be set by scripts/launch_config.py}"
: "${AUTOVIDEO_REPO_ROOT:?AUTOVIDEO_REPO_ROOT must be set}"
: "${SANA_PYTHON_BIN:?SANA_PYTHON_BIN must be set}"
: "${SANA_DEPENDENCY_OVERLAY:?SANA_DEPENDENCY_OVERLAY must be set}"
: "${SANA_KERNEL_STAGING:?SANA_KERNEL_STAGING must be set}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
BASELINE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)
export SANA_RUNTIME_ROOT=${SANA_RUNTIME_ROOT:-${AUTOVIDEO_REPO_ROOT}/external/sol_runtime}
export PYTHONPATH=${SANA_RUNTIME_ROOT}/python:${SANA_DEPENDENCY_OVERLAY}:${SANA_KERNEL_STAGING}${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=${HF_HOME:-/home/jiangzhikun/yongyan_liu/Experiments/SolAgent/20260827-official-repro/models/hf-cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${AUTOVIDEO_REPO_ROOT}/caches/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${AUTOVIDEO_REPO_ROOT}/caches/torch_extensions/inductor}
export TMPDIR=${TMPDIR:-${AUTOVIDEO_REPO_ROOT}/caches/tmp}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "${OUT_DIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TMPDIR}"
exec "${SANA_PYTHON_BIN}" "${BASELINE_DIR}/gpu_infer.py"
