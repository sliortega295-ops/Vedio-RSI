#!/usr/bin/env bash
set -euo pipefail
"${SOL_A100_PYTHON:?missing SOL_A100_PYTHON}" "${AUTOVIDEO_REPO_ROOT:?missing AUTOVIDEO_REPO_ROOT}/models/sana_video_2b_a100/baseline/gpu_infer.py"
