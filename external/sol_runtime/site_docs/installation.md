# Installation

A fresh conda env, then the editable install plus the CUDA-JIT fixups. Target
hardware is CUDA-13 class (Blackwell / GB200); Cosmos3-Super needs 4 GPUs,
SANA-Video and LTX-2.3 run on 1.

## 1. Create the environment

```bash
git clone https://github.com/lyttttt3333/sol-infer.git && cd sol-infer
PYTHON_VERSION=3.12 bash scripts/create_code_conda_env.sh   # -> ./.conda/ltx23 (Python 3.12)
conda activate "$PWD/.conda/ltx23"
```

## 2. Install dependencies

```bash
uv pip install -e "$PWD/python[diffusion]" --prerelease=allow
```

Shipped stack: torch 2.11+cu130, diffusers 0.38.

## 3. CUDA JIT fixups

The editable install pulls only the runtime CUDA libs; the runtime kernel JIT also
needs the compiler toolchain, CCCL headers, and dev symlinks. One idempotent step:

```bash
PYTHON_BIN=.conda/ltx23/bin/python bash scripts/postinstall_cuda_jit.sh   # [--with-te]
```

Add `--with-te` to install TransformerEngine for the NVFP4 `fullopt` path
(Cosmos3 / LTX); without it those gracefully fall back to BF16.

## 4. Verify

```bash
.conda/ltx23/bin/python -c "import torch, diffusers, sglang; print(torch.__version__, diffusers.__version__)"
# expect: 2.11.0+cu130 0.38.0
```

Next: get the [model weights](model_zoo.md), then run a [pipeline](pipelines/sana.md).
(A CUDA-driver-too-old warning on a GPU-less login node is expected — run on a GPU node.)
