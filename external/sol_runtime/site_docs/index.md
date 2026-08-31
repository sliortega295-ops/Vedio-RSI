# Sol-Engine

**Accelerated video-diffusion inference — SANA-Video · Cosmos3-Super · LTX-2.3.**

[:material-github: GitHub](https://github.com/lyttttt3333/sol-infer){ .md-button }
[:material-rocket-launch: Get started](installation.md){ .md-button }

Three production video-diffusion models, each reduced to **one clean acceleration
line** (plus a dense `baseline`), composed from **five reusable acceleration
methods**. All speedups below are GB200, warmup-excluded, at each model's official spec.

## Models

| Model | Acceleration line | Speedup |
|---|---|---|
| **[SANA-Video](pipelines/sana.md)** (2B, 1 GPU) | EasyCache + fusion + compile | **2.1× / 2.56×** |
| **[Cosmos3-Super](pipelines/cosmos3.md)** (64B, 4 GPU) | TeaCache + step-selective NVFP4 | **~2.26×** |
| **[LTX-2.3](pipelines/ltx.md)** (1080p/10s, 1 GPU) | KWL fusion + cache + PISA + NVFP4 + token-prune | **~2.4×** |

Each entry takes `baseline | fullopt`. See **[Optimized pipelines](pipelines/sana.md)**.

## The five acceleration methods

1. **[Cache (step-skip)](techniques/cache/index.md)** — TeaCache / EasyCache / SCSP / …
2. **[Quantization](techniques/quant/index.md)** — NVFP4 (step-selective) / FP8 / MXFP4 / SVDQuant
3. **[Kernel fusion (KWL)](techniques/kernel/index.md)** — lossless AdaLN / qknorm+RoPE / FFN / gate fusions + compile
4. **[Sparse attention](techniques/sparse/index.md)** — PISA / SVG / VSA / STA / VMoBA
5. **[Token pruning](techniques/token_prune/index.md)** — drop low-salience video tokens at mid refine steps

Each model pipeline is one specific assembly of a subset of these five — see
**[Acceleration techniques](techniques/cache/index.md)**.

## Quick start

```bash
PYTHON_VERSION=3.12 bash scripts/create_code_conda_env.sh && conda activate ./.conda/ltx23
uv pip install -e "python[diffusion]" --prerelease=allow
PYTHON_BIN=.conda/ltx23/bin/python bash scripts/postinstall_cuda_jit.sh
bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh fullopt
```

Full setup: **[Installation](installation.md)** · weights: **[Model Zoo](model_zoo.md)**.
