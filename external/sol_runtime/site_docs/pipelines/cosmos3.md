# Cosmos3-Super 64B

Cosmos3-Super is the largest model here — a 64B dual-pathway DiT (an Understanding
pathway caches text K/V once; a Generation pathway cross-attends from noisy visual
tokens every step). It runs **4-GPU sequence-parallel**. Its acceleration line is
**cache + low-precision quant**; attention stays dense (the GQA cross-attention is
already cheap relative to the 64B linears).

## Acceleration line: TeaCache + step-selective NVFP4

| Component | What it does | Type | Contribution |
|---|---|---|---|
| [TeaCache](../techniques/cache/teacache.md) | skip GEN steps by accumulated rel-L1 of timestep-modulated input | cache (lossy) | the bulk |
| [Step-selective NVFP4](../techniques/quant/nvfp4.md) | TE 4-bit on GEN linears (gate_up/down/qkv/out), first/last 3 steps BF16 | quant (lossy) | ~1.2× on top |

**Why this set.** At 64B the per-step cost is dominated by the GEN-layer linears, so
4-bit GEMMs (NVFP4) give a large per-step win — but FP4 hurts the most quality-
sensitive first/last steps, so those stay BF16 (step-selective). TeaCache then removes
whole redundant steps. Sparse attention isn't needed: the cross-attention is GQA
(64 q-heads / 8 kv-heads) and small next to the linears.

## Shipped config

TeaCache `thr 1.15 / start 10 / max-continuous 3`; NVFP4 `targets=gate_up,down,qkv,out`,
`skip_first_steps=3`, `skip_last_steps=3`. NVFP4 needs Blackwell + `transformer_engine`;
on older GPUs it gracefully falls back to BF16 (no crash).

## Prompts

Cosmos3 expects **structured-JSON** prompts — a bare sentence yields poor quality
(no reasoner expansion). The official robot-plate prompt ships at
`prompts/cosmos/robot_plate.json` and must be passed explicitly via `PROMPT_FILE`
(no built-in default).

## Run

```bash
MODEL_REPO=nvidia/Cosmos3-Super ROOT=outputs/cosmos3 \
PROMPT_FILE=prompts/cosmos/robot_plate.json PROMPT_TAG=robot_plate \
bash scripts/cosmos/slurm_cosmos3_super.sh baseline   # or: fullopt
```

Official spec: 1280×720, 189f, 24fps, 35 steps, guidance 6.0, flow_shift 10.0,
max_seq 4096, 4-GPU sequence parallel.

## Measured (GB200, warmup-excluded)

| config | warm | speedup |
|---|---|---|
| baseline (dense) | 97.2 s | 1.00× |
| fullopt: TeaCache 1.15/10/3 + NVFP4(first3/last3 dense) | 43.1 s | **~2.26×** |

(TeaCache alone ≈ 2.18×; NVFP4 adds ~1.2× on top. A more aggressive TeaCache
threshold trades quality for up to ~2.66×.)
