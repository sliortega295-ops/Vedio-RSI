# SANA-Video 2B H100 Kernel Search Draft

## Task contract

- Task: `sana2b-kernel_aw-0001`.
- Objective: find the fastest measured composed, mathematically lossless local
  implementation of the frozen SANA-Video 2B 480p workload on the leased single
  H100. Preserve the complete search trajectory, including failed candidates.
- Fixed input/output: the frozen model-card prompt, negative prompt, seed 42,
  832x480, 81 frames at 16 fps, 50 DPMSolver steps, guidance 6, one GPU, BF16
  transformer/text encoder, FP32 VAE, and an 832x480/81-frame MP4.
- Correctness: preserve the same model and scheduler, all 50 denoising steps,
  both CFG branch evaluations per step, all 20 transformer blocks per DiT call,
  and the same mathematical operators. Fusion, reordering, compilation,
  invariant preparation caches, and BF16 execution are allowed. Approximate
  reuse, step/model-call reduction, sparsity, rank reduction, and sub-16-bit
  quantization are forbidden. Correctness is argued from the method and
  structural counts; output differences are not a lossless rejection gate.
- Runtime constraints: local-mode launcher only; fixed active lease
  `GPU-847305ce-670b-91ee-e0a9-aa3b7833df23`; every run goes through the
  supplied wrapper and shared flock. The frozen baseline and its lock are never
  rerun or edited. Shared model, environment, dependency overlay, and kernel
  staging are read-only.
- Validation command: the exact local launch command for a committed config,
  followed by `scripts/collect_run.py`, `search/plan_eval.py --no-gemini
  --no-refresh-collection --assess`, structural receipt checks, ffprobe/three
  decoded-frame authenticity inspection, and trajectory validation.
- Evaluation: compare candidate `benchmark.total_s` against frozen 61.7 seconds
  using the identical timing scope recorded in `BASELINE-LOCK.json`.
- Promotion: retain only an authentic valid run whose method is lossless and
  whose latency or peak memory measurably improves the current canonical stack.
  Stop at 2.5x, round 40, or a real plateau after roughly 3-4 genuinely distinct
  no-new-best hypotheses.

## Frozen baseline and active graph

`BASELINE-LOCK.json` SHA-256 is
`3fdafdf00554ae4bafc91bc7729c3ba3e96af4edebac809782e6d6122ed23954`.
It records 61.7 seconds and 17,770 MiB runtime peak memory. The baseline log
provides a stage-level warm-path profile without rerunning it: text encoding
0.3638 s, denoising 44.1425 s, decoding 3.0510 s, and 47.56 s for the warmed
request. Thus denoising is 92.8% of the warmed request, so the initial search
must target the repeated DiT path rather than VAE or host startup.

The current source has 20 blocks, hidden width 2240 (20 heads x 112), linear
self-attention with 3D RoPE, softmax cross-attention, and a convolutional FFN
with expansion ratio 3.0. For the frozen latent geometry, the expected latent
grid is 21x60x104 before the DiT patch embedding and 21x30x52 after patching,
or 32,760 tokens. This shape is an architecture-derived expectation to verify
in live receipts, not a measured profiler claim.

The runtime already exposes independent lossless switches for default
`torch.compile`, max-autotune compilation, BF16 linear-attention aggregation,
and merged self-attention QKV. These are candidate mechanisms, not accepted
performance claims. The source comment that FP32 linear-attention aggregation
was about 9% of DiT is prior implementation evidence and must be remeasured on
this exact H100 workload.

## Risks and unknowns

- The frozen benchmark's 61.7-second `gen.generate` interval includes the
  generator's internal warmup bookkeeping even though the durable scope label
  describes warm generation. Candidate comparison must use the same wrapper and
  denominator rather than reinterpret the number.
- Inductor cold compilation can dominate process wall time, fail on unsupported
  graph fragments, or expose cache-dependent variance. Record cold build time
  separately and accept only the wrapper's identical benchmark field.
- Max-autotune has a documented grouped-convolution hang risk in another CUDA
  substrate; do not try it until default compilation proves the graph and use a
  bounded process/cache-specific attempt.
- The linear-attention FP32-to-BF16 aggregation is allowed by the contract but
  changes floating-point execution order/precision. Its admissibility is method
  based; no output-difference gate is permitted.
- Cross-attention conditioning and full RoPE grids appear invariant across
  denoising calls, but caching them is valid only after proving shape/device/
  dtype and CFG-branch keys.
- A single full generation is noisy. A small improvement needs confirmation or
  should be rejected as not measurable; large improvements can be promoted from
  the canonical run with exact provenance.

## Ranked candidate families

1. Default Inductor compilation of the stable repeated transformer blocks.
   Highest expected integrated contribution and already wired as an isolated
   switch; first real round.
2. BF16 linear-attention KV aggregation on top of the retained stack. It
   replaces slow FP32 matmuls with BF16 tensor-core equivalents while retaining
   the same attention formula.
3. Merged bias-free self-attention QKV projection, reducing three equivalent
   GEMM launches to one.
4. Exact cross-attention K/V preparation cache keyed by immutable conditioning,
   branch, dtype, device, and shape; preserve both CFG DiT calls.
5. Exact full-grid 3D RoPE layout cache keyed by latent geometry/device/dtype.
6. Max-autotune compilation after default compile establishes a stable graph.
7. Profile-justified norm/modulation/gate or FFN glue fusion only if the
   accumulated stack still leaves meaningful launch/memory overhead.

## First implementation and required evidence

Create one config that changes only `SANA_ENABLE_COMPILE=1`, commit it, and run:

```text
/home/jiangzhikun/yongyan_liu/Experiments/SolAgent/20260827-official-repro/envs/sana-cu128-phase0/bin/python scripts/launch_config.py config/sana_video_2b_h100/kernel_r01_compile_default.toml --mode local --run-root runs --name-suffix r01-compile-default
```

Promotion requires the committed manifest and parent SHA, wrapper build marker,
real MP4/benchmark/run-config/ffprobe/frame receipts, exact workload and GPU UUID,
lossless method/count argument, assessed speed versus 61.7 seconds, and one
contiguous `TRAJECTORY.jsonl` record. A compile/build/runtime failure is rejected
but recorded and still consumes round 1.
