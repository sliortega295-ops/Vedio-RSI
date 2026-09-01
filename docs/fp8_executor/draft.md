# SANA-Video H100 FP8 Executor Draft

## Task contract

- Task: `sana2b-fp8-executor`.
- Objective: implement an independently schedulable FP8 executor for the
  SANA-Video 2B Sol-Agent, modelled on Sol-Engine's load-time NVFP4 transform,
  and make its best retained result composable with the accepted Kernel R20 +
  EasyCache R12 baseline.
- Fixed input/output: the pinned SANA-Video 2B revision, prompt files, seed 42,
  832x480, 81 frames, 16 fps, 50 DPMSolver steps, guidance 6, BF16 text encoder,
  FP32 VAE, and one H100. Quantization may change transformer numerics but must
  not change the model, scheduler, prompt, seed, step count, CFG structure,
  output dimensions, or video contract.
- Correctness and quality: OFF must preserve the current BF16 path exactly.
  ON must prove that native FP8 kernels executed, report every converted and
  fallback module, produce finite tensors, pass an H100 module smoke with cosine
  similarity at least 0.995 and relative RMSE at most 0.10, and produce a valid
  832x480/81-frame video without critical visual corruption. These module
  tolerances are an implementation sanity gate, not a claim that FP8 is
  mathematically lossless.
- Hardware and dependencies: NVIDIA H100/SM90, CUDA 12.8, PyTorch 2.11, the
  existing experiment-local SGLang runtime and staged `sgl_kernel`. No new
  checkpoint copy and no mutation of shared model, environment, dependency,
  or kernel-staging directories. Unsupported hardware or kernels must fail
  closed or use an explicitly recorded BF16 fallback; a fallback-only run is
  not an FP8 performance result.
- Validation command: local orchestration/manifest/unit tests followed by the
  H100 FP8 component smoke in the isolated 20260901 project.
- Evaluation command: launch a committed FP8 manifest with
  `scripts/launch_config.py`, collect the canonical benchmark and output
  receipts, and compare it with the matching accepted integrated config under
  the same GPU, prompt, timing scope, and warmup policy.
- Promotion: retain a candidate only if it has real FP8 invocation evidence,
  passes the module and video gates, and improves matching integrated latency by
  at least 2%. If no candidate clears that bar, deliver the component and the
  measured structured negative rather than relabeling BF16 fallback as FP8.

Post-run boundary: the completed H100 runs evaluated an integrated FP8 +
EasyCache-retune candidate. The isolated FP8-only full-generation run required
for pure executor attribution was not run, so the code/component may be marked
validated and the integrated point accepted bounded, but no official isolated
executor `DELIVERY.json` or pure-FP8 speedup may be claimed.

## Current baseline and implementation surface

The accepted H100 branch records 61.7 seconds dense, 29.2 seconds for integrated
Prompt 1, and 33.0 seconds for integrated Prompt 2. The current integrated
recipe is Kernel R20 plus balanced EasyCache R12; it has no FP8/NVFP4 path.
SANA-Video's FFN is `GLUMBTempConv`, not the linear GELU FFN used by LTX, so a
literal rename of the NVFP4 transform would be a no-op. The useful H100 target
is the two 1x1 FFN convolutions, expressed as equivalent last-dimension GEMMs,
plus optional profiled attention projections in later candidates.

The runtime already contains mature FP8 primitives:

- online E4M3 weight packing;
- dynamic activation quantization;
- `apply_fp8_linear` dispatch to H100 FP8 GEMM;
- explicit weight/input scales and BF16 output.

The new component should reuse those primitives. It should not introduce a
second hand-written scaled-matmul implementation.

## Main risks and unknowns

- The validated Python environment sees H100 SM90 and `torch._scaled_mm`, but
  `sgl_kernel` is available only when the existing staging path is injected by
  the launcher.
- FP8 conversion overhead may erase GEMM savings, particularly for small
  projections. Module scope must therefore be measured, not assumed.
- The accepted QKV-merge path owns a detached BF16 packed weight. Quantizing
  attention projections without making the packed lifecycle explicit can
  silently bypass FP8 or duplicate weights.
- Re-expressing a 1x1 Conv2d as a last-dimension GEMM is shape-equivalent, but
  layout conversions must be proven to be views or measured as part of the
  candidate.
- FP8 is lossy. A valid MP4 alone is insufficient; activation/fallback receipts
  and bounded numerical and visual checks are required.
- The previous GPU UUID 7 is currently occupied by another process. GPU 6 was
  idle at preflight, but ownership must be rechecked immediately before every
  launch and acquired through a new experiment-local lock.

## Ranked candidate directions

1. `ffn_1x1`: dynamically quantized W8A8 FP8 for `conv_inverted` and
   `conv_point`, keeping temporal/depthwise convolutions, attention, embeddings,
   and output heads in BF16. This most closely mirrors the paper's selective
   NVFP4 FFN scope.
2. `ffn_1x1_guarded`: keep the first and last transformer blocks BF16 if the
   full-stack quality check needs a conservative guard.
3. `hot_linear`: add the large merged self-attention QKV and output projection
   only after the packed-weight lifecycle is tested and profiling shows value.
4. Reject broad `all_linear` conversion unless earlier candidates demonstrate
   both quality headroom and positive incremental latency.

## Initial implementation steps

1. Add a registered load-time `FP8FFN` model transform that owns the exclusive
   FFN-precision seam and emits model-neutral FP8 policy variables.
2. Add a SANA-specific FP8 module using the existing SGLang quantization and
   GEMM dispatcher, with capability checks, OFF identity, module selection,
   block guards, counters, and a machine-readable summary.
3. Wire the SANA FFN 1x1 projections after BF16 weight loading and before the
   first measured request.
4. Add the `fp8` orchestration registry entry and a compact six-round executor
   workflow with quality-gated delivery; add it to the SANA default technique
   list after Kernel and Cache.
5. Add manifests and tests, then run local static/unit validation before any GPU
   launch.

## Evidence required

- source revision and diff;
- registered transform and executor registry tests;
- OFF-path identity test;
- module-selection and fallback tests;
- H100 capability/backend receipt;
- converted-module list and real FP8 call count;
- module numerical smoke result;
- matching integrated OFF/FP8 ON benchmark receipts;
- valid video metadata and limited visual comparison;
- a candidate ledger recording retained, rejected, blocked, or structured
  negative status.
