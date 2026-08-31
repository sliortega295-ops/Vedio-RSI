# Operator fusions

Each fusion below is a hand-written Triton / CuTeDSL kernel that replaces a chain of
eager ops with **one launch and one HBM round-trip**. They are listed with the chain
they collapse, *why* that chain is wasteful eagerly, and what the fused kernel does
instead. Source: `jit_kernel/diffusion/triton/ltx2_*.py` and
`jit_kernel/diffusion/cutedsl/`.

---

## RMS + AdaLN modulation

**Eager chain.** `y = rms_norm(x) ; s = 1 + scale ; y = y * s ; y = y + shift` — that is
one reduction kernel plus three elementwise kernels, each reading and writing the full
`[B, S, D]` activation (4 HBM passes, 4 launches).

**Fused** (`ltx2_rms_norm_modulate`). One Triton program per `[B,S]` row: load the row,
compute the mean-square reduction in registers, then apply `(x·rstd)·(1+scale)+shift`
in the same pass and store once → **1 launch, 1 read + 1 write**. The kernel deliberately
replays the eager bf16 rounding (`norm→bf16→fp32`, `(scale+1)→bf16`, …) so it is
bit-faithful.

The CuTeDSL `scale_residual_norm_scale_shift` variant folds the *preceding* residual
add in too: `residual + x·gate → rms_norm → ·(1+scale)+shift`, so the residual write
that would feed the next block never touches HBM separately.

## Q/K-norm + split RoPE

**Eager chain.** Per head: RMS-normalize Q and K with their learned weights, cast to
bf16, then apply rotary embedding (split-half rotation with cos/sin) — separately for Q
and for K. That is normalize-write, then re-read for RoPE, ×2 tensors.

**Fused** (`ltx2_qknorm_split_rope_pair`). One kernel handles **both Q and K**: it loads
the two halves of each head (`first_col`, `second_col = first_col + half_dim`), does the
RMS reduction over the head, applies the per-channel norm weight, then immediately does
the rotation `out_first = q₁·cos − q₂·sin`, `out_second = q₂·cos + q₁·sin` — all in
registers. The normalized values never go to HBM and back; cos/sin are read once.

## Dual modulation

**Eager chain.** A dual-stream block (audio↔video, A2V / V2A) RMS-normalizes the *same*
hidden state and then modulates it **twice** with two different `(scale, shift)` pairs
→ the norm is recomputed or re-read for each branch.

**Fused** (`ltx2_rmsnorm_dual_modulate`). Compute `normed = rms_norm(x)` once, keep it
in registers, and emit both `y0 = normed·(1+scale0)+shift0` and
`y1 = normed·(1+scale1)+shift1` from the single load.

**CA-dual variant** (`ltx2_rmsnorm_ca_dual_modulate_from_temb`) goes further and folds
the Ada-value construction into the same kernel: the `(scale, shift)` are not
pre-materialized but computed inside as `table[i] + temb[i]`, so the timestep-embedding
add, the table broadcast, the norm, and both modulations are one kernel.

## Ada values (all-9)

**Eager chain.** LTX's AdaLN draws 9 modulation tensors (scale/shift/gate for several
sub-layers) as `scale_shift_table[i] + temb[:, :, i]` — a broadcasted add plus a slice
for each of the 9, i.e. 9 tiny kernels each writing a `[B,S,D]` tensor.

**Fused** (`ltx2_ada_values9` / `…_packed`). One kernel reads the timestep row once and
writes all 9 outputs (the `_packed` form writes a single `[9,B,S,D]` buffer for
contiguity). There are also 3-output (`ltx2_ada_values3`, `…_indices3`) and a
norm+single-pair (`ltx2_rmsnorm_ada_scale_shift`) variant for sub-layers that only need
a subset.

## Residual gate

**Eager chain.** The post-sublayer update is `out = residual + (update + bias) · gate`
— a bias-add, a multiply by the (broadcast) gate, and a residual add: 3 elementwise
passes.

**Fused** (`ltx2_bias_residual_gate`). One `addcmul`-style kernel: load `update`,
`residual`, broadcast `gate` (supports `[B,D]` or `[B,1|T,D]`) and `bias`, compute
`residual + (update + bias)·gate` in fp32 and store. The gate's broadcast stride is
handled in-kernel so no `expand`/materialize is needed.

## FFN `proj_in` + GELU

**Eager chain.** The FFN up-projection is `addmm` (GEMM + bias) followed by a separate
tanh-GELU kernel that reads and writes the *large* `[B, S, D_ff]` intermediate.

**Fused** (`ltx2_bias_gelu_tanh_inplace`). The bias add and the tanh-GELU
(`x·σ(1.5958·(x + 0.044715·x³))`) are done **in place** on the GEMM output — the big
intermediate is read and written exactly once, and the bias kernel disappears. (A
no-bias `ltx2_gelu_tanh_inplace` exists for the variant where bias is already folded
into the GEMM epilogue.)

## Audio QKVG

**Eager chain.** Audio self-attention projects Q, K, V and a gate with four separate
linear layers → four GEMMs, each with launch + epilogue overhead and poor tensor-core
utilization at audio's small token counts.

**Fused.** Concatenate the four projection weights and run **one** GEMM, then slice
Q/K/V/G from the output. Fewer launches and one large, well-shaped matmul instead of
four small ones.

## VAE GroupNorm + SiLU

**Eager chain.** The VAE decoder's GroupNorm (a per-group reduction) is followed by a
separate SiLU activation over the full feature map — two passes over large spatial
tensors.

**Fused** (`group_norm_silu`). The group reduction and the `x·σ(x)` activation are one
kernel; for very large groups it tiles the reduction (`_CHUNK_SIZE`) so it stays within
SRAM. Used on the [tiled VAE decode](compile.md) path.

---

All of the above are **algorithm-lossless**: they reproduce the eager bf16 intermediate
rounding, so outputs match eager up to fused-op rounding only. Each is independently
toggled via its `SGLANG_HQ_KWL_FUSED_*` flag (see [overview](index.md)).
