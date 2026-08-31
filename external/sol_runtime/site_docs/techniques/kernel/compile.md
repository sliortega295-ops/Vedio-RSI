# Compile & VAE

- **Attention gate-to-out compile** — Inductor-compile the fixed-shape
  `sigmoid·scale → reshape → to_out` subgraph (video self-attention).
- **Tiled VAE decoder compile** — Inductor-compile the fixed 1080p tiled decode.

## torch.compile mode (caveat)

`SGLANG_TORCH_COMPILE_MODE` selects the inductor mode. The full-model
`max-autotune-no-cudagraphs` path **deadlocks on a cold cache** (a grouped-conv
Triton autotune hangs at `cuda.synchronize`). Cold-safe = `default`; for the faster
max-autotune path use `TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC=1` + a persistent
`TORCHINDUCTOR_CACHE_DIR` (the first cold run warms it).
