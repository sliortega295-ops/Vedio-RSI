# SANA-Video 2B H100 reproduction target

This model contract materializes the pinned SGLang SANA runtime into every
experiment worktree, while keeping the model snapshot, cu128 environment,
dependency overlay, and compiled SM90 kernel staging read-only.

The dense control is fixed at 832x480, 81 frames, 16 fps, 50 denoising steps,
guidance 6, seed 42, and the model-card long prompt with `motion score: 30.`.
The VAE runs in fp32; the transformer and Gemma text encoder run in bf16. The
baseline wrapper rejects every exposed compile, QKV merge, linear-attention
bf16, and EasyCache switch when its immutable config ID is selected.

Every GPU command reads the canonical persistent lease, blocks on its flock,
and then fails closed if `nvidia-smi` reports a compute application on the
leased UUID. Foreign processes are never stopped. `lease_gpu.py` creates the
lease only after the selected UUID is idle; `probe_h100.py` is the minimal
import plus BF16 RMSNorm gate; `gpu_infer.py` runs and receipts a full video.

Candidate configs may enable the exposed switches or edit the experiment-local
`external/sol_runtime`. They must retain the same workload and must launch only
through this wrapper so GPU serialization and complete receipts remain intact.

PISA is deliberately excluded. See `PISA_NOT_APPLICABLE.md` for the source
contract showing that the target's main video-token self-attention is linear
attention with head dimension 112, while the archived PISA backend implements
piecewise sparse softmax attention and does not support head dimension 112.
