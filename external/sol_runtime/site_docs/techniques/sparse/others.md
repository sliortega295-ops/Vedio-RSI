# Other sparse backends

Additional sparse / efficient attention backends are provided in
`runtime/layers/attention/backends/` for experimentation; not on a default line:

- **block-sparse** (`block_sparse_attn.py`) — static block-sparsity mask.
- **sparse-linear** (`sparse_linear_attn.py`) — linear-attention with sparsity.
- **RainFusion** (`rain_fusion_attn.py`) — training-free block-wise sparse attention.
- **LASER** (`laser_attn.py`).

Dense baselines for comparison: `flash_attn`, `flash_attn_2`, `sdpa`,
`sage_attn`, `sage_attn3`.
