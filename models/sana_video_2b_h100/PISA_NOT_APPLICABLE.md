# PISA disposition: NOT_APPLICABLE

Status: `NOT_APPLICABLE` for this SANA-Video 2B reproduction. No Attention/PISA
executor is launched.

Source evidence pinned by this experiment:

- `external/sol_runtime/python/sglang/multimodal_gen/configs/models/dits/sana_video.py`
  defines `attention_head_dim = 112` and `num_attention_heads = 20`.
- `external/sol_runtime/python/sglang/multimodal_gen/runtime/models/dits/sana_video.py`
  implements the video-token self-attention as `SanaVideoLinearAttention`. Its
  forward pass applies a ReLU feature map and evaluates linear-attention KV
  aggregation; it is not dense softmax attention.
- `external/sol_runtime/python/sglang/multimodal_gen/runtime/layers/attention/backends/piecewise_attn.py`
  declares PISA/PiecewiseAttention head sizes `[32, 64, 96, 128, 160, 192, 224,
  256]`; the target head dimension 112 is absent.
- The archived official contract
  `workflow/attention_pa/nodes/codex_executor/attention_scope.md` defines PISA
  as exact selected-block softmax attention plus an approximate remainder, and
  its representative video softmax shape uses head dimension 256.

Adapting PISA here would therefore require replacing the target attention
algorithm and extending an unsupported kernel shape. That is a new research
project, not faithful reproduction of the archived PISA workflow on this model.
