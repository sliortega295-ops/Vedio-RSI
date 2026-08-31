# Model Zoo

Each pipeline loads weights from the HuggingFace Hub. Reuse a local cache if present,
otherwise download (network node only; runs can then go offline with `HF_HUB_OFFLINE=1`).

| Model | HF repo | GPUs |
|---|---|---|
| SANA-Video 2B (480p) | [`Efficient-Large-Model/SANA-Video_2B_480p_diffusers`](https://huggingface.co/Efficient-Large-Model/SANA-Video_2B_480p_diffusers) | 1 |
| Cosmos3-Super 64B | [`nvidia/Cosmos3-Super`](https://huggingface.co/nvidia/Cosmos3-Super) | 4 |
| LTX-2.3 | [`Lightricks/LTX-2.3`](https://huggingface.co/Lightricks/LTX-2.3) | 1 |

```bash
export HF_HOME="$PWD/.hf_cache"        # or your existing cache
huggingface-cli download Efficient-Large-Model/SANA-Video_2B_480p_diffusers
huggingface-cli download nvidia/Cosmos3-Super
huggingface-cli download Lightricks/LTX-2.3
```

Per-model download helpers (resumable, for headless / CPU-mover nodes) live under
`scripts/{sana,cosmos,ltx}/*download*`.
