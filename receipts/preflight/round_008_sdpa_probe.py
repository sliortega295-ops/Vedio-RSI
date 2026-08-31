#!/usr/bin/env python3
"""Probe exact SANA cross-attention SDPA backend selection on the leased H100."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lease-file", required=True)
    parser.add_argument("--guard-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.guard_dir)
    from gpu_guard import locked_idle_lease

    with locked_idle_lease(args.lease_file) as (lease, gpu):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
        import torch
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel

        torch.manual_seed(8)
        device = torch.device("cuda:0")
        batch, heads, query_tokens, head_dim = 1, 20, 32760, 112
        branch_results = {}

        def profile_events(fn):
            import torch.profiler as tp

            with tp.profile(
                activities=[tp.ProfilerActivity.CPU, tp.ProfilerActivity.CUDA]
            ) as prof:
                prof_out = fn()
                torch.cuda.synchronize()
            del prof_out

            def device_time(event):
                return float(
                    getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0.0))
                )

            return [
                {
                    "key": event.key,
                    "device_time_total_us": device_time(event),
                    "cpu_time_total_us": float(event.cpu_time_total),
                    "count": event.count,
                }
                for event in sorted(
                    prof.key_averages(), key=device_time, reverse=True
                )[:20]
            ]

        for branch, text_tokens in (("positive", 110), ("negative", 56)):
            # Recreate the real post-linear layout: contiguous B,S,H,D transposed
            # to B,H,S,D. The exact tokenizer audit showed both branch masks are
            # all ones, so the baseline additive mask is identically zero.
            query = torch.randn(
                batch, query_tokens, heads, head_dim, device=device, dtype=torch.bfloat16
            ).transpose(1, 2)
            key = torch.randn(
                batch, text_tokens, heads, head_dim, device=device, dtype=torch.bfloat16
            ).transpose(1, 2)
            value = torch.randn(
                batch, text_tokens, heads, head_dim, device=device, dtype=torch.bfloat16
            ).transpose(1, 2)
            base_mask = torch.zeros(
                batch, 1, text_tokens, device=device, dtype=torch.bfloat16
            )
            mask = base_mask.view(batch, 1, 1, text_tokens).expand(
                batch, heads, query_tokens, text_tokens
            )

            def call(backend, use_zero_mask):
                kwargs = {"attn_mask": mask if use_zero_mask else None}
                if backend is None:
                    return F.scaled_dot_product_attention(query, key, value, **kwargs)
                with sdpa_kernel(backends=[backend]):
                    return F.scaled_dot_product_attention(query, key, value, **kwargs)

            results = {}
            reference = None
            candidates = [
                ("current_auto_zero_mask", None, True),
                ("auto_no_mask", None, False),
                ("flash_no_mask", SDPBackend.FLASH_ATTENTION, False),
                ("efficient_no_mask", SDPBackend.EFFICIENT_ATTENTION, False),
                ("cudnn_no_mask", SDPBackend.CUDNN_ATTENTION, False),
            ]
            for name, backend, use_zero_mask in candidates:
                try:
                    for _ in range(2):
                        output = call(backend, use_zero_mask)
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(5):
                        output = call(backend, use_zero_mask)
                    end.record()
                    end.synchronize()
                    elapsed_ms = start.elapsed_time(end) / 5.0
                    if reference is None:
                        reference = output.detach()
                        max_abs = 0.0
                        exact = True
                    else:
                        max_abs = float((output - reference).abs().max().item())
                        exact = bool(torch.equal(output, reference))
                    results[name] = {
                        "status": "passed",
                        "mean_cuda_event_ms": elapsed_ms,
                        "exact_equal_to_current": exact,
                        "max_abs_diff_to_current": max_abs,
                    }
                    del output
                except Exception as exc:
                    results[name] = {
                        "status": "rejected",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }

            current_fn = lambda: call(None, True)
            no_mask_fn = lambda: call(None, False)
            branch_results[branch] = {
                "text_tokens": text_tokens,
                "mask_all_valid": True,
                "results": results,
                "current_auto_profile_events": profile_events(current_fn),
                "no_mask_auto_profile_events": profile_events(no_mask_fn),
            }
            del reference, query, key, value, mask, base_mask
            torch.cuda.synchronize()

        payload = {
            "schema_version": 1,
            "status": "completed",
            "gpu": gpu,
            "lease_uuid": lease.gpu_uuid,
            "torch_version": torch.__version__,
            "flash_attention_available": torch.backends.cuda.is_flash_attention_available(),
            "shared_shape": {
                "batch": batch,
                "heads": heads,
                "query_tokens": query_tokens,
                "head_dim": head_dim,
                "dtype": "torch.bfloat16",
                "mask_is_expanded_view": True,
            },
            "branches": branch_results,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
