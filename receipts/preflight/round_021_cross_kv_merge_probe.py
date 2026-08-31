#!/usr/bin/env python3
"""Exact-shape H100 gate for merging SANA cross-attention K/V projections."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _timed(torch, fn, repeats: int = 30):
    for _ in range(6):
        value = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        value = fn()
    end.record()
    end.synchronize()
    return value, start.elapsed_time(end) / repeats


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
        from sgl_kernel import rmsnorm

        torch.manual_seed(21)
        device = torch.device("cuda:0")
        batch, text_tokens, cross_dim, inner_dim = 1, 300, 2240, 2240
        encoder = torch.randn(
            batch, text_tokens, cross_dim, device=device, dtype=torch.bfloat16
        )
        key_weight = torch.randn(
            inner_dim, cross_dim, device=device, dtype=torch.bfloat16
        ) / cross_dim**0.5
        value_weight = torch.randn_like(key_weight)
        key_bias = torch.randn(inner_dim, device=device, dtype=torch.bfloat16)
        value_bias = torch.randn_like(key_bias)
        norm_weight = (
            1
            + 0.05
            * torch.randn(inner_dim, device=device, dtype=torch.bfloat16)
        )
        packed_weight = torch.cat((key_weight, value_weight), dim=0)
        packed_bias = torch.cat((key_bias, value_bias), dim=0)

        def current_chain():
            key = F.linear(encoder, key_weight, key_bias)
            value = F.linear(encoder, value_weight, value_bias)
            key = rmsnorm(key.reshape(-1, inner_dim), norm_weight, 1e-6).view(
                batch, text_tokens, inner_dim
            )
            return key, value

        def candidate_chain():
            packed = F.linear(encoder, packed_weight, packed_bias)
            key, value = packed.chunk(2, dim=-1)
            # Downstream RMSNorm/view require the same dense per-tensor layout.
            key = key.contiguous()
            value = value.contiguous()
            key = rmsnorm(key.reshape(-1, inner_dim), norm_weight, 1e-6).view(
                batch, text_tokens, inner_dim
            )
            return key, value

        current, current_ms = _timed(torch, current_chain)
        candidate, candidate_ms = _timed(torch, candidate_chain)
        torch.testing.assert_close(candidate[0], current[0], rtol=0.02, atol=0.125)
        torch.testing.assert_close(candidate[1], current[1], rtol=0.02, atol=0.125)
        if not all(tensor.is_contiguous() for tensor in candidate):
            raise RuntimeError("candidate failed to restore dense K/V layouts")

        key_max_abs = float((candidate[0] - current[0]).abs().max().item())
        value_max_abs = float((candidate[1] - current[1]).abs().max().item())
        delta_ms = current_ms - candidate_ms
        predicted_s = delta_ms * 2000 / 1000
        passed = candidate_ms < current_ms * 0.95 and predicted_s >= 0.10
        payload = {
            "schema_version": 1,
            "status": "passed" if passed else "rejected",
            "gpu": gpu,
            "lease_uuid": lease.gpu_uuid,
            "torch": torch.__version__,
            "formal_contract": {
                "encoder_shape": list(encoder.shape),
                "encoder_stride": list(encoder.stride()),
                "key_value_weight_shape": list(key_weight.shape),
                "packed_weight_shape": list(packed_weight.shape),
                "dtype": str(encoder.dtype),
                "text_tokens_source": "Gemma2EncoderConfig.text_len=300",
            },
            "correctness": {
                "key_max_abs_diff": key_max_abs,
                "value_max_abs_diff": value_max_abs,
                "assert_close": {"rtol": 0.02, "atol": 0.125},
                "dense_output_layouts": all(
                    tensor.is_contiguous() for tensor in candidate
                ),
            },
            "timing": {
                "current_two_projections_plus_key_norm_ms": current_ms,
                "candidate_packed_projection_copies_plus_key_norm_ms": candidate_ms,
                "speedup": current_ms / candidate_ms,
                "delta_ms_per_cross_attention": delta_ms,
                "formal_cross_attention_invocations": 2000,
                "predicted_full_run_s_saved": predicted_s,
                "gate": "candidate < 95% of current and predicted saving >= 0.10 s",
            },
            "memory": {
                "current_projection_outputs": 2,
                "candidate_packed_projection_outputs": 1,
                "candidate_dense_layout_copies": 2,
                "extra_packed_output_elements": packed.numel()
                if "packed" in locals()
                else batch * text_tokens * 2 * inner_dim,
            },
            "semantics": (
                "The candidate concatenates only the two K/V weights and biases "
                "that consume the same immutable encoder tensor, restores dense K/V "
                "layouts, and applies the same learned key RMSNorm. Query, SDPA, all "
                "tokens/blocks/calls, and every downstream operation remain unchanged."
            ),
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True))
        return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
