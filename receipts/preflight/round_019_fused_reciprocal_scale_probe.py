#!/usr/bin/env python3
"""Exact-shape H100 gate for fusing SANA reciprocal z with output scaling."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _timed(torch, fn, repeats: int = 20):
    for _ in range(5):
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
        import triton
        import triton.language as tl

        @triton.jit
        def fused_reciprocal_scale_kernel(
            output_ptr,
            hidden_ptr,
            denominator_ptr,
            total,
            num_tokens: tl.constexpr,
            num_heads: tl.constexpr,
            head_dim: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < total
            token = offsets % num_tokens
            channel_row = offsets // num_tokens
            head = channel_row % (num_heads * head_dim) // head_dim
            batch = channel_row // (num_heads * head_dim)
            denominator_offset = (batch * num_heads + head) * num_tokens + token
            hidden = tl.load(hidden_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            denominator = tl.load(
                denominator_ptr + denominator_offset, mask=mask, other=1.0
            ).to(tl.float32)
            # Preserve the native BF16 z materialization boundary before scaling.
            z_bf16 = (1.0 / (denominator + 1.0e-15)).to(tl.bfloat16)
            output = (hidden * z_bf16.to(tl.float32)).to(tl.bfloat16)
            tl.store(output_ptr + offsets, output, mask=mask)

        def candidate(hidden, denominator):
            batch, num_heads, head_dim, num_tokens = hidden.shape
            output = torch.empty_like(hidden)
            total = hidden.numel()
            fused_reciprocal_scale_kernel[(triton.cdiv(total, 1024),)](
                output,
                hidden,
                denominator,
                total,
                num_tokens=num_tokens,
                num_heads=num_heads,
                head_dim=head_dim,
                BLOCK=1024,
                num_warps=8,
            )
            return output

        torch.manual_seed(19)
        device = torch.device("cuda:0")
        batch, heads, head_dim, tokens = 1, 20, 112, 32760
        hidden = torch.randn(
            batch, heads, head_dim, tokens, device=device, dtype=torch.bfloat16
        )
        denominator = (
            torch.rand(
                batch, heads, 1, tokens, device=device, dtype=torch.bfloat16
            )
            + 0.25
        )

        def current_chain():
            z = 1 / (denominator + 1e-15)
            return hidden * z

        current, current_ms = _timed(torch, current_chain)
        fused, candidate_ms = _timed(torch, lambda: candidate(hidden, denominator))
        torch.testing.assert_close(fused, current, rtol=0, atol=0)

        delta_ms = current_ms - candidate_ms
        predicted_s = delta_ms * 2000 / 1000
        passed = candidate_ms < current_ms * 0.95 and predicted_s >= 0.10
        payload = {
            "schema_version": 1,
            "status": "passed" if passed else "rejected",
            "gpu": gpu,
            "lease_uuid": lease.gpu_uuid,
            "torch": torch.__version__,
            "triton": triton.__version__,
            "formal_contract": {
                "hidden_shape": list(hidden.shape),
                "hidden_stride": list(hidden.stride()),
                "denominator_shape": list(denominator.shape),
                "denominator_stride": list(denominator.stride()),
                "dtype": str(hidden.dtype),
            },
            "correctness": {
                "scaled_exact": bool(torch.equal(fused, current)),
                "max_abs_diff": float((fused - current).abs().max().item()),
                "output_stride_preserved": fused.stride() == hidden.stride(),
            },
            "timing": {
                "current_reciprocal_plus_scale_ms": current_ms,
                "candidate_fused_ms": candidate_ms,
                "speedup": current_ms / candidate_ms,
                "delta_ms_per_attention": delta_ms,
                "formal_attention_invocations": 2000,
                "predicted_full_run_s_saved": predicted_s,
                "gate": "candidate < 95% of current and predicted saving >= 0.10 s",
            },
            "memory": {
                "current_materialized_z_elements": denominator.numel(),
                "candidate_materialized_z_elements": 0,
                "scaled_output_elements": hidden.numel(),
                "extra_full_tensor": False,
            },
            "semantics": (
                "The candidate retains the native BF16 reciprocal boundary and "
                "BF16 scaling exactly while removing only the materialized z tensor "
                "and one primitive boundary. No attention matmul, output, token, "
                "block, CFG call, or denoising step is skipped."
            ),
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True))
        return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
