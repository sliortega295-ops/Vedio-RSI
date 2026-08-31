#!/usr/bin/env python3
"""Exact-shape H100 gate for fusing SANA z scaling with output layout."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _timed(torch, fn, repeats: int = 12):
    for _ in range(4):
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
        import triton
        import triton.language as tl

        @triton.jit
        def fused_z_layout_kernel(
            output_ptr,
            hidden_ptr,
            z_ptr,
            total,
            num_tokens: tl.constexpr,
            num_heads: tl.constexpr,
            head_dim: tl.constexpr,
            inner_dim: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < total
            feature = offsets % inner_dim
            token = (offsets // inner_dim) % num_tokens
            batch = offsets // (inner_dim * num_tokens)
            head = feature // head_dim
            channel = feature % head_dim
            source = (
                ((batch * num_heads + head) * head_dim + channel) * num_tokens
                + token
            )
            z_offset = (batch * num_heads + head) * num_tokens + token
            hidden = tl.load(hidden_ptr + source, mask=mask, other=0.0)
            scale = tl.load(z_ptr + z_offset, mask=mask, other=0.0)
            tl.store(output_ptr + offsets, hidden * scale, mask=mask)

        def candidate(hidden, z, weight, bias):
            batch, num_heads, head_dim, num_tokens = hidden.shape
            inner_dim = num_heads * head_dim
            output = torch.empty(
                batch,
                num_tokens,
                inner_dim,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            total = output.numel()
            fused_z_layout_kernel[(triton.cdiv(total, 1024),)](
                output,
                hidden,
                z,
                total,
                num_tokens=num_tokens,
                num_heads=num_heads,
                head_dim=head_dim,
                inner_dim=inner_dim,
                BLOCK=1024,
                num_warps=8,
            )
            return output, F.linear(output, weight, bias)

        torch.manual_seed(18)
        device = torch.device("cuda:0")
        batch, heads, head_dim, tokens = 1, 20, 112, 32760
        inner_dim = heads * head_dim
        hidden = torch.randn(
            batch, heads, head_dim, tokens, device=device, dtype=torch.bfloat16
        )
        z = torch.randn(
            batch, heads, 1, tokens, device=device, dtype=torch.bfloat16
        )
        weight = torch.randn(
            inner_dim, inner_dim, device=device, dtype=torch.bfloat16
        ) / inner_dim**0.5
        bias = torch.randn(inner_dim, device=device, dtype=torch.bfloat16)

        def current_chain():
            scaled = (hidden * z).flatten(1, 2).transpose(1, 2)
            return scaled, F.linear(scaled, weight, bias)

        current, current_ms = _timed(torch, current_chain)
        fused, candidate_ms = _timed(
            torch, lambda: candidate(hidden, z, weight, bias)
        )
        current_contiguous = current[0].contiguous()
        torch.testing.assert_close(fused[0], current_contiguous, rtol=0, atol=0)
        torch.testing.assert_close(fused[1], current[1], rtol=0.02, atol=0.125)
        if fused[0].stride() != (tokens * inner_dim, inner_dim, 1):
            raise RuntimeError("candidate did not produce contiguous [B,N,H*C]")

        delta_ms = current_ms - candidate_ms
        predicted_s = delta_ms * 2000 / 1000
        passed = candidate_ms < current_ms * 0.98 and predicted_s >= 0.10
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
                "z_shape": list(z.shape),
                "z_stride": list(z.stride()),
                "output_projection_weight_shape": list(weight.shape),
                "dtype": str(hidden.dtype),
            },
            "correctness": {
                "scaled_exact": bool(torch.equal(fused[0], current_contiguous)),
                "projected_max_abs_diff": float(
                    (fused[1] - current[1]).abs().max().item()
                ),
                "projected_assert_close": {"rtol": 0.02, "atol": 0.125},
                "candidate_output_stride": list(fused[0].stride()),
                "expected_output_stride": [tokens * inner_dim, inner_dim, 1],
            },
            "timing": {
                "current_scale_layout_plus_projection_ms": current_ms,
                "candidate_fused_layout_plus_projection_ms": candidate_ms,
                "speedup": current_ms / candidate_ms,
                "delta_ms_per_attention": delta_ms,
                "formal_attention_invocations": 2000,
                "predicted_full_run_s_saved": predicted_s,
                "gate": "candidate < 98% of current and predicted saving >= 0.10 s",
            },
            "memory": {
                "current_scaled_tensor_elements": hidden.numel(),
                "candidate_scaled_tensor_elements": hidden.numel(),
                "extra_full_tensor": False,
                "candidate_layout": "contiguous [B,N,H*C]",
            },
            "semantics": (
                "The candidate preserves the native BF16 z multiplication exactly, "
                "materializes the same logical [B,N,H*C] tensor contiguously, and "
                "includes the unchanged output projection in the performance gate. "
                "No attention matmul, token, block, CFG call, or denoising step is skipped."
            ),
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True))
        return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
