#!/usr/bin/env python3
"""Exact-shape H100 gate for SANA cross-attention output layout handoff."""

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
        def cross_output_layout_kernel(
            output_ptr,
            input_ptr,
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
                ((batch * num_heads + head) * num_tokens + token) * head_dim
                + channel
            )
            value = tl.load(input_ptr + source, mask=mask, other=0.0)
            tl.store(output_ptr + offsets, value, mask=mask)

        def candidate(sdpa_output, weight, bias):
            batch, heads, tokens, head_dim = sdpa_output.shape
            inner_dim = heads * head_dim
            layout = torch.empty(
                batch,
                tokens,
                inner_dim,
                device=sdpa_output.device,
                dtype=sdpa_output.dtype,
            )
            total = layout.numel()
            cross_output_layout_kernel[(triton.cdiv(total, 1024),)](
                layout,
                sdpa_output,
                total,
                num_tokens=tokens,
                num_heads=heads,
                head_dim=head_dim,
                inner_dim=inner_dim,
                BLOCK=1024,
                num_warps=8,
            )
            return layout, F.linear(layout, weight, bias)

        torch.manual_seed(22)
        device = torch.device("cuda:0")
        batch, heads, tokens, head_dim = 1, 20, 32760, 112
        inner_dim = heads * head_dim
        sdpa_output = torch.randn(
            batch, heads, tokens, head_dim, device=device, dtype=torch.bfloat16
        )
        weight = torch.randn(
            inner_dim, inner_dim, device=device, dtype=torch.bfloat16
        ) / inner_dim**0.5
        bias = torch.randn(inner_dim, device=device, dtype=torch.bfloat16)

        def current_chain():
            layout = sdpa_output.transpose(1, 2).reshape(batch, tokens, inner_dim)
            return layout, F.linear(layout, weight, bias)

        current, current_ms = _timed(torch, current_chain)
        fused, candidate_ms = _timed(
            torch, lambda: candidate(sdpa_output, weight, bias)
        )
        torch.testing.assert_close(fused[0], current[0], rtol=0, atol=0)
        torch.testing.assert_close(fused[1], current[1], rtol=0, atol=0)
        expected_stride = (tokens * inner_dim, inner_dim, 1)
        if fused[0].stride() != expected_stride:
            raise RuntimeError("candidate changed the dense projection-input layout")

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
                "sdpa_output_shape": list(sdpa_output.shape),
                "sdpa_output_stride": list(sdpa_output.stride()),
                "projection_weight_shape": list(weight.shape),
                "dtype": str(sdpa_output.dtype),
            },
            "correctness": {
                "layout_exact": bool(torch.equal(fused[0], current[0])),
                "projection_exact": bool(torch.equal(fused[1], current[1])),
                "candidate_layout_stride": list(fused[0].stride()),
                "expected_layout_stride": list(expected_stride),
            },
            "timing": {
                "current_native_layout_plus_projection_ms": current_ms,
                "candidate_triton_layout_plus_projection_ms": candidate_ms,
                "speedup": current_ms / candidate_ms,
                "delta_ms_per_cross_attention": delta_ms,
                "formal_cross_attention_invocations": 2000,
                "predicted_full_run_s_saved": predicted_s,
                "gate": "candidate < 98% of current and predicted saving >= 0.10 s",
            },
            "memory": {
                "current_dense_layout_elements": current[0].numel(),
                "candidate_dense_layout_elements": fused[0].numel(),
                "extra_full_tensor": False,
            },
            "semantics": (
                "The candidate performs only the required [B,H,S,D] to "
                "[B,S,H*D] dense layout materialization and includes the unchanged "
                "output projection in the gate. SDPA, projection weights/bias, "
                "tokens, blocks, CFG calls, and denoising steps remain unchanged."
            ),
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True))
        return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
