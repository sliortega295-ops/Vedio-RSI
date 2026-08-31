#!/usr/bin/env python3
"""Exact-shape H100 gate for fusing SANA Q/K ReLU with paired RoPE."""

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
    parser.add_argument("--runtime-python-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.guard_dir)
    from gpu_guard import locked_idle_lease

    with locked_idle_lease(args.lease_file) as (lease, gpu):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
        os.environ["SGLANG_SANA_MINIMAL_IMPORT"] = "1"
        sys.path.insert(0, args.runtime_python_root)

        import torch
        import triton
        import triton.language as tl

        from sglang.jit_kernel.diffusion.triton.sana_rope import (
            apply_sana_paired_rotary_emb,
        )

        @triton.jit
        def fused_relu_paired_rope_kernel(
            q_relu_out_ptr,
            k_relu_out_ptr,
            q_rotate_out_ptr,
            k_rotate_out_ptr,
            q_ptr,
            k_ptr,
            cos_ptr,
            sin_ptr,
            total_pairs,
            num_tokens: tl.constexpr,
            num_heads: tl.constexpr,
            head_dim: tl.constexpr,
            half_dim: tl.constexpr,
            frequency_batch: tl.constexpr,
            stride_cos_b: tl.constexpr,
            stride_cos_t: tl.constexpr,
            stride_cos_d: tl.constexpr,
            stride_sin_b: tl.constexpr,
            stride_sin_t: tl.constexpr,
            stride_sin_d: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < total_pairs
            pair = offsets % half_dim
            row = offsets // half_dim
            head = row % num_heads
            batch_token = row // num_heads
            token = batch_token % num_tokens
            batch = batch_token // num_tokens
            element = ((batch_token * num_heads + head) * head_dim) + 2 * pair
            frequency_batch_index = batch if frequency_batch > 1 else 0
            cos_offset = (
                frequency_batch_index * stride_cos_b
                + token * stride_cos_t
                + (2 * pair) * stride_cos_d
            )
            sin_offset = (
                frequency_batch_index * stride_sin_b
                + token * stride_sin_t
                + (2 * pair + 1) * stride_sin_d
            )

            cosine = tl.load(cos_ptr + cos_offset, mask=mask, other=0.0).to(tl.float32)
            sine = tl.load(sin_ptr + sin_offset, mask=mask, other=0.0).to(tl.float32)
            q_even = tl.maximum(
                tl.load(q_ptr + element, mask=mask, other=0.0).to(tl.float32), 0.0
            ).to(tl.bfloat16)
            q_odd = tl.maximum(
                tl.load(q_ptr + element + 1, mask=mask, other=0.0).to(tl.float32),
                0.0,
            ).to(tl.bfloat16)
            k_even = tl.maximum(
                tl.load(k_ptr + element, mask=mask, other=0.0).to(tl.float32), 0.0
            ).to(tl.bfloat16)
            k_odd = tl.maximum(
                tl.load(k_ptr + element + 1, mask=mask, other=0.0).to(tl.float32),
                0.0,
            ).to(tl.bfloat16)

            tl.store(q_relu_out_ptr + element, q_even, mask=mask)
            tl.store(q_relu_out_ptr + element + 1, q_odd, mask=mask)
            tl.store(k_relu_out_ptr + element, k_even, mask=mask)
            tl.store(k_relu_out_ptr + element + 1, k_odd, mask=mask)
            q_even_fp32 = q_even.to(tl.float32)
            q_odd_fp32 = q_odd.to(tl.float32)
            k_even_fp32 = k_even.to(tl.float32)
            k_odd_fp32 = k_odd.to(tl.float32)
            tl.store(
                q_rotate_out_ptr + element,
                q_even_fp32 * cosine - q_odd_fp32 * sine,
                mask=mask,
            )
            tl.store(
                q_rotate_out_ptr + element + 1,
                q_even_fp32 * sine + q_odd_fp32 * cosine,
                mask=mask,
            )
            tl.store(
                k_rotate_out_ptr + element,
                k_even_fp32 * cosine - k_odd_fp32 * sine,
                mask=mask,
            )
            tl.store(
                k_rotate_out_ptr + element + 1,
                k_even_fp32 * sine + k_odd_fp32 * cosine,
                mask=mask,
            )

        def candidate(q, k, cos, sin):
            batch, num_tokens, num_heads, head_dim = q.shape
            total_pairs = q.numel() // 2
            outputs = [torch.empty_like(q) for _ in range(4)]
            fused_relu_paired_rope_kernel[(triton.cdiv(total_pairs, 1024),)](
                *outputs,
                q,
                k,
                cos,
                sin,
                total_pairs,
                num_tokens=num_tokens,
                num_heads=num_heads,
                head_dim=head_dim,
                half_dim=head_dim // 2,
                frequency_batch=cos.shape[0],
                stride_cos_b=cos.stride(0),
                stride_cos_t=cos.stride(1),
                stride_cos_d=cos.stride(3),
                stride_sin_b=sin.stride(0),
                stride_sin_t=sin.stride(1),
                stride_sin_d=sin.stride(3),
                BLOCK=1024,
                num_warps=8,
            )
            return tuple(outputs)

        torch.manual_seed(15)
        device = torch.device("cuda:0")
        batch, tokens, heads, head_dim = 1, 32760, 20, 112
        q = torch.randn(
            batch, tokens, heads, head_dim, device=device, dtype=torch.bfloat16
        )
        k = torch.randn_like(q)
        cos = torch.randn(
            1, tokens, 1, head_dim, device=device, dtype=torch.float32
        )
        sin = torch.randn_like(cos)
        q_before = q.clone()
        k_before = k.clone()

        def current_chain():
            q_relu = torch.relu(q)
            k_relu = torch.relu(k)
            q_rotate, k_rotate = apply_sana_paired_rotary_emb(
                q_relu, k_relu, cos, sin
            )
            return q_relu, k_relu, q_rotate, k_rotate

        current, current_ms = _timed(torch, current_chain)
        fused, candidate_ms = _timed(torch, lambda: candidate(q, k, cos, sin))
        torch.testing.assert_close(fused[0], current[0], rtol=0, atol=0)
        torch.testing.assert_close(fused[1], current[1], rtol=0, atol=0)
        torch.testing.assert_close(fused[2], current[2], rtol=0.02, atol=0.03125)
        torch.testing.assert_close(fused[3], current[3], rtol=0.02, atol=0.03125)
        if not torch.equal(q, q_before) or not torch.equal(k, k_before):
            raise RuntimeError("candidate mutated normalized Q/K inputs")
        if any(output.stride() != q.stride() for output in fused):
            raise RuntimeError("candidate changed an output layout")

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
                "qk_shape": list(q.shape),
                "qk_stride": list(q.stride()),
                "qk_dtype": str(q.dtype),
                "frequency_shape": list(cos.shape),
                "frequency_dtype": str(cos.dtype),
            },
            "correctness": {
                "q_relu_exact": bool(torch.equal(fused[0], current[0])),
                "k_relu_exact": bool(torch.equal(fused[1], current[1])),
                "q_rotate_max_abs_diff": float((fused[2] - current[2]).abs().max().item()),
                "k_rotate_max_abs_diff": float((fused[3] - current[3]).abs().max().item()),
                "rotate_assert_close": {"rtol": 0.02, "atol": 0.03125},
                "inputs_preserved": bool(
                    torch.equal(q, q_before) and torch.equal(k, k_before)
                ),
                "output_strides_preserved": True,
            },
            "timing": {
                "current_relu_plus_paired_rope_ms": current_ms,
                "candidate_fused_ms": candidate_ms,
                "speedup": current_ms / candidate_ms,
                "delta_ms_per_attention": delta_ms,
                "formal_attention_invocations": 2000,
                "predicted_full_run_s_saved": predicted_s,
                "gate": "candidate < 95% of current and predicted saving >= 0.10 s",
            },
            "memory": {
                "current_full_outputs": 4,
                "candidate_full_outputs": 4,
                "current_full_tensor_reads": 4,
                "candidate_full_tensor_reads": 2,
                "extra_full_tensor": False,
                "inputs_mutated": False,
            },
            "semantics": (
                "The candidate writes the same BF16 ReLU Q/K required by z and "
                "the same paired rotary Q/K required by the attention matmuls. "
                "No tensor, block, CFG call, or denoising step is skipped."
            ),
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True))
        return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
