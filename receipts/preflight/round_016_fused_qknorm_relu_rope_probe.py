#!/usr/bin/env python3
"""Exact-shape H100 gate for packed Q/K RMSNorm + ReLU + paired RoPE."""

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
        from sgl_kernel import rmsnorm

        from sglang.jit_kernel.diffusion.triton.sana_rope import (
            apply_sana_paired_rotary_emb,
        )

        @triton.jit
        def fused_qknorm_relu_rope_kernel(
            q_relu_out_ptr,
            k_relu_out_ptr,
            q_rotate_out_ptr,
            k_rotate_out_ptr,
            q_ptr,
            k_ptr,
            q_weight_ptr,
            k_weight_ptr,
            cos_ptr,
            sin_ptr,
            num_rows,
            inner_dim: tl.constexpr,
            num_tokens: tl.constexpr,
            head_dim: tl.constexpr,
            stride_q_b: tl.constexpr,
            stride_q_t: tl.constexpr,
            stride_k_b: tl.constexpr,
            stride_k_t: tl.constexpr,
            frequency_batch: tl.constexpr,
            stride_cos_b: tl.constexpr,
            stride_cos_t: tl.constexpr,
            stride_cos_d: tl.constexpr,
            stride_sin_b: tl.constexpr,
            stride_sin_t: tl.constexpr,
            stride_sin_d: tl.constexpr,
            eps: tl.constexpr,
            BLOCK_PAIRS: tl.constexpr,
        ):
            row = tl.program_id(0)
            pairs = tl.arange(0, BLOCK_PAIRS)
            even = 2 * pairs
            odd = even + 1
            mask = (row < num_rows) & (odd < inner_dim)
            token = row % num_tokens
            batch = row // num_tokens
            q_base = batch * stride_q_b + token * stride_q_t
            k_base = batch * stride_k_b + token * stride_k_t

            q_even = tl.load(q_ptr + q_base + even, mask=mask, other=0.0).to(tl.float32)
            q_odd = tl.load(q_ptr + q_base + odd, mask=mask, other=0.0).to(tl.float32)
            k_even = tl.load(k_ptr + k_base + even, mask=mask, other=0.0).to(tl.float32)
            k_odd = tl.load(k_ptr + k_base + odd, mask=mask, other=0.0).to(tl.float32)
            q_variance = tl.sum(q_even * q_even + q_odd * q_odd, axis=0) / inner_dim
            k_variance = tl.sum(k_even * k_even + k_odd * k_odd, axis=0) / inner_dim
            q_rstd = tl.rsqrt(q_variance + eps)
            k_rstd = tl.rsqrt(k_variance + eps)
            q_weight_even = tl.load(q_weight_ptr + even, mask=mask, other=0.0).to(tl.float32)
            q_weight_odd = tl.load(q_weight_ptr + odd, mask=mask, other=0.0).to(tl.float32)
            k_weight_even = tl.load(k_weight_ptr + even, mask=mask, other=0.0).to(tl.float32)
            k_weight_odd = tl.load(k_weight_ptr + odd, mask=mask, other=0.0).to(tl.float32)

            q_relu_even = tl.maximum(q_even * q_rstd * q_weight_even, 0.0).to(tl.bfloat16)
            q_relu_odd = tl.maximum(q_odd * q_rstd * q_weight_odd, 0.0).to(tl.bfloat16)
            k_relu_even = tl.maximum(k_even * k_rstd * k_weight_even, 0.0).to(tl.bfloat16)
            k_relu_odd = tl.maximum(k_odd * k_rstd * k_weight_odd, 0.0).to(tl.bfloat16)

            out_base = row * inner_dim
            tl.store(q_relu_out_ptr + out_base + even, q_relu_even, mask=mask)
            tl.store(q_relu_out_ptr + out_base + odd, q_relu_odd, mask=mask)
            tl.store(k_relu_out_ptr + out_base + even, k_relu_even, mask=mask)
            tl.store(k_relu_out_ptr + out_base + odd, k_relu_odd, mask=mask)

            frequency_batch_index = batch if frequency_batch > 1 else 0
            frequency_dim = even % head_dim
            cos_offset = (
                frequency_batch_index * stride_cos_b
                + token * stride_cos_t
                + frequency_dim * stride_cos_d
            )
            sin_offset = (
                frequency_batch_index * stride_sin_b
                + token * stride_sin_t
                + (frequency_dim + 1) * stride_sin_d
            )
            cosine = tl.load(cos_ptr + cos_offset, mask=mask, other=0.0).to(tl.float32)
            sine = tl.load(sin_ptr + sin_offset, mask=mask, other=0.0).to(tl.float32)
            q_relu_even_fp32 = q_relu_even.to(tl.float32)
            q_relu_odd_fp32 = q_relu_odd.to(tl.float32)
            k_relu_even_fp32 = k_relu_even.to(tl.float32)
            k_relu_odd_fp32 = k_relu_odd.to(tl.float32)
            tl.store(
                q_rotate_out_ptr + out_base + even,
                q_relu_even_fp32 * cosine - q_relu_odd_fp32 * sine,
                mask=mask,
            )
            tl.store(
                q_rotate_out_ptr + out_base + odd,
                q_relu_even_fp32 * sine + q_relu_odd_fp32 * cosine,
                mask=mask,
            )
            tl.store(
                k_rotate_out_ptr + out_base + even,
                k_relu_even_fp32 * cosine - k_relu_odd_fp32 * sine,
                mask=mask,
            )
            tl.store(
                k_rotate_out_ptr + out_base + odd,
                k_relu_even_fp32 * sine + k_relu_odd_fp32 * cosine,
                mask=mask,
            )

        def candidate(q, k, q_weight, k_weight, cos, sin):
            batch, num_tokens, inner_dim = q.shape
            head_dim = cos.shape[-1]
            heads = inner_dim // head_dim
            outputs = [
                torch.empty(
                    batch,
                    num_tokens,
                    heads,
                    head_dim,
                    device=q.device,
                    dtype=q.dtype,
                )
                for _ in range(4)
            ]
            fused_qknorm_relu_rope_kernel[(batch * num_tokens,)](
                *outputs,
                q,
                k,
                q_weight,
                k_weight,
                cos,
                sin,
                batch * num_tokens,
                inner_dim=inner_dim,
                num_tokens=num_tokens,
                head_dim=head_dim,
                stride_q_b=q.stride(0),
                stride_q_t=q.stride(1),
                stride_k_b=k.stride(0),
                stride_k_t=k.stride(1),
                frequency_batch=cos.shape[0],
                stride_cos_b=cos.stride(0),
                stride_cos_t=cos.stride(1),
                stride_cos_d=cos.stride(3),
                stride_sin_b=sin.stride(0),
                stride_sin_t=sin.stride(1),
                stride_sin_d=sin.stride(3),
                eps=1e-6,
                BLOCK_PAIRS=2048,
                num_warps=8,
            )
            return tuple(outputs)

        torch.manual_seed(16)
        device = torch.device("cuda:0")
        batch, tokens, heads, head_dim = 1, 32760, 20, 112
        inner_dim = heads * head_dim
        packed = torch.randn(
            batch,
            tokens,
            3 * inner_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        q, k, _ = packed.chunk(3, dim=-1)
        q_weight = (
            1 + 0.05 * torch.randn(inner_dim, device=device, dtype=torch.bfloat16)
        )
        k_weight = (
            1 + 0.05 * torch.randn(inner_dim, device=device, dtype=torch.bfloat16)
        )
        cos = torch.randn(
            1, tokens, 1, head_dim, device=device, dtype=torch.float32
        )
        sin = torch.randn_like(cos)
        packed_before = packed.clone()

        def current_chain():
            q_norm = rmsnorm(q.reshape(-1, inner_dim), q_weight, 1e-6).view(q.shape)
            k_norm = rmsnorm(k.reshape(-1, inner_dim), k_weight, 1e-6).view(k.shape)
            q_relu = torch.relu(q_norm).unflatten(2, (heads, head_dim))
            k_relu = torch.relu(k_norm).unflatten(2, (heads, head_dim))
            q_rotate, k_rotate = apply_sana_paired_rotary_emb(
                q_relu, k_relu, cos, sin
            )
            return q_relu, k_relu, q_rotate, k_rotate

        current, current_ms = _timed(torch, current_chain)
        fused, candidate_ms = _timed(
            torch,
            lambda: candidate(q, k, q_weight, k_weight, cos, sin),
        )
        for index in range(4):
            torch.testing.assert_close(fused[index], current[index], rtol=0.02, atol=0.125)
        if not torch.equal(packed, packed_before):
            raise RuntimeError("candidate mutated packed QKV input")
        expected_stride = (tokens * inner_dim, inner_dim, head_dim, 1)
        if any(output.stride() != expected_stride for output in fused):
            raise RuntimeError("candidate changed an output layout")

        max_abs = [
            float((fused[index] - current[index]).abs().max().item())
            for index in range(4)
        ]
        delta_ms = current_ms - candidate_ms
        predicted_s = delta_ms * 2000 / 1000
        passed = candidate_ms < current_ms * 0.95 and predicted_s >= 0.10
        q_reshape = q.reshape(-1, inner_dim)
        k_reshape = k.reshape(-1, inner_dim)
        payload = {
            "schema_version": 1,
            "status": "passed" if passed else "rejected",
            "gpu": gpu,
            "lease_uuid": lease.gpu_uuid,
            "torch": torch.__version__,
            "triton": triton.__version__,
            "formal_contract": {
                "packed_shape": list(packed.shape),
                "packed_stride": list(packed.stride()),
                "q_shape": list(q.shape),
                "q_stride": list(q.stride()),
                "q_reshape_stride": list(q_reshape.stride()),
                "q_reshape_aliases_packed_storage": q_reshape.untyped_storage().data_ptr()
                == packed.untyped_storage().data_ptr(),
                "k_reshape_aliases_packed_storage": k_reshape.untyped_storage().data_ptr()
                == packed.untyped_storage().data_ptr(),
                "weight_dtype": str(q_weight.dtype),
                "eps": 1e-6,
            },
            "correctness": {
                "q_relu_max_abs_diff": max_abs[0],
                "k_relu_max_abs_diff": max_abs[1],
                "q_rotate_max_abs_diff": max_abs[2],
                "k_rotate_max_abs_diff": max_abs[3],
                "assert_close": {"rtol": 0.02, "atol": 0.125},
                "packed_input_preserved": bool(torch.equal(packed, packed_before)),
                "output_strides_preserved": True,
            },
            "timing": {
                "current_full_chain_ms": current_ms,
                "candidate_fused_ms": candidate_ms,
                "speedup": current_ms / candidate_ms,
                "delta_ms_per_attention": delta_ms,
                "formal_attention_invocations": 2000,
                "predicted_full_run_s_saved": predicted_s,
                "gate": "candidate < 95% of current and predicted saving >= 0.10 s",
            },
            "semantics": (
                "The candidate computes the two learned RMSNorms in FP32, rounds "
                "to the BF16 model contract before ReLU/RoPE, and emits only the "
                "nonrotated and rotated Q/K tensors consumed downstream. Value, z, "
                "matmuls, blocks, CFG calls, and denoising steps are unchanged."
            ),
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True))
        return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
