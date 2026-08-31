#!/usr/bin/env python3
"""Exact-shape H100 gate for paired out-of-place SANA Q/K RoPE."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def timed(torch, fn, repeats=20):
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
        def paired_rope_kernel(
            q_out_ptr,
            k_out_ptr,
            q_ptr,
            k_ptr,
            cos_ptr,
            sin_ptr,
            total_pairs,
            num_tokens: tl.constexpr,
            num_heads: tl.constexpr,
            head_dim: tl.constexpr,
            half_dim: tl.constexpr,
            freq_batch: tl.constexpr,
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
            bt = row // num_heads
            token = bt % num_tokens
            batch = bt // num_tokens
            element = ((bt * num_heads + head) * head_dim) + 2 * pair
            freq_b = batch if freq_batch > 1 else 0
            cos_offset = (
                freq_b * stride_cos_b
                + token * stride_cos_t
                + (2 * pair) * stride_cos_d
            )
            sin_offset = (
                freq_b * stride_sin_b
                + token * stride_sin_t
                + (2 * pair + 1) * stride_sin_d
            )
            cosine = tl.load(cos_ptr + cos_offset, mask=mask, other=0.0).to(tl.float32)
            sine = tl.load(sin_ptr + sin_offset, mask=mask, other=0.0).to(tl.float32)
            q1 = tl.load(q_ptr + element, mask=mask, other=0.0).to(tl.float32)
            q2 = tl.load(q_ptr + element + 1, mask=mask, other=0.0).to(tl.float32)
            k1 = tl.load(k_ptr + element, mask=mask, other=0.0).to(tl.float32)
            k2 = tl.load(k_ptr + element + 1, mask=mask, other=0.0).to(tl.float32)
            tl.store(q_out_ptr + element, q1 * cosine - q2 * sine, mask=mask)
            tl.store(q_out_ptr + element + 1, q1 * sine + q2 * cosine, mask=mask)
            tl.store(k_out_ptr + element, k1 * cosine - k2 * sine, mask=mask)
            tl.store(k_out_ptr + element + 1, k1 * sine + k2 * cosine, mask=mask)

        def native(x, cos, sin):
            x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
            cosine = cos[..., 0::2]
            sine = sin[..., 1::2]
            out = torch.empty_like(x)
            out[..., 0::2] = x1 * cosine - x2 * sine
            out[..., 1::2] = x1 * sine + x2 * cosine
            return out.type_as(x)

        def candidate(q, k, cos, sin):
            batch, num_tokens, num_heads, head_dim = q.shape
            half_dim = head_dim // 2
            total_pairs = q.numel() // 2
            q_out = torch.empty_like(q)
            k_out = torch.empty_like(k)
            paired_rope_kernel[(triton.cdiv(total_pairs, 1024),)](
                q_out,
                k_out,
                q,
                k,
                cos,
                sin,
                total_pairs,
                num_tokens=num_tokens,
                num_heads=num_heads,
                head_dim=head_dim,
                half_dim=half_dim,
                freq_batch=cos.shape[0],
                stride_cos_b=cos.stride(0),
                stride_cos_t=cos.stride(1),
                stride_cos_d=cos.stride(3),
                stride_sin_b=sin.stride(0),
                stride_sin_t=sin.stride(1),
                stride_sin_d=sin.stride(3),
                BLOCK=1024,
                num_warps=8,
            )
            return q_out, k_out

        torch.manual_seed(13)
        device = torch.device("cuda:0")
        batch, tokens, heads, head_dim = 1, 32760, 20, 112
        dtype = torch.bfloat16
        q = torch.relu(torch.randn(batch, tokens, heads, head_dim, device=device, dtype=dtype))
        k = torch.relu(torch.randn_like(q))
        # Match SanaVideoRoPE3D: [1, N, 1, D], contiguous, FP32.
        cos = torch.randn(1, tokens, 1, head_dim, device=device, dtype=torch.float32)
        sin = torch.randn_like(cos)
        q_before = q.clone()
        k_before = k.clone()

        def run_native():
            return native(q, cos, sin), native(k, cos, sin)

        def run_candidate():
            return candidate(q, k, cos, sin)

        (q_native, k_native), native_ms = timed(torch, run_native)
        (q_candidate, k_candidate), candidate_ms = timed(torch, run_candidate)
        # Fusion changes the primitive boundary and may use an FMA, so the
        # mathematical identity is gated at the established BF16 contract
        # rather than requiring bit identity.  The first exact-only probe saw
        # 371 / 73,382,400 Q elements differ with max abs 0.015625.
        torch.testing.assert_close(q_candidate, q_native, rtol=0.02, atol=0.03125)
        torch.testing.assert_close(k_candidate, k_native, rtol=0.02, atol=0.03125)
        if not torch.equal(q, q_before) or not torch.equal(k, k_before):
            raise RuntimeError("candidate mutated non-rotated Q/K inputs needed by z")
        if q_candidate.stride() != q.stride() or k_candidate.stride() != k.stride():
            raise RuntimeError("candidate changed Q/K output layout")

        delta_ms = native_ms - candidate_ms
        predicted_s = delta_ms * 2000 / 1000
        passed = candidate_ms < native_ms * 0.90 and predicted_s >= 0.10
        payload = {
            "schema_version": 1,
            "status": "passed" if passed else "rejected",
            "gpu": gpu,
            "lease_uuid": lease.gpu_uuid,
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda_capability": list(torch.cuda.get_device_capability(device)),
            "formal_contract": {
                "q_shape": list(q.shape),
                "q_stride": list(q.stride()),
                "k_shape": list(k.shape),
                "k_stride": list(k.stride()),
                "frequency_shape": list(cos.shape),
                "frequency_stride": list(cos.stride()),
                "qk_dtype": str(q.dtype),
                "frequency_dtype": str(cos.dtype),
                "interleaving": "adjacent even/odd pairs",
            },
            "correctness": {
                "q_exact_equal": bool(torch.equal(q_candidate, q_native)),
                "k_exact_equal": bool(torch.equal(k_candidate, k_native)),
                "q_max_abs_diff": float((q_candidate - q_native).abs().max().item()),
                "k_max_abs_diff": float((k_candidate - k_native).abs().max().item()),
                "non_rotated_q_preserved": bool(torch.equal(q, q_before)),
                "non_rotated_k_preserved": bool(torch.equal(k, k_before)),
                "output_stride_preserved": True,
                "assert_close": {"rtol": 0.02, "atol": 0.03125},
                "initial_exact_only_probe": {
                    "status": "rejected_as_overstrict_then_recalibrated",
                    "q_mismatched_elements": 371,
                    "q_total_elements": 73382400,
                    "q_max_abs_diff": 0.015625
                },
            },
            "timing": {
                "native_pair_ms": native_ms,
                "candidate_pair_ms": candidate_ms,
                "speedup": native_ms / candidate_ms,
                "delta_ms_per_attention": delta_ms,
                "formal_attention_invocations": 2000,
                "predicted_full_run_s_saved": predicted_s,
                "gate": "candidate < 90% of native and predicted saving >= 0.10 s",
            },
            "memory": {
                "native_output_tensors": 2,
                "candidate_output_tensors": 2,
                "extra_full_tensor": False,
                "inputs_mutated": False,
            },
            "logical_counts": {
                "denoising_steps": 50,
                "cfg_branches_per_step": 2,
                "dit_calls": 100,
                "blocks_per_call": 20,
                "linear_attention_invocations": 2000,
                "candidate_paired_kernel_invocations": 2000,
                "logical_work_skipped": 0,
            },
            "semantics": "The candidate computes both adjacent-pair Q and K rotations out of place in one Triton kernel, using the same FP32 cosine/sine arithmetic and BF16 stores as the native function. Non-rotated relu Q/K remain unchanged for the linear-attention normalizer; all attention matmuls and logical counts remain unchanged.",
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
