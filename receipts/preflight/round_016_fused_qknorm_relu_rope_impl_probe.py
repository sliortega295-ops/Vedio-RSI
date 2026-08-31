#!/usr/bin/env python3
"""Exact-shape H100 gate for the installed fused SANA Q/K wrapper."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        from sgl_kernel import rmsnorm

        from sglang.jit_kernel.diffusion.triton.sana_rope import (
            apply_sana_fused_qk_norm_relu_rotary_emb,
            apply_sana_paired_rotary_emb,
        )

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

        def candidate():
            return apply_sana_fused_qk_norm_relu_rotary_emb(
                q,
                k,
                q_weight,
                k_weight,
                cos,
                sin,
                eps=1e-6,
            )

        current, current_ms = _timed(torch, current_chain)
        fused, candidate_ms = _timed(torch, candidate)
        for index in range(4):
            torch.testing.assert_close(fused[index], current[index], rtol=0.02, atol=0.125)
        if not torch.equal(packed, packed_before):
            raise RuntimeError("installed wrapper mutated packed QKV input")
        expected_stride = (tokens * inner_dim, inner_dim, head_dim, 1)
        if any(output.stride() != expected_stride for output in fused):
            raise RuntimeError("installed wrapper changed an output layout")

        guard_checks = {}
        try:
            apply_sana_fused_qk_norm_relu_rotary_emb(
                q[..., ::2],
                k[..., ::2],
                q_weight[::2],
                k_weight[::2],
                cos,
                sin,
                eps=1e-6,
            )
        except ValueError as exc:
            guard_checks["non_last_contiguous_qk"] = {
                "passed": True,
                "error": str(exc),
            }
        else:
            raise RuntimeError("installed wrapper accepted incompatible Q/K layout")

        max_abs = [
            float((fused[index] - current[index]).abs().max().item())
            for index in range(4)
        ]
        delta_ms = current_ms - candidate_ms
        predicted_s = delta_ms * 2000 / 1000
        passed = candidate_ms < current_ms * 0.95 and predicted_s >= 0.10
        source_path = Path(
            sys.modules[apply_sana_fused_qk_norm_relu_rotary_emb.__module__].__file__
        )
        payload = {
            "schema_version": 1,
            "status": "passed" if passed else "rejected",
            "gpu": gpu,
            "lease_uuid": lease.gpu_uuid,
            "torch": torch.__version__,
            "triton": triton.__version__,
            "implementation": {
                "module": apply_sana_fused_qk_norm_relu_rotary_emb.__module__,
                "source_path": str(source_path),
                "source_sha256": _sha256(source_path),
                "fail_closed_guard_checks": guard_checks,
            },
            "formal_contract": {
                "packed_shape": list(packed.shape),
                "packed_stride": list(packed.stride()),
                "q_shape": list(q.shape),
                "q_stride": list(q.stride()),
                "weight_dtype": str(q_weight.dtype),
                "frequency_shape": list(cos.shape),
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
                "Installed wrapper computes learned Q/K RMSNorm, BF16 ReLU, and "
                "paired rotary outputs while preserving the packed QKV input and "
                "all downstream logical work."
            ),
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True))
        return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
