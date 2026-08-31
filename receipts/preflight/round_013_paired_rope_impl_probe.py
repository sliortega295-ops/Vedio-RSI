#!/usr/bin/env python3
"""Exact-shape H100 gate for the installed paired SANA Q/K RoPE wrapper."""

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

        from sglang.jit_kernel.diffusion.triton.sana_rope import (
            apply_sana_paired_rotary_emb,
        )

        def native(x, cos, sin):
            x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
            cosine = cos[..., 0::2]
            sine = sin[..., 1::2]
            out = torch.empty_like(x)
            out[..., 0::2] = x1 * cosine - x2 * sine
            out[..., 1::2] = x1 * sine + x2 * cosine
            return out.type_as(x)

        torch.manual_seed(13)
        device = torch.device("cuda:0")
        batch, tokens, heads, head_dim = 1, 32760, 20, 112
        q = torch.relu(
            torch.randn(
                batch,
                tokens,
                heads,
                head_dim,
                device=device,
                dtype=torch.bfloat16,
            )
        )
        k = torch.relu(torch.randn_like(q))
        cos = torch.randn(
            1, tokens, 1, head_dim, device=device, dtype=torch.float32
        )
        sin = torch.randn_like(cos)
        q_before = q.clone()
        k_before = k.clone()

        def run_native():
            return native(q, cos, sin), native(k, cos, sin)

        def run_candidate():
            return apply_sana_paired_rotary_emb(q, k, cos, sin)

        (q_native, k_native), native_ms = _timed(torch, run_native)
        (q_candidate, k_candidate), candidate_ms = _timed(torch, run_candidate)
        torch.testing.assert_close(q_candidate, q_native, rtol=0.02, atol=0.03125)
        torch.testing.assert_close(k_candidate, k_native, rtol=0.02, atol=0.03125)
        if not torch.equal(q, q_before) or not torch.equal(k, k_before):
            raise RuntimeError("installed wrapper mutated non-rotated Q/K")
        if q_candidate.stride() != q.stride() or k_candidate.stride() != k.stride():
            raise RuntimeError("installed wrapper changed Q/K output layout")

        guard_checks = {}
        invalid_q = q[..., ::2]
        invalid_k = k[..., ::2]
        try:
            apply_sana_paired_rotary_emb(invalid_q, invalid_k, cos[..., ::2], sin[..., ::2])
        except ValueError as exc:
            guard_checks["noncontiguous_qk"] = {"passed": True, "error": str(exc)}
        else:
            raise RuntimeError("installed wrapper accepted noncontiguous Q/K")

        source_path = Path(sys.modules[apply_sana_paired_rotary_emb.__module__].__file__)
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
            "implementation": {
                "module": apply_sana_paired_rotary_emb.__module__,
                "source_path": str(source_path),
                "source_sha256": _sha256(source_path),
                "fail_closed_guard_checks": guard_checks,
            },
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
            "semantics": (
                "Installed wrapper computes both out-of-place adjacent-pair Q/K "
                "rotations in one Triton kernel. Non-rotated relu Q/K, attention "
                "matmuls, and logical work counts are unchanged."
            ),
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True))
        return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
