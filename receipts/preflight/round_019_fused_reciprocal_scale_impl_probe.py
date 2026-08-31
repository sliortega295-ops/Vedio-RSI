#!/usr/bin/env python3
"""Exact-shape H100 gate for installed SANA reciprocal-scale wrapper."""

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
            apply_sana_fused_reciprocal_scale,
        )

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
        hidden_before = hidden.clone()
        denominator_before = denominator.clone()

        def current_chain():
            z = 1 / (denominator + 1e-15)
            return hidden * z

        current, current_ms = _timed(torch, current_chain)
        fused, candidate_ms = _timed(
            torch,
            lambda: apply_sana_fused_reciprocal_scale(hidden, denominator),
        )
        torch.testing.assert_close(fused, current, rtol=0, atol=0)
        if not torch.equal(hidden, hidden_before) or not torch.equal(
            denominator, denominator_before
        ):
            raise RuntimeError("installed wrapper mutated an input")
        if fused.stride() != hidden.stride():
            raise RuntimeError("installed wrapper changed the output layout")

        guard_checks = {}
        try:
            apply_sana_fused_reciprocal_scale(hidden[..., ::2], denominator)
        except ValueError as exc:
            guard_checks["noncontiguous_hidden"] = {
                "passed": True,
                "error": str(exc),
            }
        else:
            raise RuntimeError("installed wrapper accepted noncontiguous hidden")
        try:
            apply_sana_fused_reciprocal_scale(hidden, denominator.float())
        except ValueError as exc:
            guard_checks["non_bf16_denominator"] = {
                "passed": True,
                "error": str(exc),
            }
        else:
            raise RuntimeError("installed wrapper accepted FP32 denominator")

        delta_ms = current_ms - candidate_ms
        predicted_s = delta_ms * 2000 / 1000
        passed = candidate_ms < current_ms * 0.95 and predicted_s >= 0.10
        source_path = Path(
            sys.modules[apply_sana_fused_reciprocal_scale.__module__].__file__
        )
        payload = {
            "schema_version": 1,
            "status": "passed" if passed else "rejected",
            "gpu": gpu,
            "lease_uuid": lease.gpu_uuid,
            "torch": torch.__version__,
            "triton": triton.__version__,
            "implementation": {
                "module": apply_sana_fused_reciprocal_scale.__module__,
                "source_path": str(source_path),
                "source_sha256": _sha256(source_path),
                "fail_closed_guard_checks": guard_checks,
            },
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
                "inputs_preserved": bool(
                    torch.equal(hidden, hidden_before)
                    and torch.equal(denominator, denominator_before)
                ),
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
            "semantics": (
                "Installed wrapper preserves the BF16 reciprocal boundary and "
                "BF16 scaled output bit exactly, preserves both inputs/layout, and "
                "removes only the materialized z tensor and one primitive boundary."
            ),
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True))
        return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
