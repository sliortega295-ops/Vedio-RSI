#!/usr/bin/env python3
"""Exact-shape H100 gate for installed SANA cross-output layout wrapper."""

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
        import torch.nn.functional as F
        import triton

        from sglang.jit_kernel.diffusion.triton.sana_rope import (
            apply_sana_cross_attention_output_layout,
        )

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
        input_before = sdpa_output.clone()

        def current_chain():
            layout = sdpa_output.transpose(1, 2).reshape(batch, tokens, inner_dim)
            return layout, F.linear(layout, weight, bias)

        def candidate_chain():
            layout = apply_sana_cross_attention_output_layout(sdpa_output)
            return layout, F.linear(layout, weight, bias)

        current, current_ms = _timed(torch, current_chain)
        candidate, candidate_ms = _timed(torch, candidate_chain)
        torch.testing.assert_close(candidate[0], current[0], rtol=0, atol=0)
        torch.testing.assert_close(candidate[1], current[1], rtol=0, atol=0)
        if not torch.equal(sdpa_output, input_before):
            raise RuntimeError("installed wrapper mutated SDPA output")
        expected_stride = (tokens * inner_dim, inner_dim, 1)
        if candidate[0].stride() != expected_stride:
            raise RuntimeError("installed wrapper changed dense output layout")

        guard_checks = {}
        try:
            apply_sana_cross_attention_output_layout(sdpa_output.transpose(2, 3))
        except ValueError as exc:
            guard_checks["noncontiguous_input"] = {
                "passed": True,
                "error": str(exc),
            }
        else:
            raise RuntimeError("installed wrapper accepted noncontiguous input")
        try:
            apply_sana_cross_attention_output_layout(sdpa_output.float())
        except ValueError as exc:
            guard_checks["non_bf16_input"] = {
                "passed": True,
                "error": str(exc),
            }
        else:
            raise RuntimeError("installed wrapper accepted FP32 input")

        delta_ms = current_ms - candidate_ms
        predicted_s = delta_ms * 2000 / 1000
        passed = candidate_ms < current_ms * 0.98 and predicted_s >= 0.10
        source_path = Path(
            sys.modules[apply_sana_cross_attention_output_layout.__module__].__file__
        )
        payload = {
            "schema_version": 1,
            "status": "passed" if passed else "rejected",
            "gpu": gpu,
            "lease_uuid": lease.gpu_uuid,
            "torch": torch.__version__,
            "triton": triton.__version__,
            "implementation": {
                "module": apply_sana_cross_attention_output_layout.__module__,
                "source_path": str(source_path),
                "source_sha256": _sha256(source_path),
                "fail_closed_guard_checks": guard_checks,
            },
            "formal_contract": {
                "sdpa_output_shape": list(sdpa_output.shape),
                "sdpa_output_stride": list(sdpa_output.stride()),
                "projection_weight_shape": list(weight.shape),
                "dtype": str(sdpa_output.dtype),
            },
            "correctness": {
                "layout_exact": bool(torch.equal(candidate[0], current[0])),
                "projection_exact": bool(torch.equal(candidate[1], current[1])),
                "input_preserved": bool(torch.equal(sdpa_output, input_before)),
                "output_stride_preserved": candidate[0].stride()
                == expected_stride,
            },
            "timing": {
                "current_native_layout_plus_projection_ms": current_ms,
                "candidate_installed_layout_plus_projection_ms": candidate_ms,
                "speedup": current_ms / candidate_ms,
                "delta_ms_per_cross_attention": delta_ms,
                "formal_cross_attention_invocations": 2000,
                "predicted_full_run_s_saved": predicted_s,
                "gate": "candidate < 98% of current and predicted saving >= 0.10 s",
            },
            "semantics": (
                "Installed wrapper materializes exactly the same dense projection "
                "input from SDPA output, preserves the input, and leaves the output "
                "projection and every logical model operation unchanged."
            ),
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True))
        return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
