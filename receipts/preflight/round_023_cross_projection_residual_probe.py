#!/usr/bin/env python3
"""Exact-shape gate for cross-attention output-projection residual fusion."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path


def _measure(torch, fn, *, trials: int = 5, repeats: int = 6):
    for _ in range(3):
        value = fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            value = fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / repeats)
    return value, samples, statistics.median(samples)


def _difference(torch, candidate, reference):
    delta = (candidate.float() - reference.float()).abs()
    return {
        "exact": bool(torch.equal(candidate, reference)),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "different_elements": int(torch.count_nonzero(candidate != reference).item()),
        "total_elements": candidate.numel(),
    }


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

        torch.manual_seed(23)
        device = torch.device("cuda:0")
        batch, tokens, dim = 1, 32760, 2240
        projected_input = torch.randn(
            batch, tokens, dim, device=device, dtype=torch.bfloat16
        )
        residual = torch.randn_like(projected_input)
        weight = torch.randn(dim, dim, device=device, dtype=torch.bfloat16) / dim**0.5
        bias = torch.randn(dim, device=device, dtype=torch.bfloat16)
        flat_input = projected_input.reshape(-1, dim)
        flat_residual = residual.reshape(-1, dim)

        def current_chain():
            return F.linear(projected_input, weight, bias) + residual

        def residual_epilogue_then_bias():
            output = torch.addmm(flat_residual, flat_input, weight.t(), beta=1.0)
            return (output + bias).reshape(batch, tokens, dim)

        def preadd_bias_then_epilogue():
            base = flat_residual + bias
            return torch.addmm(base, flat_input, weight.t(), beta=1.0).reshape(
                batch, tokens, dim
            )

        current, current_samples, current_ms = _measure(torch, current_chain)
        candidate_a, candidate_a_samples, candidate_a_ms = _measure(
            torch, residual_epilogue_then_bias
        )
        candidate_b, candidate_b_samples, candidate_b_ms = _measure(
            torch, preadd_bias_then_epilogue
        )

        variants = {
            "residual_epilogue_then_bias": {
                "timing_samples_ms": candidate_a_samples,
                "median_ms": candidate_a_ms,
                "speedup": current_ms / candidate_a_ms,
                "predicted_full_run_s_saved": (current_ms - candidate_a_ms) * 2.0,
                "correctness": _difference(torch, candidate_a, current),
            },
            "preadd_bias_then_residual_epilogue": {
                "timing_samples_ms": candidate_b_samples,
                "median_ms": candidate_b_ms,
                "speedup": current_ms / candidate_b_ms,
                "predicted_full_run_s_saved": (current_ms - candidate_b_ms) * 2.0,
                "correctness": _difference(torch, candidate_b, current),
            },
        }
        exact_variants = [
            name for name, result in variants.items() if result["correctness"]["exact"]
        ]
        passed_variants = [
            name
            for name in exact_variants
            if variants[name]["median_ms"] < current_ms * 0.98
            and variants[name]["predicted_full_run_s_saved"] >= 0.10
        ]
        status = "passed" if passed_variants else "rejected"
        payload = {
            "schema_version": 1,
            "status": status,
            "gpu": gpu,
            "lease_uuid": lease.gpu_uuid,
            "torch": torch.__version__,
            "formal_contract": {
                "projected_input_shape": list(projected_input.shape),
                "residual_shape": list(residual.shape),
                "weight_shape": list(weight.shape),
                "bias_shape": list(bias.shape),
                "dtype": str(projected_input.dtype),
                "formal_cross_attention_invocations": 2000,
            },
            "current": {
                "operations": "BF16 linear with bias, followed by BF16 residual add",
                "timing_samples_ms": current_samples,
                "median_ms": current_ms,
            },
            "variants": variants,
            "gate": {
                "requires": "bit-exact BF16 output, >=2% local gain, and >=0.10 s predicted full-run saving",
                "exact_variants": exact_variants,
                "passed_variants": passed_variants,
            },
            "decision": (
                "Promote the fastest passing exact variant."
                if passed_variants
                else "Reject: no addmm ordering preserved the native BF16 output while meeting the performance gate."
            ),
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True))
        return 0 if status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
