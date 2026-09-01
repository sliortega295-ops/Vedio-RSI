#!/usr/bin/env python3
"""H100 smoke for the selective SANA W8A8 E4M3 pointwise projections."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.models.dits.sana_fp8 import (
    SGLangH100FP8Backend,
    SanaFP8PointwiseConv2d,
)


def timed_ms(fn, iterations: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / iterations


def evaluate_shape(
    *, name: str, in_channels: int, out_channels: int, tokens: int, iterations: int
) -> dict[str, object]:
    conv = nn.Conv2d(in_channels, out_channels, 1, bias=True).cuda().bfloat16()
    x = torch.randn(1, in_channels, 1, tokens, device="cuda", dtype=torch.bfloat16)
    x = x.contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        reference = conv(x)
        backend = SGLangH100FP8Backend()
        candidate = SanaFP8PointwiseConv2d.from_conv(
            conv, module_name=name, backend=backend
        )
        actual = candidate(x)
        ref_f = reference.float().flatten()
        act_f = actual.float().flatten()
        error = act_f - ref_f
        rel_rmse = float(
            torch.sqrt(torch.mean(error.square()))
            / torch.sqrt(torch.mean(ref_f.square())).clamp_min(1e-12)
        )
        cosine = float(F.cosine_similarity(ref_f, act_f, dim=0))
        bf16_ms = timed_ms(lambda: conv(x), iterations)
        fp8_ms = timed_ms(lambda: candidate(x), iterations)
    return {
        "name": name,
        "shape": {
            "tokens": tokens,
            "in_channels": in_channels,
            "out_channels": out_channels,
        },
        "cosine_similarity": cosine,
        "relative_rmse": rel_rmse,
        "max_abs_error": float(error.abs().max()),
        "finite": bool(torch.isfinite(actual).all()),
        "bf16_ms": bf16_ms,
        "fp8_ms": fp8_ms,
        "micro_speedup": bf16_ms / fp8_ms,
        "fp8_calls": candidate.fp8_calls,
        "weight_dtype": str(candidate.weight.dtype),
        "backend": backend.preflight(candidate.weight.device, torch.bfloat16),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-cosine", type=float, default=0.995)
    parser.add_argument("--max-relative-rmse", type=float, default=0.10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cases = [
        evaluate_shape(
            name="conv_inverted",
            in_channels=2240,
            out_channels=13440,
            tokens=args.tokens,
            iterations=args.iterations,
        ),
        evaluate_shape(
            name="conv_point",
            in_channels=6720,
            out_channels=2240,
            tokens=args.tokens,
            iterations=args.iterations,
        ),
    ]
    checks = {
        "finite": all(bool(case["finite"]) for case in cases),
        "cosine": all(
            float(case["cosine_similarity"]) >= args.min_cosine for case in cases
        ),
        "relative_rmse": all(
            float(case["relative_rmse"]) <= args.max_relative_rmse for case in cases
        ),
        "real_fp8_calls": all(int(case["fp8_calls"]) > 0 for case in cases),
        "fp8_weight_dtype": all(
            "float8_e4m3fn" in str(case["weight_dtype"]) for case in cases
        ),
    }
    payload = {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "seed": args.seed,
        "tolerances": {
            "min_cosine": args.min_cosine,
            "max_relative_rmse": args.max_relative_rmse,
        },
        "checks": checks,
        "cases": cases,
        "note": "Microbench timings are diagnostic only; promotion requires matching full generation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
