#!/usr/bin/env python3
"""Exact-shape H100 gate for SANA's existing Triton scale/shift kernel."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def timed(torch, fn, repeats=8):
    for _ in range(3):
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
        from sglang.jit_kernel.diffusion.triton.scale_shift import (
            fuse_scale_shift_kernel,
        )

        torch.manual_seed(9)
        device = torch.device("cuda:0")
        dtype = torch.bfloat16

        # Match the actual block source: the per-block [6,D] table and the
        # timestep [B,S,6D] are added, then unbound into [B,S,D] views.
        table = torch.randn(6, 2240, device=device, dtype=dtype)
        timestep = torch.randn(1, 1, 6 * 2240, device=device, dtype=dtype)
        modulation = table[None, None] + timestep.reshape(1, 1, 6, 2240)
        shift_msa, scale_msa, _, shift_mlp, scale_mlp, _ = modulation.unbind(dim=2)
        hidden = torch.randn(1, 32760, 2240, device=device, dtype=dtype)

        if not hidden.is_contiguous():
            raise RuntimeError("formal LayerNorm output surrogate must be contiguous")
        for name, tensor in {
            "shift_msa": shift_msa,
            "scale_msa": scale_msa,
            "shift_mlp": shift_mlp,
            "scale_mlp": scale_mlp,
        }.items():
            if tensor.shape != (1, 1, 2240) or tensor.dtype != dtype:
                raise RuntimeError(f"unexpected {name} contract: {tensor.shape}, {tensor.dtype}")

        def native_msa():
            return hidden * (1 + scale_msa) + shift_msa

        def fused_msa():
            return fuse_scale_shift_kernel(
                hidden, scale_msa, shift_msa, scale_constant=1.0
            )

        def native_mlp():
            return hidden * (1 + scale_mlp) + shift_mlp

        def fused_mlp():
            return fuse_scale_shift_kernel(
                hidden, scale_mlp, shift_mlp, scale_constant=1.0
            )

        native_msa_out, native_msa_ms = timed(torch, native_msa)
        fused_msa_out, fused_msa_ms = timed(torch, fused_msa)
        native_mlp_out, native_mlp_ms = timed(torch, native_mlp)
        fused_mlp_out, fused_mlp_ms = timed(torch, fused_mlp)

        torch.testing.assert_close(
            fused_msa_out, native_msa_out, rtol=2e-2, atol=1.25e-1
        )
        torch.testing.assert_close(
            fused_mlp_out, native_mlp_out, rtol=2e-2, atol=1.25e-1
        )
        for name, output in {
            "fused_msa": fused_msa_out,
            "fused_mlp": fused_mlp_out,
        }.items():
            if (
                output.shape != hidden.shape
                or output.dtype != hidden.dtype
                or output.device != hidden.device
                or not output.is_contiguous()
            ):
                raise RuntimeError(f"unexpected {name} output contract")

        payload = {
            "schema_version": 1,
            "status": "passed",
            "gpu": gpu,
            "lease_uuid": lease.gpu_uuid,
            "torch": torch.__version__,
            "formal_contract": {
                "hidden_shape": list(hidden.shape),
                "hidden_stride": list(hidden.stride()),
                "modulation_shape": list(modulation.shape),
                "modulation_stride": list(modulation.stride()),
                "view_shape": list(scale_msa.shape),
                "view_stride": list(scale_msa.stride()),
                "dtype": str(dtype),
                "output_contiguous": True,
            },
            "msa_site": {
                "native_ms": native_msa_ms,
                "fused_ms": fused_msa_ms,
                "speedup": native_msa_ms / fused_msa_ms,
                "exact_equal": bool(torch.equal(fused_msa_out, native_msa_out)),
                "max_abs_diff_bf16": float(
                    (fused_msa_out - native_msa_out).abs().max().item()
                ),
                "assert_close": {"rtol": 0.02, "atol": 0.125},
            },
            "mlp_site": {
                "native_ms": native_mlp_ms,
                "fused_ms": fused_mlp_ms,
                "speedup": native_mlp_ms / fused_mlp_ms,
                "exact_equal": bool(torch.equal(fused_mlp_out, native_mlp_out)),
                "max_abs_diff_bf16": float(
                    (fused_mlp_out - native_mlp_out).abs().max().item()
                ),
                "assert_close": {"rtol": 0.02, "atol": 0.125},
            },
            "logical_counts": {
                "denoising_steps": 50,
                "cfg_branches_per_step": 2,
                "dit_calls": 100,
                "blocks_per_call": 20,
                "block_calls": 2000,
                "modulation_calls": 4000,
                "native_pointwise_launches_per_call": 3,
                "fused_pointwise_launches_per_call": 1,
                "removed_cuda_launches": 8000,
            },
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
