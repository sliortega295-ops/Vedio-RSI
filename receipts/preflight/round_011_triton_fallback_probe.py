#!/usr/bin/env python3
"""Exact-shape H100 gate for R11 repository Triton LayerNorm fusion fallback."""

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
        import torch.nn.functional as F
        from sglang.jit_kernel.diffusion.triton.scale_shift import (
            fuse_layernorm_scale_shift_gate_select01_kernel,
            fuse_scale_shift_kernel,
        )

        torch.manual_seed(11)
        device = torch.device("cuda:0")
        dtype = torch.bfloat16
        B, L, C = 1, 32760, 2240
        eps = 1e-6
        table = torch.randn(6, C, device=device, dtype=dtype)
        timestep = torch.randn(B, 1, 6 * C, device=device, dtype=dtype)
        modulation = table[None, None] + timestep.reshape(B, 1, 6, C)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            modulation.unbind(dim=2)
        )
        hidden = torch.randn(B, L, C, device=device, dtype=dtype)
        branch = torch.randn_like(hidden)
        index0 = torch.zeros(B, L, device=device, dtype=torch.int32)

        if not hidden.is_contiguous() or hidden.stride() != (L * C, C, 1):
            raise RuntimeError(f"unexpected hidden contract: {hidden.stride()}")

        def run_site(scale, shift, gate):
            scale2 = scale.squeeze(1)
            shift2 = shift.squeeze(1)
            gate2 = gate.squeeze(1)

            def current():
                normed = F.layer_norm(
                    hidden, (C,), weight=None, bias=None, eps=eps
                )
                return fuse_scale_shift_kernel(
                    normed, scale, shift, scale_constant=1.0
                )

            def candidate():
                return fuse_layernorm_scale_shift_gate_select01_kernel(
                    hidden,
                    weight=None,
                    bias=None,
                    scale0=scale2,
                    shift0=shift2,
                    gate0=gate2,
                    scale1=scale2,
                    shift1=shift2,
                    gate1=gate2,
                    index=index0,
                    eps=eps,
                )

            def current_with_downstream_residual():
                modulated = current()
                residual_out = torch.addcmul(hidden, gate, branch)
                return modulated, residual_out

            def candidate_with_downstream_residual():
                modulated, gate_out = candidate()
                residual_out = torch.addcmul(hidden, gate_out, branch)
                return modulated, residual_out

            current_out, current_ms = timed(torch, current)
            (candidate_out, gate_out), candidate_ms = timed(torch, candidate)
            (
                (current_e2e_out, current_residual),
                current_with_residual_ms,
            ) = timed(torch, current_with_downstream_residual)
            (
                (candidate_e2e_out, candidate_residual),
                candidate_with_residual_ms,
            ) = timed(torch, candidate_with_downstream_residual)
            gate_ref = gate.expand_as(hidden)
            torch.testing.assert_close(
                candidate_out, current_out, rtol=2e-2, atol=2.5e-1
            )
            if not torch.equal(gate_out, gate_ref):
                raise RuntimeError("candidate gate output does not exactly match broadcast gate")
            torch.testing.assert_close(
                candidate_e2e_out, current_e2e_out, rtol=2e-2, atol=2.5e-1
            )
            if not torch.equal(candidate_residual, current_residual):
                raise RuntimeError("materialized gate changes downstream addcmul output")
            if (
                candidate_out.shape != hidden.shape
                or candidate_out.dtype != dtype
                or not candidate_out.is_contiguous()
                or gate_out.shape != hidden.shape
                or gate_out.dtype != dtype
                or not gate_out.is_contiguous()
            ):
                raise RuntimeError("unexpected candidate output contract")
            return {
                "current_r10_ms": current_ms,
                "candidate_ms": candidate_ms,
                "speedup": current_ms / candidate_ms,
                "current_with_downstream_residual_ms": current_with_residual_ms,
                "candidate_with_downstream_residual_ms": candidate_with_residual_ms,
                "end_to_end_relevant_speedup": (
                    current_with_residual_ms / candidate_with_residual_ms
                ),
                "end_to_end_relevant_delta_ms": (
                    current_with_residual_ms - candidate_with_residual_ms
                ),
                "candidate_out_max_abs_diff_bf16": float(
                    (candidate_out - current_out).abs().max().item()
                ),
                "candidate_out_exact_equal": bool(
                    torch.equal(candidate_out, current_out)
                ),
                "gate_out_exact_equal_to_broadcast": True,
                "downstream_addcmul_exact_equal": True,
                "assert_close": {"rtol": 0.02, "atol": 0.25},
            }

        msa = run_site(scale_msa, shift_msa, gate_msa)
        mlp = run_site(scale_mlp, shift_mlp, gate_mlp)
        bytes_per_full_tensor = hidden.numel() * hidden.element_size()
        payload = {
            "schema_version": 1,
            "status": "passed",
            "gpu": gpu,
            "lease_uuid": lease.gpu_uuid,
            "torch": torch.__version__,
            "cuda_capability": list(torch.cuda.get_device_capability(device)),
            "formal_contract": {
                "hidden_shape": list(hidden.shape),
                "hidden_stride": list(hidden.stride()),
                "dtype": str(dtype),
                "modulation_shape": list(modulation.shape),
                "modulation_stride": list(modulation.stride()),
                "site_view_shape": list(scale_msa.shape),
                "site_2d_shape": list(scale_msa.squeeze(1).shape),
                "index_shape": list(index0.shape),
                "index_dtype": str(index0.dtype),
                "weight": None,
                "bias": None,
                "eps": eps,
                "elementwise_affine": False,
            },
            "msa_site": msa,
            "mlp_site": mlp,
            "memory_tradeoff": {
                "materialized_gate_out_bytes_per_site": bytes_per_full_tensor,
                "materialized_gate_out_mib_per_site": bytes_per_full_tensor / 1048576,
                "current_broadcast_gate_storage_bytes": gate_msa.numel()
                * gate_msa.element_size(),
                "candidate_outputs_full_tensors": 2,
                "current_chain_outputs_full_tensors": 2,
                "note": "Candidate replaces the separate LayerNorm and modulation outputs with one modulated output plus an explicit full gate tensor. The extra full gate is consumed immediately by the existing residual addcmul; formal full-run peak must be measured.",
            },
            "logical_counts": {
                "denoising_steps": 50,
                "cfg_branches_per_step": 2,
                "dit_calls": 100,
                "blocks_per_call": 20,
                "block_calls": 2000,
                "norm_modulation_sites": 4000,
                "current_kernels_per_site": 2,
                "candidate_kernels_per_site": 1,
                "removed_cuda_launches_per_formal_run": 4000,
                "logical_work_skipped": 0,
            },
            "semantics": "Both select branches receive the same SANA scale, shift, and gate and index is identically zero. The kernel computes affine-free FP32 LayerNorm with eps=1e-6 followed by x*(1+scale)+shift, and materializes the unchanged broadcast gate for the existing residual addcmul. No model call, block, normalization, attention, FFN, or residual work is skipped.",
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
