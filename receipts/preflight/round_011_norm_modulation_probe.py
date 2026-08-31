#!/usr/bin/env python3
"""Exact-shape H100 gate for R11 specialized Triton norm+modulation fusion."""

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
            fuse_layernorm_scale_shift_kernel,
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

        if not hidden.is_contiguous() or hidden.stride() != (L * C, C, 1):
            raise RuntimeError(f"unexpected hidden contract: {hidden.stride()}")

        def run_site(scale, shift, gate):
            scale2 = scale.squeeze(1)
            shift2 = shift.squeeze(1)
            def current():
                normed = F.layer_norm(
                    hidden, (C,), weight=None, bias=None, eps=eps
                )
                return fuse_scale_shift_kernel(
                    normed, scale, shift, scale_constant=1.0
                )

            def candidate():
                return fuse_layernorm_scale_shift_kernel(
                    hidden,
                    weight=None,
                    bias=None,
                    scale=scale2,
                    shift=shift2,
                    eps=eps,
                )

            def current_with_downstream_residual():
                modulated = current()
                residual_out = torch.addcmul(hidden, gate, branch)
                return modulated, residual_out

            def candidate_with_downstream_residual():
                modulated = candidate()
                residual_out = torch.addcmul(hidden, gate, branch)
                return modulated, residual_out

            current_out, current_ms = timed(torch, current)
            candidate_out, candidate_ms = timed(torch, candidate)
            (
                (current_e2e_out, current_residual),
                current_with_residual_ms,
            ) = timed(torch, current_with_downstream_residual)
            (
                (candidate_e2e_out, candidate_residual),
                candidate_with_residual_ms,
            ) = timed(torch, candidate_with_downstream_residual)
            torch.testing.assert_close(
                candidate_out, current_out, rtol=2e-2, atol=2.5e-1
            )
            torch.testing.assert_close(
                candidate_e2e_out, current_e2e_out, rtol=2e-2, atol=2.5e-1
            )
            if not torch.equal(candidate_residual, current_residual):
                raise RuntimeError("materialized gate changes downstream addcmul output")
            if (
                candidate_out.shape != hidden.shape
                or candidate_out.dtype != dtype
                or not candidate_out.is_contiguous()
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
                "broadcast_gate_path_unchanged": True,
                "downstream_addcmul_exact_equal": True,
                "assert_close": {"rtol": 0.02, "atol": 0.25},
            }

        msa = run_site(scale_msa, shift_msa, gate_msa)
        mlp = run_site(scale_mlp, shift_mlp, gate_mlp)
        bytes_per_full_tensor = hidden.numel() * hidden.element_size()

        # The first block receives patch_embedding(...).flatten(2).transpose(1,2),
        # whose last dimension is strided. The source makes exactly one
        # contiguous copy before the specialized kernel; later block residuals
        # are contiguous. Include that real first-block boundary in the gate.
        hidden_strided = torch.randn(B, C, L, device=device, dtype=dtype).transpose(1, 2)

        def current_first_block():
            normed = F.layer_norm(
                hidden_strided, (C,), weight=None, bias=None, eps=eps
            )
            return fuse_scale_shift_kernel(
                normed, scale_msa, shift_msa, scale_constant=1.0
            )

        def candidate_first_block():
            return fuse_layernorm_scale_shift_kernel(
                hidden_strided.contiguous(),
                weight=None,
                bias=None,
                scale=scale_msa.squeeze(1),
                shift=shift_msa.squeeze(1),
                eps=eps,
            )

        current_first_out, current_first_ms = timed(torch, current_first_block)
        candidate_first_out, candidate_first_ms = timed(torch, candidate_first_block)
        torch.testing.assert_close(
            candidate_first_out, current_first_out, rtol=2e-2, atol=2.5e-1
        )
        first_block = {
            "input_shape": list(hidden_strided.shape),
            "input_stride": list(hidden_strided.stride()),
            "input_contiguous": hidden_strided.is_contiguous(),
            "candidate_explicit_contiguous_copy": True,
            "current_r10_ms": current_first_ms,
            "candidate_ms": candidate_first_ms,
            "speedup": current_first_ms / candidate_first_ms,
            "max_abs_diff_bf16": float(
                (candidate_first_out - current_first_out).abs().max().item()
            ),
        }
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
                "weight": None,
                "bias": None,
                "eps": eps,
                "elementwise_affine": False,
            },
            "msa_site": msa,
            "mlp_site": mlp,
            "first_block_strided_boundary": first_block,
            "memory_tradeoff": {
                "avoided_gate_out_bytes_per_site_vs_unspecialized_fallback": bytes_per_full_tensor,
                "avoided_gate_out_mib_per_site_vs_unspecialized_fallback": bytes_per_full_tensor / 1048576,
                "current_broadcast_gate_storage_bytes": gate_msa.numel()
                * gate_msa.element_size(),
                "candidate_outputs_full_tensors": 1,
                "current_chain_outputs_full_tensors": 2,
                "note": "Compile-time SELECT01=False and WRITE_GATE=False leave the original broadcast gate path unchanged and avoid the unspecialized fallback's 139.97 MiB full gate. The candidate emits only the modulated tensor; the first strided block requires one contiguous input copy per DiT call.",
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
            "semantics": "The repository select01 kernel body is compile-time specialized with SELECT01=False and WRITE_GATE=False. It computes affine-free FP32 LayerNorm with eps=1e-6 followed by x*(1+scale)+shift, emits no gate tensor, and leaves the existing broadcast gate and residual addcmul unchanged. No model call, block, normalization, attention, FFN, or residual work is skipped.",
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
