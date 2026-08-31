#!/usr/bin/env python3
"""Exact-shape H100 gate/residual probe for SANA block epilogue fusion."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def timed(torch, fn, repeats=5):
    for _ in range(2):
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
        from sgl_kernel import silu_and_mul
        from sglang.multimodal_gen.runtime.models.dits.sana_video import GLUMBTempConv

        torch.manual_seed(8)
        device = torch.device("cuda:0")

        # Full real module logic on a bounded shape validates the weight-half
        # permutation through conv_inverted and the grouped depthwise conv.
        os.environ["SGLANG_SANA_BLOCK_EPILOGUE_FUSION"] = "0"
        reference_module = GLUMBTempConv(32, 32, expand_ratio=3.0).to(
            device=device, dtype=torch.bfloat16
        )
        os.environ["SGLANG_SANA_BLOCK_EPILOGUE_FUSION"] = "1"
        fused_module = GLUMBTempConv(32, 32, expand_ratio=3.0).to(
            device=device, dtype=torch.bfloat16
        )
        fused_module.load_state_dict(reference_module.state_dict())
        fused_module.build_fused_gate()
        module_input = torch.randn(
            1, 3, 4, 5, 32, device=device, dtype=torch.bfloat16
        )
        with torch.no_grad():
            module_reference = reference_module(module_input)
            module_fused = fused_module(module_input)
        module_max_abs = float((module_fused - module_reference).abs().max().item())
        torch.testing.assert_close(
            module_fused, module_reference, rtol=2e-2, atol=1e-3
        )

        # Exact formal GLU tensor shape after the 20x block's depthwise conv.
        gate_input = torch.empty(
            (21, 13440, 30, 52),
            device=device,
            dtype=torch.bfloat16,
            memory_format=torch.channels_last,
        ).normal_()
        gate_nhwc = gate_input.permute(0, 2, 3, 1)
        if not gate_nhwc.is_contiguous():
            raise RuntimeError("exact Conv2d channels-last view is not NHWC contiguous")

        def native_gate():
            gate, value = gate_input.chunk(2, dim=1)
            return value * F.silu(gate)

        def fused_gate():
            return silu_and_mul(gate_nhwc).permute(0, 3, 1, 2)

        native_gate_out, native_gate_ms = timed(torch, native_gate)
        fused_gate_out, fused_gate_ms = timed(torch, fused_gate)
        gate_max_abs = float((fused_gate_out - native_gate_out).abs().max().item())
        gate_exact = bool(torch.equal(fused_gate_out, native_gate_out))
        torch.testing.assert_close(
            fused_gate_out, native_gate_out, rtol=2e-2, atol=6.25e-2
        )

        # Exact formal B,N,C residual tensor and broadcast gate shape.
        residual = torch.randn(
            1, 32760, 2240, device=device, dtype=torch.bfloat16
        )
        branch_value = torch.randn_like(residual)
        branch_gate = torch.randn(1, 1, 2240, device=device, dtype=torch.bfloat16)

        def native_residual():
            return residual + branch_gate * branch_value

        def fused_residual():
            return torch.addcmul(residual, branch_gate, branch_value)

        native_res_out, native_res_ms = timed(torch, native_residual)
        fused_res_out, fused_res_ms = timed(torch, fused_residual)
        residual_max_abs = float((fused_res_out - native_res_out).abs().max().item())
        residual_exact = bool(torch.equal(fused_res_out, native_res_out))
        torch.testing.assert_close(
            fused_res_out, native_res_out, rtol=2e-2, atol=1.25e-1
        )

        payload = {
            "schema_version": 1,
            "status": "passed",
            "gpu": gpu,
            "lease_uuid": lease.gpu_uuid,
            "torch": torch.__version__,
            "bounded_full_module": {
                "input_shape": [1, 3, 4, 5, 32],
                "weight_half_swap_and_depthwise_mapping_passed": True,
                "max_abs_diff_bf16": module_max_abs,
                "assert_close": {"rtol": 0.02, "atol": 0.001},
            },
            "formal_gate_shape": {
                "nchw": [21, 13440, 30, 52],
                "nhwc_view_contiguous": True,
                "native_ms": native_gate_ms,
                "aot_fused_ms": fused_gate_ms,
                "speedup": native_gate_ms / fused_gate_ms,
                "exact_equal": gate_exact,
                "max_abs_diff_bf16": gate_max_abs,
                "assert_close": {"rtol": 0.02, "atol": 0.0625},
            },
            "formal_residual_shape": {
                "value": [1, 32760, 2240],
                "gate": [1, 1, 2240],
                "native_ms": native_res_ms,
                "addcmul_ms": fused_res_ms,
                "speedup": native_res_ms / fused_res_ms,
                "exact_equal": residual_exact,
                "max_abs_diff_bf16": residual_max_abs,
                "assert_close": {"rtol": 0.02, "atol": 0.125},
            },
            "logical_counts": {
                "block_calls": 2000,
                "aot_gate_calls": 2000,
                "gated_residual_calls": 4000,
                "removed_cuda_launches": 6000,
            },
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
