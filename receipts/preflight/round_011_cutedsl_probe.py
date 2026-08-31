#!/usr/bin/env python3
"""Fail-closed exact-shape support gate for R11 CuTe norm+modulation fusion."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lease-file", required=True)
    parser.add_argument("--guard-dir", required=True)
    parser.add_argument("--kernel-source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.guard_dir)
    from gpu_guard import locked_idle_lease

    with locked_idle_lease(args.lease_file) as (lease, gpu):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
        import torch

        device = torch.device("cuda:0")
        capability = torch.cuda.get_device_capability(device)
        dtype = torch.bfloat16
        kernel_source = Path(args.kernel_source)
        source_bytes = kernel_source.read_bytes()
        source_text = source_bytes.decode()
        support_rule_present = (
            "if D % 256 != 0 or D > 8192:" in source_text
            and "D={D} not supported, must be multiple of 256 and <= 8192" in source_text
        )
        cuda_bindings_available = importlib.util.find_spec("cuda.bindings.driver") is not None
        cutlass_available = importlib.util.find_spec("cutlass") is not None
        import_error = None
        observed_error = None
        cache_before = None
        cache_after = None
        if cutlass_available:
            try:
                from sglang.jit_kernel.diffusion.cutedsl.scale_residual_norm_scale_shift import (
                    _COMPILE_CACHE,
                    fused_norm_scale_shift,
                )

                hidden = torch.zeros((1, 32760, 2240), device=device, dtype=dtype)
                scale = torch.zeros((1, 1, 2240), device=device, dtype=dtype)
                shift = torch.zeros((1, 1, 2240), device=device, dtype=dtype)
                cache_before = len(_COMPILE_CACHE)
                try:
                    fused_norm_scale_shift(
                        hidden, None, None, scale, shift, "layer", 1e-6
                    )
                except Exception as exc:
                    observed_error = f"{type(exc).__name__}: {exc}"
                cache_after = len(_COMPILE_CACHE)
            except Exception as exc:
                import_error = f"{type(exc).__name__}: {exc}"
        else:
            import_error = "ModuleNotFoundError: No module named 'cutlass'"

        status = (
            "rejected"
            if capability == (9, 0)
            and cuda_bindings_available
            and not cutlass_available
            and support_rule_present
            and 2240 % 256 != 0
            else "unexpected"
        )
        payload = {
            "schema_version": 1,
            "status": status,
            "disposition": "preflight_rejection_not_a_formal_round",
            "gpu": gpu,
            "lease_uuid": lease.gpu_uuid,
            "torch": torch.__version__,
            "cuda_capability": list(capability),
            "sm90": capability == (9, 0),
            "imports": {
                "cuda_bindings_available": cuda_bindings_available,
                "cutlass_available": cutlass_available,
                "fused_norm_scale_shift_imported": cutlass_available and import_error is None,
                "observed_import_error": import_error,
            },
            "first_live_attempt": {
                "status": "failed_before_candidate_execution",
                "error": "ModuleNotFoundError: No module named 'cutlass'",
                "preserved": True,
            },
            "formal_contract": {
                "hidden_shape": [1, 32760, 2240],
                "hidden_stride": [73382400, 2240, 1],
                "hidden_dtype": str(dtype),
                "scale_shape": [1, 1, 2240],
                "scale_stride": [2240, 2240, 1],
                "shift_shape": [1, 1, 2240],
                "shift_stride": [2240, 2240, 1],
                "weight": None,
                "bias": None,
                "norm_type": "layer",
                "eps": 1e-6,
                "elementwise_affine": False,
            },
            "support_rule": "D must be a multiple of 256 and <= 8192",
            "support_rule_present_in_pinned_source": support_rule_present,
            "kernel_source_path": str(kernel_source),
            "kernel_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "formal_hidden_dim": 2240,
            "formal_hidden_dim_mod_256": 2240 % 256,
            "observed_error": observed_error,
            "compile_cache_entries_before": cache_before,
            "compile_cache_entries_after": cache_after,
            "lazy_compilation_reached": (
                cache_before is not None and cache_after != cache_before
            ),
            "warmup_accounting": "No CuTe kernel compiled or ran; candidate is rejected before lazy compilation, so there is no candidate warmup to charge to a formal total.",
            "reason": "The pinned environment lacks the CuTe DSL cutlass Python module, and independently the exact SANA hidden dimension 2240 violates the pinned kernel's hard D%256==0 contract. Padding to 2304 or adding dependencies would be a different implementation and is outside this existing-kernel hypothesis.",
        }
        if status != "rejected":
            raise RuntimeError(json.dumps(payload, indent=2, sort_keys=True))
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
