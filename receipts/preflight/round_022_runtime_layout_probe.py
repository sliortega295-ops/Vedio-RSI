#!/usr/bin/env python3
"""Record the real-layout SDPA output used by SANA cross attention."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


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

        from sglang.jit_kernel.diffusion.triton.sana_rope import (
            apply_sana_cross_attention_output_layout,
        )

        torch.manual_seed(2201)
        device = torch.device("cuda:0")
        batch, heads, tokens, text_tokens, head_dim = 1, 20, 32760, 300, 112
        inner_dim = heads * head_dim

        # Match the model path: contiguous [B,S,H,D] linear outputs are
        # transposed to the logical [B,H,S,D] SDPA inputs.
        query_storage = torch.randn(
            batch, tokens, heads, head_dim, device=device, dtype=torch.bfloat16
        )
        key_storage = torch.randn(
            batch, text_tokens, heads, head_dim, device=device, dtype=torch.bfloat16
        )
        value_storage = torch.randn_like(key_storage)
        query = query_storage.transpose(1, 2)
        key = key_storage.transpose(1, 2)
        value = value_storage.transpose(1, 2)

        output = F.scaled_dot_product_attention(query, key, value)
        torch.cuda.synchronize()
        native = output.transpose(1, 2).reshape(batch, tokens, inner_dim)
        reference = output.transpose(1, 2).contiguous().reshape(
            batch, tokens, inner_dim
        )
        torch.testing.assert_close(native, reference, rtol=0, atol=0)

        wrapper_error = None
        try:
            apply_sana_cross_attention_output_layout(output)
        except ValueError as exc:
            wrapper_error = str(exc)
        else:
            raise RuntimeError("R22 wrapper unexpectedly accepted the runtime layout")

        payload = {
            "schema_version": 1,
            "status": "rejected",
            "gpu": gpu,
            "lease_uuid": lease.gpu_uuid,
            "torch": torch.__version__,
            "formal_contract": {
                "query_shape": list(query.shape),
                "key_value_shape": list(key.shape),
                "dtype": str(query.dtype),
            },
            "layout": {
                "query_stride": list(query.stride()),
                "query_contiguous": query.is_contiguous(),
                "sdpa_output_shape": list(output.shape),
                "sdpa_output_stride": list(output.stride()),
                "sdpa_output_contiguous": output.is_contiguous(),
                "native_dense_shape": list(native.shape),
                "native_dense_stride": list(native.stride()),
                "native_dense_contiguous": native.is_contiguous(),
                "native_shares_storage_with_sdpa_output": (
                    native.untyped_storage().data_ptr()
                    == output.untyped_storage().data_ptr()
                ),
                "native_data_ptr_equals_sdpa_output": (
                    native.data_ptr() == output.data_ptr()
                ),
                "native_exact": bool(torch.equal(native, reference)),
            },
            "candidate": {
                "wrapper_rejected_runtime_layout": wrapper_error is not None,
                "wrapper_error": wrapper_error,
            },
            "decision": (
                "Reject R22: on the model-matched SDPA input layout, the output "
                "already has BSHD-compatible backing storage. The native "
                "transpose+reshape is a zero-copy view, so the design probe's "
                "contiguous-BHSD copy assumption does not apply."
            ),
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
