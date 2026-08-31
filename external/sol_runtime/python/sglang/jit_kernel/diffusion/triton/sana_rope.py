# SPDX-License-Identifier: Apache-2.0
"""SANA-Video-specific paired out-of-place rotary embedding kernel."""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore


@triton.jit
def _sana_paired_rotary_emb_kernel(
    q_out_ptr,
    k_out_ptr,
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    total_pairs,
    num_tokens: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    frequency_batch: tl.constexpr,
    stride_cos_b: tl.constexpr,
    stride_cos_t: tl.constexpr,
    stride_cos_d: tl.constexpr,
    stride_sin_b: tl.constexpr,
    stride_sin_t: tl.constexpr,
    stride_sin_d: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_pairs

    pair = offsets % half_dim
    row = offsets // half_dim
    head = row % num_heads
    batch_token = row // num_heads
    token = batch_token % num_tokens
    batch = batch_token // num_tokens

    element = ((batch_token * num_heads + head) * head_dim) + 2 * pair
    frequency_batch_index = batch if frequency_batch > 1 else 0
    cos_offset = (
        frequency_batch_index * stride_cos_b
        + token * stride_cos_t
        + (2 * pair) * stride_cos_d
    )
    sin_offset = (
        frequency_batch_index * stride_sin_b
        + token * stride_sin_t
        + (2 * pair + 1) * stride_sin_d
    )

    cosine = tl.load(cos_ptr + cos_offset, mask=mask, other=0.0).to(tl.float32)
    sine = tl.load(sin_ptr + sin_offset, mask=mask, other=0.0).to(tl.float32)
    q_even = tl.load(q_ptr + element, mask=mask, other=0.0).to(tl.float32)
    q_odd = tl.load(q_ptr + element + 1, mask=mask, other=0.0).to(tl.float32)
    k_even = tl.load(k_ptr + element, mask=mask, other=0.0).to(tl.float32)
    k_odd = tl.load(k_ptr + element + 1, mask=mask, other=0.0).to(tl.float32)

    tl.store(q_out_ptr + element, q_even * cosine - q_odd * sine, mask=mask)
    tl.store(q_out_ptr + element + 1, q_even * sine + q_odd * cosine, mask=mask)
    tl.store(k_out_ptr + element, k_even * cosine - k_odd * sine, mask=mask)
    tl.store(k_out_ptr + element + 1, k_even * sine + k_odd * cosine, mask=mask)


@triton.jit
def _sana_fused_qknorm_relu_rotary_emb_kernel(
    q_relu_out_ptr,
    k_relu_out_ptr,
    q_rotate_out_ptr,
    k_rotate_out_ptr,
    q_ptr,
    k_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_ptr,
    sin_ptr,
    num_rows,
    inner_dim: tl.constexpr,
    num_tokens: tl.constexpr,
    head_dim: tl.constexpr,
    stride_q_b: tl.constexpr,
    stride_q_t: tl.constexpr,
    stride_k_b: tl.constexpr,
    stride_k_t: tl.constexpr,
    frequency_batch: tl.constexpr,
    stride_cos_b: tl.constexpr,
    stride_cos_t: tl.constexpr,
    stride_cos_d: tl.constexpr,
    stride_sin_b: tl.constexpr,
    stride_sin_t: tl.constexpr,
    stride_sin_d: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
):
    row = tl.program_id(0)
    pairs = tl.arange(0, BLOCK_PAIRS)
    even = 2 * pairs
    odd = even + 1
    mask = (row < num_rows) & (odd < inner_dim)
    token = row % num_tokens
    batch = row // num_tokens
    q_base = batch * stride_q_b + token * stride_q_t
    k_base = batch * stride_k_b + token * stride_k_t

    q_even = tl.load(q_ptr + q_base + even, mask=mask, other=0.0).to(tl.float32)
    q_odd = tl.load(q_ptr + q_base + odd, mask=mask, other=0.0).to(tl.float32)
    k_even = tl.load(k_ptr + k_base + even, mask=mask, other=0.0).to(tl.float32)
    k_odd = tl.load(k_ptr + k_base + odd, mask=mask, other=0.0).to(tl.float32)
    q_variance = tl.sum(q_even * q_even + q_odd * q_odd, axis=0) / inner_dim
    k_variance = tl.sum(k_even * k_even + k_odd * k_odd, axis=0) / inner_dim
    q_rstd = tl.rsqrt(q_variance + eps)
    k_rstd = tl.rsqrt(k_variance + eps)
    q_weight_even = tl.load(q_weight_ptr + even, mask=mask, other=0.0).to(tl.float32)
    q_weight_odd = tl.load(q_weight_ptr + odd, mask=mask, other=0.0).to(tl.float32)
    k_weight_even = tl.load(k_weight_ptr + even, mask=mask, other=0.0).to(tl.float32)
    k_weight_odd = tl.load(k_weight_ptr + odd, mask=mask, other=0.0).to(tl.float32)

    q_relu_even = tl.maximum(q_even * q_rstd * q_weight_even, 0.0).to(tl.bfloat16)
    q_relu_odd = tl.maximum(q_odd * q_rstd * q_weight_odd, 0.0).to(tl.bfloat16)
    k_relu_even = tl.maximum(k_even * k_rstd * k_weight_even, 0.0).to(tl.bfloat16)
    k_relu_odd = tl.maximum(k_odd * k_rstd * k_weight_odd, 0.0).to(tl.bfloat16)

    out_base = row * inner_dim
    tl.store(q_relu_out_ptr + out_base + even, q_relu_even, mask=mask)
    tl.store(q_relu_out_ptr + out_base + odd, q_relu_odd, mask=mask)
    tl.store(k_relu_out_ptr + out_base + even, k_relu_even, mask=mask)
    tl.store(k_relu_out_ptr + out_base + odd, k_relu_odd, mask=mask)

    frequency_batch_index = batch if frequency_batch > 1 else 0
    frequency_dim = even % head_dim
    cos_offset = (
        frequency_batch_index * stride_cos_b
        + token * stride_cos_t
        + frequency_dim * stride_cos_d
    )
    sin_offset = (
        frequency_batch_index * stride_sin_b
        + token * stride_sin_t
        + (frequency_dim + 1) * stride_sin_d
    )
    cosine = tl.load(cos_ptr + cos_offset, mask=mask, other=0.0).to(tl.float32)
    sine = tl.load(sin_ptr + sin_offset, mask=mask, other=0.0).to(tl.float32)
    q_relu_even_fp32 = q_relu_even.to(tl.float32)
    q_relu_odd_fp32 = q_relu_odd.to(tl.float32)
    k_relu_even_fp32 = k_relu_even.to(tl.float32)
    k_relu_odd_fp32 = k_relu_odd.to(tl.float32)
    tl.store(
        q_rotate_out_ptr + out_base + even,
        q_relu_even_fp32 * cosine - q_relu_odd_fp32 * sine,
        mask=mask,
    )
    tl.store(
        q_rotate_out_ptr + out_base + odd,
        q_relu_even_fp32 * sine + q_relu_odd_fp32 * cosine,
        mask=mask,
    )
    tl.store(
        k_rotate_out_ptr + out_base + even,
        k_relu_even_fp32 * cosine - k_relu_odd_fp32 * sine,
        mask=mask,
    )
    tl.store(
        k_rotate_out_ptr + out_base + odd,
        k_relu_even_fp32 * sine + k_relu_odd_fp32 * cosine,
        mask=mask,
    )


def apply_sana_paired_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate SANA Q and K together while preserving their non-rotated inputs.

    The specialized path intentionally fails closed outside the proven SANA-Video
    layout. The caller's disabled path remains the native PyTorch implementation.
    """

    if query.device.type != "cuda":
        raise ValueError("paired SANA RoPE requires CUDA tensors")
    if query.dtype is not torch.bfloat16 or key.dtype is not torch.bfloat16:
        raise ValueError("paired SANA RoPE is specialized for BF16 Q/K")
    if query.ndim != 4 or key.shape != query.shape:
        raise ValueError("paired SANA RoPE expects matching [B, N, H, D] Q/K")
    if query.device != key.device or not query.is_contiguous() or not key.is_contiguous():
        raise ValueError("paired SANA RoPE requires contiguous same-device Q/K")

    batch, num_tokens, num_heads, head_dim = query.shape
    if head_dim % 2:
        raise ValueError("paired SANA RoPE requires an even head dimension")
    expected_frequency_shape = (num_tokens, 1, head_dim)
    for name, frequency in (("cos", freqs_cos), ("sin", freqs_sin)):
        if frequency.ndim != 4 or tuple(frequency.shape[1:]) != expected_frequency_shape:
            raise ValueError(
                f"paired SANA RoPE {name} must have shape [1|B, N, 1, D]"
            )
        if frequency.shape[0] not in (1, batch):
            raise ValueError(f"paired SANA RoPE {name} batch must be 1 or Q/K batch")
        if frequency.dtype is not torch.float32 or frequency.device != query.device:
            raise ValueError(f"paired SANA RoPE {name} must be FP32 on the Q/K device")
        if frequency.stride(-1) != 1:
            raise ValueError(f"paired SANA RoPE {name} last dimension must be contiguous")
    if freqs_cos.shape != freqs_sin.shape:
        raise ValueError("paired SANA RoPE cosine and sine shapes must match")

    q_out = torch.empty_like(query)
    k_out = torch.empty_like(key)
    total_pairs = query.numel() // 2
    half_dim = head_dim // 2
    grid = (triton.cdiv(total_pairs, 1024),)
    _sana_paired_rotary_emb_kernel[grid](
        q_out,
        k_out,
        query,
        key,
        freqs_cos,
        freqs_sin,
        total_pairs,
        num_tokens=num_tokens,
        num_heads=num_heads,
        head_dim=head_dim,
        half_dim=half_dim,
        frequency_batch=freqs_cos.shape[0],
        stride_cos_b=freqs_cos.stride(0),
        stride_cos_t=freqs_cos.stride(1),
        stride_cos_d=freqs_cos.stride(3),
        stride_sin_b=freqs_sin.stride(0),
        stride_sin_t=freqs_sin.stride(1),
        stride_sin_d=freqs_sin.stride(3),
        BLOCK=1024,
        num_warps=8,
    )
    return q_out, k_out


def apply_sana_fused_qk_norm_relu_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse the two learned Q/K RMSNorms, ReLUs, and paired SANA RoPE."""

    if query.device.type != "cuda":
        raise ValueError("fused SANA Q/K norm-RoPE requires CUDA tensors")
    if query.dtype is not torch.bfloat16 or key.dtype is not torch.bfloat16:
        raise ValueError("fused SANA Q/K norm-RoPE is specialized for BF16 Q/K")
    if query.ndim != 3 or key.shape != query.shape:
        raise ValueError("fused SANA Q/K norm-RoPE expects matching [B, N, C] Q/K")
    if query.device != key.device or query.stride(-1) != 1 or key.stride(-1) != 1:
        raise ValueError("fused SANA Q/K norm-RoPE requires last-contiguous Q/K")

    batch, num_tokens, inner_dim = query.shape
    for name, weight in (("query", query_weight), ("key", key_weight)):
        if weight.shape != (inner_dim,) or not weight.is_contiguous():
            raise ValueError(f"fused SANA {name} weight must be contiguous [C]")
        if weight.device != query.device or weight.dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError(f"fused SANA {name} weight must be BF16/FP32 on Q device")
    if not isinstance(eps, float) or eps <= 0:
        raise ValueError("fused SANA Q/K norm-RoPE eps must be a positive float")

    if freqs_cos.ndim != 4:
        raise ValueError("fused SANA Q/K norm-RoPE expects [1|B, N, 1, D] frequencies")
    head_dim = freqs_cos.shape[-1]
    expected_frequency_shape = (num_tokens, 1, head_dim)
    for name, frequency in (("cos", freqs_cos), ("sin", freqs_sin)):
        if frequency.ndim != 4 or tuple(frequency.shape[1:]) != expected_frequency_shape:
            raise ValueError(
                f"fused SANA Q/K norm-RoPE {name} must be [1|B, N, 1, D]"
            )
        if frequency.shape[0] not in (1, batch):
            raise ValueError(f"fused SANA Q/K norm-RoPE {name} batch must be 1 or B")
        if frequency.dtype is not torch.float32 or frequency.device != query.device:
            raise ValueError(f"fused SANA Q/K norm-RoPE {name} must be FP32 on Q device")
        if frequency.stride(-1) != 1:
            raise ValueError(f"fused SANA Q/K norm-RoPE {name} must be last-contiguous")
    if freqs_cos.shape != freqs_sin.shape or head_dim % 2 or inner_dim % head_dim:
        raise ValueError("fused SANA Q/K norm-RoPE has incompatible head/frequency shape")

    num_heads = inner_dim // head_dim
    outputs = [
        torch.empty(
            batch,
            num_tokens,
            num_heads,
            head_dim,
            device=query.device,
            dtype=query.dtype,
        )
        for _ in range(4)
    ]
    _sana_fused_qknorm_relu_rotary_emb_kernel[(batch * num_tokens,)](
        *outputs,
        query,
        key,
        query_weight,
        key_weight,
        freqs_cos,
        freqs_sin,
        batch * num_tokens,
        inner_dim=inner_dim,
        num_tokens=num_tokens,
        head_dim=head_dim,
        stride_q_b=query.stride(0),
        stride_q_t=query.stride(1),
        stride_k_b=key.stride(0),
        stride_k_t=key.stride(1),
        frequency_batch=freqs_cos.shape[0],
        stride_cos_b=freqs_cos.stride(0),
        stride_cos_t=freqs_cos.stride(1),
        stride_cos_d=freqs_cos.stride(3),
        stride_sin_b=freqs_sin.stride(0),
        stride_sin_t=freqs_sin.stride(1),
        stride_sin_d=freqs_sin.stride(3),
        eps=eps,
        BLOCK_PAIRS=triton.next_power_of_2(inner_dim // 2),
        num_warps=8,
    )
    return tuple(outputs)
