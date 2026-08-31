#!/usr/bin/env python3
"""CPU source/lifecycle tests for the exact SANA cross-attention KV cache."""

import gc
import os
import weakref

import torch

from sglang.multimodal_gen.runtime.models.dits.sana_video import (
    SanaVideoCrossAttention,
    _BoundedTensorIdentityCache,
    _module_parameter_signature,
)


def main():
    torch.manual_seed(7)

    # The cache holds a strong source reference, covers source mutation, and is
    # bounded to the two live CFG contexts across warmup -> formal replacement.
    cache = _BoundedTensorIdentityCache(capacity=2)
    weight_module = torch.nn.Linear(4, 4, bias=False)
    weights = _module_parameter_signature(weight_module)
    warm = torch.randn(1, 3, 4)
    warm_ref = weakref.ref(warm)
    cache.store(warm, weights, "warm")
    assert cache.lookup(warm, weights) == "warm"
    warm.add_(1)
    assert cache.lookup(warm, weights) is None, "source version mutation must miss"
    cache.store(warm, weights, "warm-v2")
    del warm
    gc.collect()
    assert warm_ref() is not None, "live entry must strongly retain its source"

    cond = torch.randn(1, 3, 4)
    uncond = torch.randn(1, 3, 4)
    cache.store(cond, weights, "cond")
    cache.store(uncond, weights, "uncond")
    gc.collect()
    assert len(cache._entries) == 2
    assert cache.lookup(cond, weights) == "cond"
    assert cache.lookup(uncond, weights) == "uncond"
    assert warm_ref() is None, "warmup source must be released after bounded eviction"

    # Parameter lifecycle/version changes invalidate an otherwise identical hit.
    signature_before = _module_parameter_signature(weight_module)
    with torch.no_grad():
        weight_module.weight.add_(0.25)
    signature_after = _module_parameter_signature(weight_module)
    assert signature_before != signature_after
    assert cache.lookup(cond, signature_after) is None

    # The model-level caption projection uses the same cache contract before
    # per-layer K/V projection. Simulate that exact project->norm value path.
    caption_projection = torch.nn.Linear(6, 8)
    caption_norm = torch.nn.LayerNorm(8)
    caption_cache = _BoundedTensorIdentityCache(capacity=2)
    caption_calls = 0

    def cached_caption(source):
        nonlocal caption_calls
        signature = _module_parameter_signature(caption_projection, caption_norm)
        cached = caption_cache.lookup(source, signature)
        if cached is not None:
            return cached
        caption_calls += 1
        value = caption_norm(caption_projection(source))
        caption_cache.store(source, signature, value)
        return value

    warm_caption = torch.randn(1, 3, 6)
    cond_caption = torch.randn(1, 3, 6)
    uncond_caption = torch.randn(1, 3, 6)
    with torch.no_grad():
        for source in (
            warm_caption,
            warm_caption,
            cond_caption,
            uncond_caption,
            cond_caption,
            uncond_caption,
        ):
            expected = caption_norm(caption_projection(source))
            actual = cached_caption(source)
            assert torch.equal(expected, actual)
    assert caption_calls == 3
    assert {id(entry[0]) for entry in caption_cache._entries} == {
        id(cond_caption),
        id(uncond_caption),
    }

    # Real module test: OFF recomputes every call; ON computes K/V once for each
    # of two context objects, produces the same output, and evicts warmup.
    os.environ["SGLANG_SANA_XATTN_KV_CACHE"] = "0"
    reference = SanaVideoCrossAttention(
        query_dim=8, cross_attention_dim=6, num_heads=2, head_dim=4, qk_norm=False
    )
    os.environ["SGLANG_SANA_XATTN_KV_CACHE"] = "1"
    optimized = SanaVideoCrossAttention(
        query_dim=8, cross_attention_dim=6, num_heads=2, head_dim=4, qk_norm=False
    )
    optimized.load_state_dict(reference.state_dict())
    assert reference._kv_cache is None
    assert optimized._kv_cache is not None

    ref_k_calls = 0
    opt_k_calls = 0

    def count_ref(_module, _inputs, _output):
        nonlocal ref_k_calls
        ref_k_calls += 1

    def count_opt(_module, _inputs, _output):
        nonlocal opt_k_calls
        opt_k_calls += 1

    reference.to_k.register_forward_hook(count_ref)
    optimized.to_k.register_forward_hook(count_opt)
    hidden = torch.randn(1, 5, 8)
    warm_ctx = torch.randn(1, 3, 6)
    cond_ctx = torch.randn(1, 3, 6)
    uncond_ctx = torch.randn(1, 3, 6)

    with torch.no_grad():
        for context in (warm_ctx, warm_ctx, cond_ctx, uncond_ctx, cond_ctx, uncond_ctx):
            expected = reference(hidden, context)
            actual = optimized(hidden, context)
            assert torch.equal(expected, actual)

    assert ref_k_calls == 6
    assert opt_k_calls == 3, "warmup, cond, and uncond should each miss exactly once"
    assert len(optimized._kv_cache._entries) == 2
    assert {id(entry[0]) for entry in optimized._kv_cache._entries} == {
        id(cond_ctx),
        id(uncond_ctx),
    }

    # Mutating K weights invalidates both cached branch entries and recomputes.
    with torch.no_grad():
        reference.to_k.weight.add_(0.125)
        optimized.to_k.weight.add_(0.125)
        expected = reference(hidden, cond_ctx)
        actual = optimized(hidden, cond_ctx)
    assert torch.equal(expected, actual)
    assert opt_k_calls == 4

    print(
        {
            "status": "passed",
            "off_k_calls": ref_k_calls,
            "cached_k_calls": opt_k_calls,
            "bounded_entries": len(optimized._kv_cache._entries),
            "caption_projection_calls": caption_calls,
            "warmup_evicted": id(warm_ctx)
            not in {id(entry[0]) for entry in optimized._kv_cache._entries},
            "weight_mutation_invalidated": True,
        }
    )


if __name__ == "__main__":
    main()
