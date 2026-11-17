"""
test_inference_cache.py — Correctness tests for the factor-caching inference optimisation.

Checks:
  1. Uncached and "all"-cached forward passes produce bit-identical logits.
  2. Uncached and "layerwise"-cached forward passes produce bit-identical logits.
  3. cache_is_valid() reflects actual cache state.
  4. clear_inference_cache() removes all cached tensors.
  5. factor_cache_bytes() is positive after prepare_for_inference("all").
  6. factor_cache_bytes() is zero after clear_inference_cache().
  7. model.train() auto-clears the cache.
  8. prepare_for_inference() on a training model raises RuntimeError.
  9. Calling prepare_for_inference("all") twice is safe (idempotent).
 10. Calling clear_inference_cache() on a non-latent model is a no-op.
 11. Layerwise cache is empty between blocks (not held in memory across blocks).
 12. generate() with "all" caching produces bit-identical tokens to uncached.
 13. generate() with "layerwise" caching produces bit-identical tokens to uncached.
"""

from __future__ import annotations

import sys
from pathlib import Path
import torch

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from latent_transformer_v0 import ModelConfig, TinyLM
from inference_cache import (
    prepare_for_inference,
    clear_inference_cache,
    cache_is_valid,
    factor_cache_bytes,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")
DTYPE  = torch.float32
SEED   = 42


def _make_model(mode: str = "latent", n_layers: int = 2,
                d_model: int = 64, d_ff: int = 128,
                n_heads: int = 2, rank: int = 16, latent_dim: int = 8) -> TinyLM:
    torch.manual_seed(SEED)
    cfg = ModelConfig(
        vocab_size=256, context=64, n_layers=n_layers,
        d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        dropout=0.0, mode=mode,
        rank=rank, latent_dim=latent_dim,
        constructor_hidden=64, constructor_type_dim=8, constructor_fourier_bands=4,
    )
    return TinyLM(cfg).to(device=DEVICE, dtype=DTYPE)


def _random_input(model: TinyLM, batch: int = 2, seq: int = 16) -> torch.Tensor:
    torch.manual_seed(SEED + 1)
    return torch.randint(0, model.cfg.vocab_size, (batch, seq), device=DEVICE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_01_all_cache_logits_identical():
    """Cached forward must produce bit-identical logits to uncached."""
    model = _make_model()
    idx   = _random_input(model)

    model.eval()
    with torch.no_grad():
        logits_ref, _ = model(idx)

    prepare_for_inference(model, strategy="all")
    with torch.no_grad():
        logits_cached, _ = model(idx)

    assert torch.equal(logits_ref, logits_cached), (
        f"Logit mismatch (all): max_diff={( logits_ref - logits_cached).abs().max():.2e}"
    )
    clear_inference_cache(model)
    print("PASS test_01_all_cache_logits_identical")


def test_02_layerwise_cache_logits_identical():
    """Layerwise-cached forward must produce bit-identical logits to uncached."""
    model = _make_model()
    idx   = _random_input(model)

    model.eval()
    with torch.no_grad():
        logits_ref, _ = model(idx)

    prepare_for_inference(model, strategy="layerwise")
    with torch.no_grad():
        logits_cached, _ = model(idx)

    assert torch.equal(logits_ref, logits_cached), (
        f"Logit mismatch (layerwise): max_diff={(logits_ref - logits_cached).abs().max():.2e}"
    )
    clear_inference_cache(model)
    print("PASS test_02_layerwise_cache_logits_identical")


def test_03_cache_is_valid_reflects_state():
    """cache_is_valid() should be False before prepare, True after."""
    model = _make_model()
    model.eval()
    assert not cache_is_valid(model), "Should not be valid before prepare"
    prepare_for_inference(model, strategy="all")
    assert cache_is_valid(model), "Should be valid after prepare(all)"
    clear_inference_cache(model)
    assert not cache_is_valid(model), "Should be invalid after clear"
    print("PASS test_03_cache_is_valid_reflects_state")


def test_04_clear_removes_all_caches():
    """After clear_inference_cache, no LatentLinear should hold a cache."""
    from inference_cache import _iter_latent_linears, _CACHE_ATTR
    model = _make_model()
    model.eval()
    prepare_for_inference(model, strategy="all")
    clear_inference_cache(model)
    for ll in _iter_latent_linears(model):
        assert getattr(ll, _CACHE_ATTR, None) is None, "Cache not cleared"
    print("PASS test_04_clear_removes_all_caches")


def test_05_factor_cache_bytes_positive_after_prepare():
    """factor_cache_bytes() should be > 0 after prepare(all)."""
    model = _make_model()
    model.eval()
    prepare_for_inference(model, strategy="all")
    nbytes = factor_cache_bytes(model)
    assert nbytes > 0, f"Expected > 0 bytes, got {nbytes}"
    clear_inference_cache(model)
    print(f"PASS test_05_factor_cache_bytes_positive_after_prepare  ({nbytes:,} bytes)")


def test_06_factor_cache_bytes_zero_after_clear():
    """factor_cache_bytes() should be 0 after clear."""
    model = _make_model()
    model.eval()
    prepare_for_inference(model, strategy="all")
    clear_inference_cache(model)
    assert factor_cache_bytes(model) == 0
    print("PASS test_06_factor_cache_bytes_zero_after_clear")


def test_07_train_autoclears_cache():
    """Calling model.train() must automatically clear the cache."""
    model = _make_model()
    model.eval()
    prepare_for_inference(model, strategy="all")
    assert cache_is_valid(model)
    model.train()   # should trigger auto-clear
    assert not cache_is_valid(model), "train() did not clear the cache"
    print("PASS test_07_train_autoclears_cache")


def test_08_prepare_on_training_model_raises():
    """prepare_for_inference() on a training-mode model must raise RuntimeError."""
    model = _make_model()
    assert model.training, "Model should start in training mode"
    try:
        prepare_for_inference(model, strategy="all")
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass
    print("PASS test_08_prepare_on_training_model_raises")


def test_09_double_prepare_is_safe():
    """Calling prepare_for_inference twice must not corrupt the cache."""
    model = _make_model()
    idx   = _random_input(model)

    model.eval()
    with torch.no_grad():
        logits_ref, _ = model(idx)

    prepare_for_inference(model, strategy="all")
    prepare_for_inference(model, strategy="all")  # second call
    with torch.no_grad():
        logits_cached, _ = model(idx)

    assert torch.equal(logits_ref, logits_cached), "Double-prepare corrupted results"
    clear_inference_cache(model)
    print("PASS test_09_double_prepare_is_safe")


def test_10_clear_on_non_latent_is_noop():
    """clear_inference_cache() on a dense or lowrank model must not raise."""
    for mode in ("dense", "lowrank"):
        model = _make_model(mode=mode)
        model.eval()
        clear_inference_cache(model)   # must not raise
    print("PASS test_10_clear_on_non_latent_is_noop")


def test_11_layerwise_cache_empty_between_blocks():
    """
    After the full forward pass with layerwise caching, the cache should be
    empty (factors released by post-hooks).
    """
    model = _make_model()
    idx   = _random_input(model)

    model.eval()
    prepare_for_inference(model, strategy="layerwise")
    with torch.no_grad():
        model(idx)

    # After the forward pass all block post-hooks should have cleared the cache
    assert factor_cache_bytes(model) == 0, (
        f"Layerwise cache not empty after forward: {factor_cache_bytes(model):,} bytes"
    )
    clear_inference_cache(model)
    print("PASS test_11_layerwise_cache_empty_between_blocks")


def test_12_generate_all_cache_identical():
    """generate() with all-caching must produce identical token sequences."""
    model = _make_model()
    idx   = _random_input(model, batch=1, seq=8)

    model.eval()
    # Uncached generation (generate() calls eval() internally)
    with torch.no_grad():
        tokens_ref = model.generate(idx, max_new_tokens=16)[0].tolist()

    prepare_for_inference(model, strategy="all")
    with torch.no_grad():
        tokens_cached = model.generate(idx, max_new_tokens=16)[0].tolist()

    # generate() calls model.train(was_training) at the end, which clears the cache.
    # Re-prepare to verify we can still use it.
    assert tokens_ref == tokens_cached, (
        f"Token mismatch (all)\n  ref:    {tokens_ref}\n  cached: {tokens_cached}"
    )
    print("PASS test_12_generate_all_cache_identical")


def test_13_generate_layerwise_cache_identical():
    """generate() with layerwise-caching must produce identical token sequences."""
    model = _make_model()
    idx   = _random_input(model, batch=1, seq=8)

    model.eval()
    with torch.no_grad():
        tokens_ref = model.generate(idx, max_new_tokens=16)[0].tolist()

    prepare_for_inference(model, strategy="layerwise")
    with torch.no_grad():
        tokens_cached = model.generate(idx, max_new_tokens=16)[0].tolist()

    assert tokens_ref == tokens_cached, (
        f"Token mismatch (layerwise)\n  ref:    {tokens_ref}\n  cached: {tokens_cached}"
    )
    print("PASS test_13_generate_layerwise_cache_identical")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    tests = [
        test_01_all_cache_logits_identical,
        test_02_layerwise_cache_logits_identical,
        test_03_cache_is_valid_reflects_state,
        test_04_clear_removes_all_caches,
        test_05_factor_cache_bytes_positive_after_prepare,
        test_06_factor_cache_bytes_zero_after_clear,
        test_07_train_autoclears_cache,
        test_08_prepare_on_training_model_raises,
        test_09_double_prepare_is_safe,
        test_10_clear_on_non_latent_is_noop,
        test_11_layerwise_cache_empty_between_blocks,
        test_12_generate_all_cache_identical,
        test_13_generate_layerwise_cache_identical,
    ]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
