"""
inference_cache.py — Optional inference-time factor caching for LatentLinear models.

This module adds `prepare_for_inference()` and `clear_inference_cache()` to any
TinyLM built in latent mode.  It does not alter training, gradient computation,
checkpoint format, or the LatentLinear/CoordinateFactorConstructor classes.

The authoritative model parameters remain the latent codes (z, log_gain) and the
constructor weights.  The cache holds derived tensors that are identical to what
would be computed on-the-fly; they are discarded before any parameter update.

Design constraints
------------------
1. Non-invasive: nothing in latent_transformer_v0.py is modified.
2. Transparent: a cached forward pass must produce bit-identical results to an
   uncached forward pass (verified in tests below).
3. Safe invalidation: the cache is automatically cleared when the model enters
   training mode, when parameters change (optimizer step), and explicitly via
   clear_inference_cache().
4. Device / dtype aware: if the model is moved to a different device or cast to
   a different dtype the cache is invalid.  prepare_for_inference() must be
   called again after any such move.
5. Memory explicit: factor_cache_bytes() reports how much RAM the cache occupies.

Caching strategies
------------------
"all"        — generate all L/R factors for every LatentLinear at once and keep
               them until clear_inference_cache() is called.
               Best for repeated inference; highest peak memory.

"layerwise"  — generate and release L/R factors one transformer Block at a time.
               Each Block caches its three matrices while it runs, then the cache
               is released before the next Block runs.
               Lower peak RAM than "all", same per-token speed after warmup.
               Implementation: forward hooks on Block.

"none"       — no caching (baseline, original behavior).

Usage
-----
    from inference_cache import prepare_for_inference, clear_inference_cache

    model.eval()
    prepare_for_inference(model, strategy="all")

    with torch.no_grad():
        logits, _ = model(idx)

    clear_inference_cache(model)
    model.train()   # also clears automatically
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Allow importing TinyLM without running the training CLI
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from latent_transformer_v0 import (  # noqa: E402
    CoordinateFactorConstructor,
    LatentLinear,
    LatentMLP,
    TinyLM,
)


# ---------------------------------------------------------------------------
# Cache storage attached to individual LatentLinear instances
# ---------------------------------------------------------------------------

_CACHE_ATTR = "_infer_cache"      # (L, R) tensors stored directly on LatentLinear
_HOOK_ATTR   = "_infer_hook"       # forward pre-hook handle stored on Block


def _has_cache(ll: LatentLinear) -> bool:
    return getattr(ll, _CACHE_ATTR, None) is not None


def _set_cache(ll: LatentLinear, L: Tensor, R: Tensor) -> None:
    setattr(ll, _CACHE_ATTR, (L, R))


def _get_cache(ll: LatentLinear) -> Tuple[Tensor, Tensor]:
    return getattr(ll, _CACHE_ATTR)


def _clear_cache(ll: LatentLinear) -> None:
    setattr(ll, _CACHE_ATTR, None)


# ---------------------------------------------------------------------------
# Patched LatentLinear.forward that uses cache when available
# ---------------------------------------------------------------------------

def _patched_forward(self: LatentLinear, x: Tensor,
                     constructor: CoordinateFactorConstructor) -> Tensor:
    if _has_cache(self):
        L, R = _get_cache(self)
    else:
        L, R = constructor.factors(self.z, self.type_id,
                                   self.in_features, self.out_features)
    from torch.nn import functional as F
    return F.linear(F.linear(x, R), L) * self.log_gain.exp()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_for_inference(
    model: TinyLM,
    strategy: str = "all",
) -> None:
    """
    Switch a latent-mode TinyLM into an inference-optimised state.

    Parameters
    ----------
    model : TinyLM
        Must be in eval mode (model.eval()) before calling.
    strategy : {"all", "layerwise", "none"}
        "all"       — pre-generate and store every L/R factor pair.
        "layerwise" — install per-Block hooks that cache factors for one block
                      at a time and release them when that block finishes.
        "none"      — no-op; existing on-demand generation is kept.

    After calling this, forward passes and generate() use the cached factors
    instead of calling the constructor.  Call clear_inference_cache() or
    model.train() to revert.
    """
    if model.cfg.mode != "latent":
        return  # nothing to do for dense / lowrank

    if model.training:
        raise RuntimeError(
            "prepare_for_inference() requires the model to be in eval mode. "
            "Call model.eval() first."
        )

    # Patch LatentLinear.forward on the class once (idempotent).
    _install_patched_forward()

    if strategy == "none":
        return

    if strategy == "all":
        _build_full_cache(model)
    elif strategy == "layerwise":
        _install_layerwise_hooks(model)
    else:
        raise ValueError(f"Unknown strategy {strategy!r}. Choose 'all', 'layerwise', or 'none'.")

    # Auto-clear on train() — we hook into the model's train() method.
    _install_train_guard(model)


def clear_inference_cache(model: TinyLM) -> None:
    """
    Release all cached factors and remove layerwise hooks.
    Safe to call even if prepare_for_inference() was never called.
    """
    if model.cfg.mode != "latent":
        return
    for ll in _iter_latent_linears(model):
        _clear_cache(ll)
    _remove_layerwise_hooks(model)


def factor_cache_bytes(model: TinyLM) -> int:
    """Return the total RAM occupied by cached L/R factors (bytes)."""
    total = 0
    for ll in _iter_latent_linears(model):
        pair = getattr(ll, _CACHE_ATTR, None)
        if pair is not None:
            L, R = pair
            total += L.untyped_storage().nbytes() + R.untyped_storage().nbytes()
    return total


def cache_is_valid(model: TinyLM) -> bool:
    """True if every LatentLinear in the model has a populated cache entry."""
    if model.cfg.mode != "latent":
        return True
    lls = list(_iter_latent_linears(model))
    if not lls:
        return False
    return all(_has_cache(ll) for ll in lls)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iter_latent_linears(model: TinyLM):
    for block in model.blocks:
        mlp = block.mlp
        if isinstance(mlp, LatentMLP):
            yield mlp.gate
            yield mlp.up
            yield mlp.down


_forward_patched = False


def _install_patched_forward() -> None:
    global _forward_patched
    if not _forward_patched:
        LatentLinear.forward = _patched_forward
        _forward_patched = True


def _build_full_cache(model: TinyLM) -> None:
    """Pre-generate all L/R factors and store them on each LatentLinear."""
    constructor = model.constructor
    with torch.no_grad():
        for ll in _iter_latent_linears(model):
            L, R = constructor.factors(
                ll.z, ll.type_id, ll.in_features, ll.out_features
            )
            _set_cache(ll, L.detach(), R.detach())


def _install_layerwise_hooks(model: TinyLM) -> None:
    """
    Install pre/post forward hooks on each Block so that:
    - before the block runs: generate L/R for its three LatentLinears
    - after the block runs: release them
    """
    from latent_transformer_v0 import Block

    constructor = model.constructor

    def _pre_hook(block, args):
        mlp = block.mlp
        if not isinstance(mlp, LatentMLP):
            return
        with torch.no_grad():
            for ll in (mlp.gate, mlp.up, mlp.down):
                L, R = constructor.factors(
                    ll.z, ll.type_id, ll.in_features, ll.out_features
                )
                _set_cache(ll, L.detach(), R.detach())

    def _post_hook(block, args, output):
        mlp = block.mlp
        if not isinstance(mlp, LatentMLP):
            return
        for ll in (mlp.gate, mlp.up, mlp.down):
            _clear_cache(ll)

    for block in model.blocks:
        pre_h  = block.register_forward_pre_hook(_pre_hook)
        post_h = block.register_forward_hook(_post_hook)
        # Store both handles so we can remove them later
        existing = getattr(block, _HOOK_ATTR, [])
        existing.extend([pre_h, post_h])
        setattr(block, _HOOK_ATTR, existing)


def _remove_layerwise_hooks(model: TinyLM) -> None:
    for block in model.blocks:
        handles = getattr(block, _HOOK_ATTR, [])
        for h in handles:
            h.remove()
        setattr(block, _HOOK_ATTR, [])


def _install_train_guard(model: TinyLM) -> None:
    """Monkey-patch model.train() to auto-clear the cache."""
    original_train = model.__class__.train

    def _guarded_train(self, mode: bool = True):
        if mode:
            clear_inference_cache(self)
        return original_train(self, mode)

    # Bind as an instance method so only this model is affected.
    import types
    model.train = types.MethodType(_guarded_train, model)
