from __future__ import annotations

import jax.numpy as jnp

from .config import FeatureConfig


def apply_feature_expansion(x: jnp.ndarray, cfg: FeatureConfig) -> jnp.ndarray:
    x = cfg.input_scale * x
    feats = []
    if cfg.include_identity or cfg.kind == 'identity':
        feats.append(x)

    if cfg.kind in ('sin', 'sincos'):
        for k in range(1, cfg.num_frequencies + 1):
            feats.append(jnp.sin(k * x))
    if cfg.kind in ('cos', 'sincos'):
        for k in range(1, cfg.num_frequencies + 1):
            feats.append(jnp.cos(k * x))
    if cfg.kind == 'exp':
        for k in range(1, cfg.num_frequencies + 1):
            feats.append(jnp.exp(k * x))

    if not feats:
        feats = [x]
    return jnp.concatenate(feats, axis=-1)
