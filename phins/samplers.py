from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp


def _as_bounds(domain, input_dim: int):
    """Convert a 1D or multi-D domain into lower/upper arrays."""
    if domain is None:
        raise ValueError("Random collocation sampling requires a domain or data to infer one.")

    if input_dim == 1 and len(domain) == 2 and not isinstance(domain[0], (tuple, list)):
        return jnp.array([float(domain[0])]), jnp.array([float(domain[1])])

    if len(domain) != input_dim:
        raise ValueError(
            f"Expected {input_dim} domain intervals, got {len(domain)}. "
            "Use (min, max) for 1D or ((min, max), ...) for multi-D."
        )

    lo = jnp.array([float(d[0]) for d in domain])
    hi = jnp.array([float(d[1]) for d in domain])
    return lo, hi


def infer_domain_from_points(points: Optional[jnp.ndarray], input_dim: int):
    """Infer rectangular bounds from existing points."""
    if points is None:
        return None
    points = jnp.asarray(points)
    if points.ndim != 2 or points.shape[1] != input_dim:
        return None
    lo = jnp.min(points, axis=0)
    hi = jnp.max(points, axis=0)
    if input_dim == 1:
        return (float(lo[0]), float(hi[0]))
    return tuple((float(lo[i]), float(hi[i])) for i in range(input_dim))


def sample_uniform_collocation(key, domain, n_points: int, input_dim: int = 1) -> jnp.ndarray:
    """Sample uniform random collocation points over a rectangular domain."""
    lo, hi = _as_bounds(domain, input_dim)
    u = jax.random.uniform(key, shape=(int(n_points), int(input_dim)))
    return lo + (hi - lo) * u
