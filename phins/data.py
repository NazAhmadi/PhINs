from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import jax.numpy as jnp


@dataclass
class PINNDataBundle:
    t_data: jnp.ndarray
    y_data: Optional[jnp.ndarray] = None
    t_ic: Optional[jnp.ndarray] = None
    y_ic: Optional[jnp.ndarray] = None
    t_collocation: Optional[jnp.ndarray] = None
    metadata: Dict[str, object] = field(default_factory=dict)
