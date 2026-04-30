from __future__ import annotations

from typing import Dict, Sequence

import jax.nn as jnn
import jax.numpy as jnp

from .config import ParameterSpec


def apply_constraint(x, constraint: str):
    if constraint == 'none':
        return x
    if constraint == 'positive_softplus':
        return jnn.softplus(x)
    if constraint == 'positive_exp':
        return jnp.exp(x)
    raise ValueError(f'Unknown constraint: {constraint}')


def raw_constant_parameters(parameter_specs: Sequence[ParameterSpec]) -> Dict[str, jnp.ndarray]:
    out = {}
    for spec in parameter_specs:
        if spec.mode == 'constant':
            out[spec.name] = jnp.array(spec.init_value)
    return out


def transform_constant_parameters(raw_params: Dict[str, jnp.ndarray], parameter_specs: Sequence[ParameterSpec]):
    out = {}
    for spec in parameter_specs:
        if spec.mode == 'constant':
            out[spec.name] = apply_constraint(raw_params[spec.name], spec.constraint)
    return out


def split_outputs(y, state_names, parameter_specs):
    state_dim = len(state_names)
    n_outputs = y.shape[1]
    
    # Collect and validate all time-varying parameter indices
    tv_indices = {}
    for spec in parameter_specs:
        if spec.mode == 'time_varying':
            if spec.output_index is None:
                raise ValueError(
                    f"output_index must be set for time-varying parameter '{spec.name}'"
                )
            
            idx = spec.output_index
            
            # Validate index is non-negative
            if idx < 0:
                raise ValueError(
                    f"Parameter '{spec.name}': output_index must be non-negative, got {idx}"
                )
            
            # Validate index doesn't conflict with states
            if idx < state_dim:
                raise ValueError(
                    f"Parameter '{spec.name}': output_index {idx} conflicts with states "
                    f"(indices 0-{state_dim-1} are reserved for states {state_names}). "
                    f"Time-varying parameters must use indices >= {state_dim}"
                )
            
            # Validate index is within network output dimension
            if idx >= n_outputs:
                raise ValueError(
                    f"Parameter '{spec.name}': output_index {idx} out of range. "
                    f"Network outputs {n_outputs} values (indices 0-{n_outputs-1}), "
                    f"but you requested index {idx}"
                )
            
            # Validate no duplicate indices
            if idx in tv_indices.values():
                other_name = [name for name, other_idx in tv_indices.items() if other_idx == idx][0]
                raise ValueError(
                    f"Parameter '{spec.name}' and '{other_name}' both use output_index {idx}. "
                    f"Each time-varying parameter must have a unique output_index"
                )
            
            tv_indices[spec.name] = idx
    
    # Extract states and time-varying parameters
    states = {name: y[:, i:i+1] for i, name in enumerate(state_names)}
    time_varying = {
        spec.name: apply_constraint(y[:, spec.output_index:spec.output_index+1], spec.constraint)
        for spec in parameter_specs
        if spec.mode == 'time_varying'
    }
    
    return states, time_varying
