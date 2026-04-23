"""
Neural Network Architectures for PhINs
=======================================

Supports:
1. Standard MLPs (Multi-Layer Perceptrons)
2. PIKANs (Physics-Informed Kolmogorov-Arnold Networks) using Chebyshev polynomials

Based on successful implementations from chemotherapy gray-box discovery models.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List
import jax
import jax.numpy as jnp

from .config import ArchitectureConfig

Array = jnp.ndarray


def get_activation(name: str) -> Callable[[Array], Array]:
    """Get activation function by name."""
    table = {
        'tanh': jax.nn.tanh,
        'sin': jnp.sin,
        'relu': jax.nn.relu,
        'gelu': jax.nn.gelu,
        'softplus': jax.nn.softplus,
    }
    if name not in table:
        raise ValueError(f'Unknown activation: {name}. Options: {list(table.keys())}')
    return table[name]


# ══════════════════════════════════════════════════════════════════════════════
# STANDARD MLP
# ══════════════════════════════════════════════════════════════════════════════

def init_mlp(layer_sizes, key):
    """
    Initialize MLP parameters.
    
    Args:
        layer_sizes: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
        key: JAX random key
        
    Returns:
        List of parameter dictionaries with 'W' and 'b' keys
    """
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params = []
    for k, n_in, n_out in zip(keys, layer_sizes[:-1], layer_sizes[1:]):
        W = jax.random.normal(k, (n_in, n_out)) / jnp.sqrt(n_in)
        b = jnp.zeros((n_out,))
        params.append({'W': W, 'b': b})
    return params


def forward_mlp(params, x, activation_name='tanh'):
    """
    Forward pass through MLP.
    
    Args:
        params: List of layer parameters
        x: Input array [batch_size, input_dim]
        activation_name: Activation function name
        
    Returns:
        Output array [batch_size, output_dim]
    """
    act = get_activation(activation_name)
    *hidden, last = params
    h = x
    for layer in hidden:
        h = act(h @ layer['W'] + layer['b'])
    return h @ last['W'] + last['b']


# ══════════════════════════════════════════════════════════════════════════════
# PIKAN (Chebyshev-KAN)
# ══════════════════════════════════════════════════════════════════════════════

def chebyshev_recursive(x: Array, degree: int) -> Array:
    """
    Compute Chebyshev polynomial T_n(x) recursively.
    
    Uses recurrence: T_0(x) = 1, T_1(x) = x, T_{n+1}(x) = 2x*T_n(x) - T_{n-1}(x)
    
    Args:
        x: Input array
        degree: Polynomial degree n
        
    Returns:
        T_n(x)
    """
    if degree == 0:
        return jnp.ones_like(x)
    elif degree == 1:
        return x
    
    T_n_minus_2 = jnp.ones_like(x)
    T_n_minus_1 = x
    
    for n in range(2, degree + 1):
        T_n = 2 * x * T_n_minus_1 - T_n_minus_2
        T_n_minus_2, T_n_minus_1 = T_n_minus_1, T_n
    
    return T_n


def init_chebyshev_kan(layer_sizes: List[int], degree: int, key) -> List[Dict]:
    """
    Initialize Chebyshev-KAN parameters.
    
    Architecture: Each hidden layer uses Chebyshev polynomial basis functions.
    Last layer is a standard linear transformation.
    
    Args:
        layer_sizes: Network architecture [input_dim, hidden1, ..., output_dim]
        degree: Chebyshev polynomial degree
        key: JAX random key
        
    Returns:
        List of parameter dictionaries
    """
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params = []
    
    # Hidden layers with Chebyshev basis
    for k, n_in, n_out in zip(keys[:-1], layer_sizes[:-2], layer_sizes[1:-1]):
        # Shape: (n_in, n_out, degree+1)
        # Each input-output connection has its own polynomial coefficients
        W = jax.random.normal(k, (n_in, n_out, degree + 1)) / (n_in * (degree + 1))
        B = jax.random.normal(k, (n_out,))
        params.append({'W': W, 'B': B})
    
    # Last layer (standard linear)
    W = jax.random.normal(keys[-1], (layer_sizes[-2], layer_sizes[-1])) / jnp.sqrt(layer_sizes[-2])
    B = jax.random.normal(keys[-1], (layer_sizes[-1],))
    params.append({'W': W, 'B': B})
    
    return params


def forward_chebyshev_kan(params: List[Dict], x: Array, degree: int, 
                          activation_name: str = 'tanh',
                          input_scale: float = 0.01) -> Array:
    """
    Forward pass through Chebyshev-KAN.
    
    Process:
    1. Scale input (for numerical stability)
    2. For each hidden layer:
       - Apply activation
       - Compute Chebyshev basis [T_0(x), T_1(x), ..., T_degree(x)]
       - Linear combination with learned coefficients
       - Apply activation again
    3. Final linear layer
    
    Args:
        params: Network parameters
        x: Input [batch_size, input_dim]
        degree: Chebyshev polynomial degree
        activation_name: Activation function
        input_scale: Scaling factor for input (improves numerics)
        
    Returns:
        Output [batch_size, output_dim]
    """
    act = get_activation(activation_name)
    
    # Scale input for numerical stability
    X = x * input_scale
    
    # Hidden layers with Chebyshev basis
    *hidden, last = params
    for layer in hidden:
        cheby_coeffs = layer['W']  # Shape: (n_in, n_out, degree+1)
        
        # Apply activation
        X = act(X)
        
        # Compute Chebyshev basis for each input feature
        # Stack all polynomial degrees: [batch, features, degree+1]
        X_stack = jnp.stack([chebyshev_recursive(X, d) for d in range(degree + 1)], axis=-1)
        
        # Einstein summation: batch × input × degrees -> batch × output
        # 'bid' = batch, input, degree
        # 'iod' = input, output, degree
        # 'bo' = batch, output
        X = jnp.einsum("bid,iod->bo", X_stack, cheby_coeffs) + layer['B']
        
        # Apply activation again
        X = act(X)
    
    # Final linear layer
    return X @ last['W'] + last['B']


# ══════════════════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelBundle:
    """Container for model parameters and forward function."""
    params: List[Dict]
    apply: Callable[[List[Dict], Array], Array]
    architecture: str  # 'mlp' or 'pikan'


def build_model(cfg: ArchitectureConfig, input_dim: int, output_dim: int, key) -> ModelBundle:
    """
    Build neural network model based on configuration.
    
    Args:
        cfg: Architecture configuration
        input_dim: Input dimension
        output_dim: Output dimension
        key: JAX random key
        
    Returns:
        ModelBundle with parameters and forward function
    """
    layer_sizes = [input_dim, *cfg.hidden_layers, output_dim]
    
    if cfg.kind == 'mlp':
        params = init_mlp(layer_sizes, key)
        apply_fn = lambda p, x: forward_mlp(p, x, cfg.activation)
        return ModelBundle(params=params, apply=apply_fn, architecture='mlp')
    
    elif cfg.kind == 'pikan':
        # Chebyshev-KAN (Physics-Informed Kolmogorov-Arnold Network)
        degree = cfg.chebyshev_degree
        params = init_chebyshev_kan(layer_sizes, degree, key)
        apply_fn = lambda p, x: forward_chebyshev_kan(
            p, x, degree, cfg.activation, input_scale=0.01
        )
        return ModelBundle(params=params, apply=apply_fn, architecture='pikan')
    
    else:
        raise ValueError(f"Unknown architecture: {cfg.kind}. Options: 'mlp', 'pikan'")


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def count_parameters(params: List[Dict]) -> int:
    """Count total number of parameters in network."""
    total = 0
    for layer in params:
        for key, val in layer.items():
            total += val.size
    return total


def print_model_summary(model: ModelBundle, input_dim: int, output_dim: int):
    """Print summary of model architecture."""
    print("="*70)
    print("MODEL SUMMARY")
    print("="*70)
    print(f"Architecture: {model.architecture.upper()}")
    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    print(f"Total parameters: {count_parameters(model.params):,}")
    print(f"Layers: {len(model.params)}")
    print("="*70)
