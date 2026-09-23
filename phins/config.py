from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple


FeatureType = Literal['identity', 'sin', 'cos', 'sincos', 'exp']
ArchitectureType = Literal['mlp', 'kan']
ActivationType = Literal['tanh', 'sin', 'relu', 'gelu', 'softplus']
OptimizerType = Literal['adam', 'adamw', 'radam', 'adamax', 'rmsprop', 'sgd']
SchedulerType = Literal['none', 'cosine', 'exponential', 'piecewise']
ConstraintType = Literal['none', 'positive_softplus', 'positive_exp']
CollocationSamplingType = Literal['fixed', 'uniform_random']


@dataclass
class FeatureConfig:
    kind: FeatureType = 'identity'
    num_frequencies: int = 0
    input_scale: float = 1.0
    include_identity: bool = True


@dataclass
class ArchitectureConfig:
    kind: ArchitectureType = 'mlp'
    hidden_layers: Sequence[int] = (64, 64, 64)
    activation: ActivationType = 'tanh'
    output_dim: int = 1  # kept for backward compatibility; trainer infers required output dim
    chebyshev_degree: int = 3


@dataclass
class ParameterSpec:
    name: str
    mode: Literal['constant', 'time_varying'] = 'constant'
    constraint: ConstraintType = 'none'
    init_value: float = 0.0
    output_index: Optional[int] = None


@dataclass
class RBAConfig:
    enabled: bool = False
    target_terms: Optional[List[str]] = None
    lr_lambda: float = 1e-2
    gamma: float = 0.999
    lam_min: float = 0.0
    init_value: float = 1.0
    normalize_by_max: bool = True


@dataclass
class AdaptiveWeightConfig:
    enabled: bool = False
    target_terms: Optional[List[str]] = None
    method: Literal['inverse_loss'] = 'inverse_loss'
    update_every: int = 100
    gamma: float = 0.9
    eps: float = 1e-8
    init_value: float = 1.0
    normalize_sum_to: float = 1.0


@dataclass
class CollocationConfig:
    """Controls collocation-point generation.

    sampling='fixed': use PINNDataBundle.t_collocation exactly as before.
    sampling='uniform_random': resample collocation points during training.

    domain can be:
      - (t_min, t_max) for one-dimensional inputs
      - ((x1_min, x1_max), ..., (xd_min, xd_max)) for multi-dimensional inputs

    If domain is None and sampling='uniform_random', the trainer infers bounds from
    t_collocation first, then t_data.
    """
    sampling: CollocationSamplingType = 'fixed'
    domain: Optional[Tuple] = None
    n_points: int = 256
    resample_every: int = 1


@dataclass
class TrainingConfig:
    epochs: int = 10_000
    learning_rate: float = 1e-3
    optimizer: OptimizerType = 'adam'
    scheduler: SchedulerType = 'none'
    scheduler_kwargs: Dict[str, float] = field(default_factory=dict)
    loss_weights: Dict[str, float] = field(default_factory=lambda: {'data': 1.0, 'residual': 1.0, 'ic': 1.0})
    print_every: int = 500
    seed: int = 0
    adaptive_weights: AdaptiveWeightConfig | None = None
    rba: RBAConfig | None = None


@dataclass
class DataConfig:
    input_dim: int = 1
    state_names: Sequence[str] = field(default_factory=list)
    parameter_specs: Sequence[ParameterSpec] = field(default_factory=list)
    collocation_shape: Tuple[int, ...] = (256, 1)
    collocation: CollocationConfig = field(default_factory=CollocationConfig)


@dataclass
class PINNConfig:
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
