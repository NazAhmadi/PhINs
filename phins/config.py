from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple


FeatureType = Literal['identity', 'sin', 'cos', 'sincos', 'exp']
ArchitectureType = Literal['mlp', 'chebyshev_kan']
ActivationType = Literal['tanh', 'sin', 'relu', 'gelu', 'softplus']
OptimizerType = Literal['adam', 'adamw', 'radam', 'adamax', 'rmsprop', 'sgd']
SchedulerType = Literal['none', 'cosine', 'exponential', 'piecewise']
ConstraintType = Literal['none', 'positive_softplus', 'positive_exp']


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
    output_dim: int = 1
    chebyshev_degree: int = 3


@dataclass
class ParameterSpec:
    name: str
    mode: Literal['constant', 'time_varying'] = 'constant'
    constraint: ConstraintType = 'none'
    init_value: float = 0.0
    output_index: Optional[int] = None


# THIS PART IS ADDED: optional residual-based attention (RBA) config.
@dataclass
class RBAConfig:
    enabled: bool = False
    target_terms: Optional[List[str]] = None
    lr_lambda: float = 1e-2
    gamma: float = 0.999
    lam_min: float = 0.0
    init_value: float = 1.0
    normalize_by_max: bool = True


# THIS PART IS ADDED: optional adaptive weighting across loss terms.
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
class TrainingConfig:
    epochs: int = 10_000
    learning_rate: float = 1e-3
    optimizer: OptimizerType = 'adam'
    scheduler: SchedulerType = 'none'
    scheduler_kwargs: Dict[str, float] = field(default_factory=dict)
    loss_weights: Dict[str, float] = field(default_factory=lambda: {'data': 1.0, 'residual': 1.0, 'ic': 1.0})
    print_every: int = 500
    seed: int = 0
    # THIS PART IS ADDED: optional adaptive term weighting and RBA.
    adaptive_weights: AdaptiveWeightConfig | None = None
    rba: RBAConfig | None = None


@dataclass
class DataConfig:
    input_dim: int = 1
    state_names: Sequence[str] = field(default_factory=list)
    parameter_specs: Sequence[ParameterSpec] = field(default_factory=list)
    collocation_shape: Tuple[int, ...] = (256, 1)


@dataclass
class PINNConfig:
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
