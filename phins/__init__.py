"""
PhINs - Physics-Informed Neural Networks for Pharmacometrics
"""

from .config import (
    PINNConfig, FeatureConfig, ArchitectureConfig,
    TrainingConfig, DataConfig, ParameterSpec,
    RBAConfig, AdaptiveWeightConfig,
)
from .data import PINNDataBundle
from .problem import PINNProblem
from .trainer import PINNTrainer, TrainResult

__version__ = "2.1.0"
__all__ = [
    'PINNConfig', 'FeatureConfig', 'ArchitectureConfig',
    'TrainingConfig', 'DataConfig', 'ParameterSpec',
    # THIS PART IS ADDED: export new weighting configs.
    'RBAConfig', 'AdaptiveWeightConfig',
    'PINNDataBundle', 'PINNProblem', 'PINNTrainer', 'TrainResult'
]
