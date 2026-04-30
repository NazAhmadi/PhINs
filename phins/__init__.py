"""
PhINs - Physics-Informed Neural Networks for Pharmacometrics
"""

from .config import (
    PINNConfig, FeatureConfig, ArchitectureConfig,
    TrainingConfig, DataConfig, ParameterSpec,
    RBAConfig, AdaptiveWeightConfig, CollocationConfig,
)
from .data import PINNDataBundle
from .problem import PINNProblem
from .trainer import PINNTrainer, TrainResult

__version__ = "2.2.0"
__all__ = [
    'PINNConfig', 'FeatureConfig', 'ArchitectureConfig',
    'TrainingConfig', 'DataConfig', 'ParameterSpec',
    'RBAConfig', 'AdaptiveWeightConfig', 'CollocationConfig',
    'PINNDataBundle', 'PINNProblem', 'PINNTrainer', 'TrainResult'
]
