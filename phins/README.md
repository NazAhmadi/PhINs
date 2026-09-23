# PhINs

PhINs (Pharmacometrics-Informed Networks) is a JAX-based library for physics-informed learning in pharmacometrics and quantitative systems pharmacology.

## Package structure

The core package is organized as:

```text
phins/
├── __init__.py
├── config.py
├── constraints.py
├── data.py
├── features.py
├── models.py
├── optim.py
├── problem.py
├── samplers.py
└── trainer.py
```

### Main files

#### `config.py`

Contains configuration dataclasses:

- `FeatureConfig`
- `ArchitectureConfig`
- `ParameterSpec`
- `RBAConfig`
- `AdaptiveWeightConfig`
- `CollocationConfig`
- `TrainingConfig`
- `DataConfig`
- `PINNConfig`

#### `features.py`

Implements feature expansion:

- raw identity input
- sine features
- cosine features
- sine/cosine features
- exponential features

#### `models.py`

Implements neural architectures:

- MLP
- Chebyshev-KAN / PIKAN-style KAN

#### `constraints.py`

Handles:

- splitting raw network outputs into states and time-varying parameters
- constant parameter transforms
- positivity constraints
- validation of time-varying parameter output indices

#### `data.py`

Defines the `PINNDataBundle` for storing:

- observation data
- initial conditions
- collocation points
- optional metadata

#### `problem.py`

Builds:

- data loss
- initial-condition loss
- residual terms
- optional extra losses
- context dictionaries for residual functions
- state derivatives using a Jacobian-based computation while preserving the `ctx["state_grads"][state_name]` API

#### `samplers.py`

Implements collocation sampling utilities, including uniform random collocation points over a user-defined domain.

#### `optim.py`

Implements:

- optimizer construction
- learning-rate schedules

#### `trainer.py`

Runs:

- optimization
- learning-rate schedules
- fixed or randomly resampled collocation points
- residual-based attention
- adaptive weighting
- prediction
