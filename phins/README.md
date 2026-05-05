# Coming Soon

This section will be uploaded upon acceptance of the tutorial manuscript.

Stay tuned — it will be available soon!

## Package structure

A typical layout is:

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

#### `problem.py`

Builds:

- data loss
- initial-condition loss
- residual terms
- optional extra losses
- context dictionaries for residual functions
- state derivatives using a Jacobian-based computation, while preserving the old `ctx["state_grads"][state_name]` API

#### `samplers.py`

Implements collocation sampling utilities, including uniform random collocation points over a user-defined domain.

#### `trainer.py`

Runs:

- optimization
- learning-rate schedules
- fixed or randomly resampled collocation points
- residual-based attention
- adaptive weighting
- prediction

---

## How the library works

The core workflow is:

```python
problem = PINNProblem(config, residual_fn, data_bundle)
trainer = PINNTrainer(problem)
result = trainer.fit()
pred = trainer.predict(t_test)
```

