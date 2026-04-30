# PhINs: Pharmacometrics Informed Networks
<p align="center">
  <img src="./PhINs.png" alt="PhINs workflow" width="850"/>
</p>

**PhINs** is a research library for Physics-Informed Neural Networks (PINNs) and Chebyshev-KAN / PIKAN-style models for inverse problems, gray-box discovery, hidden mechanism recovery, and mechanistic learning from sparse or partially observed data.

## Current API summary

### Main imports

```python
from phins import (
    PINNConfig, FeatureConfig, ArchitectureConfig,
    TrainingConfig, DataConfig, ParameterSpec,
    RBAConfig, AdaptiveWeightConfig, CollocationConfig,
    PINNDataBundle, PINNProblem, PINNTrainer,
)
```

### Supported architectures

```python
ArchitectureConfig(kind="mlp")
ArchitectureConfig(kind="kan", chebyshev_degree=3)
```

The current library uses `kind="kan"` for the Chebyshev-KAN / PIKAN-style architecture.

### Supported feature expansions

```python
FeatureConfig(kind="identity")
FeatureConfig(kind="sin", num_frequencies=3)
FeatureConfig(kind="cos", num_frequencies=3)
FeatureConfig(kind="sincos", num_frequencies=3)
FeatureConfig(kind="exp", num_frequencies=2)
```

### Supported parameter constraints

```python
constraint="none"
constraint="positive_softplus"
constraint="positive_exp"
```

## Quick start

```python
import jax.numpy as jnp
from phins import (
    PINNConfig, FeatureConfig, ArchitectureConfig,
    TrainingConfig, DataConfig, ParameterSpec,
    PINNDataBundle, PINNProblem, PINNTrainer,
)

t_data = jnp.linspace(0, 10, 21)[:, None]
y_data = jnp.zeros((21, 1))
t_ic = jnp.array([[0.0]])
y_ic = jnp.array([[0.0]])
t_col = jnp.linspace(0, 10, 100)[:, None]

bundle = PINNDataBundle(
    t_data=t_data,
    y_data=y_data,
    t_ic=t_ic,
    y_ic=y_ic,
    t_collocation=t_col,
)

def residual_fn(ctx, t):
    x = ctx["states"]["x"]
    dx = ctx["state_grads"]["x"]
    a = ctx["constant_params"]["a"]
    return {"ode_x": dx + a * x}

cfg = PINNConfig(
    feature=FeatureConfig(kind="identity"),
    architecture=ArchitectureConfig(
        kind="mlp",
        hidden_layers=(64, 64, 64),
        activation="tanh",
    ),
    training=TrainingConfig(
        epochs=5000,
        learning_rate=1e-3,
        optimizer="adam",
        scheduler="cosine",
        scheduler_kwargs={"decay_steps": 5000},
        print_every=500,
        loss_weights={
            "data": 1.0,
            "ic": 1.0,
            "ode_x": 1.0,
            "residual": 0.0,
        },
    ),
    data=DataConfig(
        input_dim=1,
        state_names=["x"],
        parameter_specs=[
            ParameterSpec(
                name="a",
                mode="constant",
                constraint="positive_softplus",
                init_value=0.0,
            )
        ],
    ),
)

problem = PINNProblem(cfg, residual_fn, bundle)
trainer = PINNTrainer(problem)
result = trainer.fit()

t_test = jnp.linspace(0, 10, 100)[:, None]
pred = trainer.predict(t_test)
x_pred = pred["states"]["x"]
```

## New feature: random collocation sampling

By default, the library behaves as before and uses fixed collocation points from:

```python
PINNDataBundle(t_collocation=t_col)
```

To resample collocation points during training, use `CollocationConfig`.

### 1D random collocation

```python
from phins import CollocationConfig

cfg = PINNConfig(
    data=DataConfig(
        input_dim=1,
        state_names=["x"],
        parameter_specs=[],
        collocation=CollocationConfig(
            sampling="uniform_random",
            domain=(0.0, 10.0),
            n_points=256,
            resample_every=1,
        ),
    )
)
```

### Multi-dimensional random collocation

```python
CollocationConfig(
    sampling="uniform_random",
    domain=((0.0, 1.0), (-1.0, 1.0)),
    n_points=1024,
    resample_every=1,
)
```

If `domain=None`, the trainer tries to infer bounds from `t_collocation` first and then from `t_data`.

After training, the last sampled collocation set is available as:

```python
trainer.current_t_collocation
```

## Time-varying parameters

```python
ParameterSpec(
    name="k12_tv",
    mode="time_varying",
    constraint="none",
    output_index=2,
)
```

If states are `["C1", "C2"]`, then:

```text
output 0 -> C1
output 1 -> C2
output 2 -> k12_tv
```

The updated trainer now infers output dimension using the largest time-varying `output_index`, so custom indices are supported.

## Partial observations

The default `data` loss compares all configured states with `y_data`. For partial observations, set `"data": 0.0` and use `extra_data_loss_fn`.

```python
def extra_data_loss_fn(ctx, db):
    C1_pred = ctx["states"]["C1"]
    return {
        "data_C1": jnp.mean((C1_pred - db.y_data) ** 2)
    }

problem = PINNProblem(
    config=cfg,
    residual_fn=residual_fn,
    data_bundle=bundle,
    extra_data_loss_fn=extra_data_loss_fn,
)
```

Use:

```python
loss_weights = {
    "data": 0.0,
    "data_C1": 1.0,
    "ic": 1.0,
    "ode_C1": 10.0,
    "ode_C2": 10.0,
    "residual": 0.0,
}
```

## RBA

```python
from phins import RBAConfig

training = TrainingConfig(
    rba=RBAConfig(
        enabled=True,
        target_terms=["ode_C1", "ode_C2"],
        lr_lambda=1e-2,
        gamma=0.999,
        lam_min=0.0,
        init_value=1.0,
        normalize_by_max=True,
    )
)
```

RBA reweights points inside residual terms.

## Adaptive loss weighting

The current implementation supports:

```python
method="inverse_loss"
```

Example:

```python
from phins import AdaptiveWeightConfig

training = TrainingConfig(
    adaptive_weights=AdaptiveWeightConfig(
        enabled=True,
        target_terms=["data_C1", "ode_C1", "ode_C2"],
        method="inverse_loss",
        update_every=100,
        gamma=0.9,
        eps=1e-8,
        init_value=1.0,
        normalize_sum_to=1.0,
    )
)
```

## Practical advice

Start with:

1. `mlp`
2. identity features
3. fixed collocation
4. short training

Then add:

1. bounded parameter mappings
2. random collocation
3. RBA
4. adaptive loss weighting
5. `kan`
