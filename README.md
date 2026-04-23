# PhINs

**PhINs** is a research library for **Physics-Informed Neural Networks (PINNs)** and **Physics-Informed KANs / PIKANs** for inverse problems, gray-box discovery, hidden mechanism recovery, and mechanistic learning from sparse or partially observed data.

It is built for workflows where you want to combine:
- observed data
- governing equations
- unknown constant parameters
- unknown time-varying parameters or functions
- interpretable mechanistic structure

PhINs is especially useful for:
- pharmacometrics
- PK / PD / QSP
- systems biology
- inverse ODE problems
- gray-box scientific machine learning

---

## Table of contents

- [What PhINs can do](#what-phins-can-do)
- [Package features](#package-features)
- [Installation](#installation)
- [Package structure](#package-structure)
- [How the library works](#how-the-library-works)
- [Quick start](#quick-start)
- [Defining data](#defining-data)
- [Defining the residual function](#defining-the-residual-function)
- [Feature expansion](#feature-expansion)
- [Architecture selection](#architecture-selection)
- [Constant and time-varying parameters](#constant-and-time-varying-parameters)
- [Partial observations](#partial-observations)
- [Constraints and bounded mappings](#constraints-and-bounded-mappings)
- [Loss weights](#loss-weights)
- [RBA](#rba)
- [Adaptive loss weighting](#adaptive-loss-weighting)
- [Typical examples](#typical-examples)
- [Colab / Google Drive notes](#colab--google-drive-notes)
- [Current status](#current-status)

---

## What PhINs can do

PhINs supports the following modeling patterns:

### 1. Forward physics-informed learning
You know the governing equation and want a PINN solution.

### 2. Inverse parameter estimation
You know the equation form, but some parameters are unknown and should be inferred from data.

### 3. Time-varying parameter inference
A parameter such as a rate, forcing, or transfer coefficient changes with time and must be learned as an extra network output.

### 4. Gray-box discovery
Part of the right-hand side is unknown and is learned as a neural function while the known mechanistic structure is preserved.

### 5. Partially observed systems
You may observe only a subset of the states while still enforcing the full dynamics.

### 6. MLP vs PIKAN / Chebyshev-KAN comparison
The same problem can be solved with either a standard MLP or a KAN-like architecture.

---

## Package features

PhINs currently includes support for:

### Architectures
- `mlp`
- `chebyshev_kan`
- `pikan` (alias / KAN-style usage depending on your version)

### Feature expansions
- `identity`
- `sin`
- `cos`
- `sincos`
- `exp`

### Parameter types
- constant parameters
- time-varying parameters

### Parameter constraints
- `none`
- `positive_softplus`
- `positive_exp`

### Data settings
- full-state observations
- partial observations
- custom observed-state losses
- masked / sparse data workflows depending on version

### Loss components
- data loss
- initial condition loss
- residual loss
- extra custom losses

### Training options
- optimizer selection
- scheduler selection
- static user-defined loss weights
- optional **RBA** for residual terms
- optional **adaptive loss weighting** across terms

### Utilities
- domain decomposition module
- tutorial-style examples
- problem/trainer abstraction
- prediction API

---

## Installation

### Option 1: install from source

```bash
git clone <your-repo-url>
cd PhINs
pip install -e .
```

### Option 2: use locally without installation

```python
import sys
sys.path.append("/path/to/PhINs")
import phins
```

### Recommended Python packages

Depending on your exact version of the repository, you will typically need:

```bash
pip install jax jaxlib optax numpy scipy matplotlib pandas
```

If your version includes extra research modules or older experiments, you may also need:

```bash
pip install equinox jaxtyping lineax optimistix chex pyDOE tqdm
```

### Minimal runtime dependencies for the core library
For the core `phins` package and standard examples, the main requirements are usually:
- `jax`
- `jaxlib`
- `optax`
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`

If you are using Google Colab, install missing packages first, then import the library.

---

## Package structure

A typical layout is:

```text
phins/
├── __init__.py
├── config.py
├── constraints.py
├── data.py
├── decomposition.py
├── features.py
├── models.py
├── optim.py
├── problem.py
└── trainer.py
```

### Main files

#### `config.py`
Contains dataclasses such as:
- `FeatureConfig`
- `ArchitectureConfig`
- `ParameterSpec`
- `TrainingConfig`
- `DataConfig`
- `PINNConfig`
- optionally `RBAConfig`
- optionally `AdaptiveWeightConfig`

#### `features.py`
Implements feature expansion:
- raw identity input
- sine/cosine expansions
- exponential features

#### `models.py`
Implements neural architectures such as:
- MLP
- Chebyshev-KAN / PIKAN

#### `constraints.py`
Handles:
- splitting raw outputs into states and time-varying parameters
- constant parameter transforms
- positivity constraints

#### `problem.py`
Builds:
- data loss
- IC loss
- residual terms
- optional extra losses

#### `trainer.py`
Runs:
- optimization
- learning rate schedules
- optional RBA
- optional adaptive weighting

---

## How the library works

The core workflow is:

```python
problem = PINNProblem(config, residual_fn, data_bundle)
trainer = PINNTrainer(problem)
result = trainer.fit()
pred = trainer.predict(t_test)
```

You provide:
1. **configuration**
2. **data bundle**
3. **residual function**

The library handles:
- network creation
- feature expansion
- derivative computation
- loss assembly
- optimization
- prediction

---

## Quick start

```python
import jax.numpy as jnp
from phins import (
    PINNConfig, FeatureConfig, ArchitectureConfig,
    TrainingConfig, DataConfig, ParameterSpec,
    PINNDataBundle, PINNProblem, PINNTrainer,
)

# --------------------------------------------------
# 1. Data
# --------------------------------------------------
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

# --------------------------------------------------
# 2. Residual function
# --------------------------------------------------
def residual_fn(ctx, t):
    x = ctx["states"]["x"]
    dx = ctx["state_grads"]["x"]
    a = ctx["constant_params"]["a"]

    return {
        "ode_x": dx + a * x
    }

# --------------------------------------------------
# 3. Config
# --------------------------------------------------
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

# --------------------------------------------------
# 4. Train
# --------------------------------------------------
problem = PINNProblem(cfg, residual_fn, bundle)
trainer = PINNTrainer(problem)
result = trainer.fit()

# --------------------------------------------------
# 5. Predict
# --------------------------------------------------
t_test = jnp.linspace(0, 10, 100)[:, None]
pred = trainer.predict(t_test)
x_pred = pred["states"]["x"]
```

---

## Defining data

The main container is `PINNDataBundle`.

Typical fields are:

```python
PINNDataBundle(
    t_data=...,
    y_data=...,
    t_ic=...,
    y_ic=...,
    t_collocation=...,
)
```

### Meaning
- `t_data`: time points where you have observations
- `y_data`: observed data at those points
- `t_ic`: times where initial conditions are enforced
- `y_ic`: initial condition values
- `t_collocation`: collocation points used for physics residuals

### Shape convention
Usually:
- `t_data.shape = (N, 1)`
- `y_data.shape = (N, n_observed_states)`

---

## Defining the residual function

The residual function has the form:

```python
def residual_fn(ctx, t):
    ...
    return {
        "name_of_residual_1": ...,
        "name_of_residual_2": ...,
    }
```

The context `ctx` contains:

### States
```python
ctx["states"]["C1"]
ctx["states"]["C2"]
```

### State derivatives
```python
ctx["state_grads"]["C1"]
ctx["state_grads"]["C2"]
```

### Constant parameters
```python
ctx["constant_params"]["k10"]
```

### Time-varying parameters
```python
ctx["time_varying_params"]["k12_tv"]
```

### All parameters
```python
ctx["all_params"]
```

Example:

```python
def residual_fn(ctx, t):
    C1 = ctx["states"]["C1"]
    C2 = ctx["states"]["C2"]

    dC1_dt = ctx["state_grads"]["C1"]
    dC2_dt = ctx["state_grads"]["C2"]

    k10 = ctx["constant_params"]["k10"]
    k21 = ctx["constant_params"]["k21"]
    k12 = ctx["time_varying_params"]["k12_tv"]

    return {
        "ode_C1": dC1_dt + (k10 + k12) * C1 - k21 * C2,
        "ode_C2": dC2_dt - k12 * C1 + k21 * C2,
    }
```

---

## Feature expansion

Configured through `FeatureConfig`.

Examples:

### Identity
```python
FeatureConfig(kind="identity")
```

### Sine/cosine features
```python
FeatureConfig(
    kind="sincos",
    num_frequencies=3,
    input_scale=0.1,
    include_identity=True,
)
```

### Exponential features
```python
FeatureConfig(
    kind="exp",
    num_frequencies=2,
    input_scale=0.05,
    include_identity=True,
)
```

### What `include_identity=True` means
It keeps the original input itself in the feature vector.  
So instead of only transformed features, the network receives:
- original time/input
- plus transformed features

This is often safer and more expressive.

---

## Architecture selection

Configured through `ArchitectureConfig`.

### MLP
```python
ArchitectureConfig(
    kind="mlp",
    hidden_layers=(64, 64, 64),
    activation="tanh",
)
```

### Chebyshev-KAN / PIKAN
```python
ArchitectureConfig(
    kind="chebyshev_kan",
    hidden_layers=(12, 12, 12),
    chebyshev_degree=3,
)
```

Depending on your version, `kind="pikan"` may also be supported.

### Recommendation
- start with **MLP**
- move to **PIKAN** only after the problem works

---

## Constant and time-varying parameters

### Constant parameter
```python
ParameterSpec(
    name="k10",
    mode="constant",
    constraint="positive_softplus",
    init_value=0.0,
)
```

### Time-varying parameter
```python
ParameterSpec(
    name="k12_tv",
    mode="time_varying",
    constraint="none",
    output_index=2,
)
```

If states are `["C1", "C2"]`, then:
- output 0 -> `C1`
- output 1 -> `C2`
- output 2 -> `k12_tv`

---

## Partial observations

PhINs supports the case where only part of the state vector is observed.

Example:
- model states = `["C1", "C2"]`
- data only for `C1`

A clean way is to use a custom data loss:

```python
def extra_data_loss_fn(ctx, db):
    C1_pred = ctx["states"]["C1"]
    return {
        "data_C1": jnp.mean((C1_pred - db.y_data) ** 2)
    }
```

and then set:

```python
loss_weights = {
    "data": 0.0,
    "data_C1": 1.0,
    "ode_C1": 1.0,
    "ode_C2": 1.0,
}
```

This is one of the most common workflows in PK/PD/QSP use cases.

---

## Constraints and bounded mappings

PhINs supports simple parameter constraints directly through `ParameterSpec`, but for many inverse problems a **manual bounded mapping** inside the residual is better.

Example: bounded time-varying `k12(t)` in a known range:

```python
def map_raw_to_k12(raw):
    k12_min = 0.05
    k12_max = 0.35
    return k12_min + (k12_max - k12_min) * jax.nn.sigmoid(raw)
```

Then inside the residual:

```python
k12_raw = ctx["time_varying_params"]["k12_tv"]
k12_t = map_raw_to_k12(k12_raw)
```

This is often more stable than learning the raw parameter directly.

---

## Loss weights

PhINs uses named loss terms.

Typical examples:
- `data`
- `ic`
- `residual`
- `ode_C1`
- `ode_C2`
- `data_C1`
- custom terms like `fd0`

Example:

```python
loss_weights={
    "data": 0.0,
    "data_C1": 1.0,
    "ic": 1.0,
    "ode_C1": 10.0,
    "ode_C2": 10.0,
    "residual": 0.0,
}
```

### Recommendation
- use explicit term weights
- keep `residual=0.0` if you are already weighting residual subterms individually

---

## RBA

If your current version includes RBA support, you can enable it through `RBAConfig`.

```python
from phins import RBAConfig
```

Then:

```python
training=TrainingConfig(
    epochs=50000,
    learning_rate=1e-4,
    loss_weights={
        "data_C1": 1.0,
        "ode_C1": 10.0,
        "ode_C2": 10.0,
        "residual": 0.0,
    },
    rba=RBAConfig(
        enabled=True,
        target_terms=["ode_C1", "ode_C2"],
        lr_lambda=1e-2,
        gamma=0.999,
        lam_min=0.0,
        init_value=1.0,
        normalize_by_max=True,
    ),
)
```

### What RBA does
RBA applies adaptive weights **inside a residual term**, typically pointwise over collocation residuals.

This is different from ordinary scalar loss weights.

### Best use
Use RBA for:
- ODE residual terms
- PDE residual terms

Do **not** usually use it for:
- data loss
- IC loss
- anchor losses

---

## Adaptive loss weighting

If your version includes adaptive loss weighting across terms, import:

```python
from phins import AdaptiveWeightConfig
```

Then:

```python
training=TrainingConfig(
    epochs=50000,
    learning_rate=1e-4,
    loss_weights={
        "data_C1": 1.0,
        "ode_C1": 10.0,
        "ode_C2": 10.0,
        "fd0": 10.0,
    },
    adaptive_weights=AdaptiveWeightConfig(
        enabled=True,
        target_terms=["data_C1", "ode_C1", "ode_C2", "fd0"],
        method="loss_ratio",
        update_every=100,
        gamma=0.9,
    ),
)
```

### What it does
Adaptive weighting rebalances **between loss terms**, while RBA rebalances **within a residual term**.

---

## Typical examples

PhINs is suited for examples such as:

### Example 1: inverse ODE with unknown constants
Learn `kg`, `kb` in a 3-state compartment model.

### Example 2: gray-box unknown RHS function
Learn an unknown function `f(t)` appearing in one ODE equation.

### Example 3: PK/PD with time-varying transfer or absorption
Learn `k12(t)` or `ka(t)` while preserving mechanistic compartments.

### Example 4: hidden dynamic factor
Learn a hidden forcing or growth modulation term such as `FD(t)`.

### Example 5: MLP vs PIKAN comparison
Solve the same inverse problem with both architectures.

---

## Colab / Google Drive notes

If using Google Colab:

### Install packages first
```bash
pip install jax jaxlib optax numpy scipy matplotlib pandas
```

### Prefer running from local `/content`
Google Drive is much slower than local disk in Colab.

A faster workflow is:
1. copy your repository from Drive to `/content`
2. run there
3. copy results back to Drive

### Example
```python
!cp -r /content/drive/MyDrive/Phins /content/Phins
import sys
sys.path.append('/content/Phins')
```

---

## Current status

PhINs is best described as a **research software library**:
- flexible
- easy to customize
- useful for inverse scientific ML
- especially good for ODE/PINN/gray-box experimentation

It is not intended to be a rigid black-box framework.  
It is designed for users who want to control:
- the residual
- the outputs
- the constraints
- the loss design
- the training strategy

---

## Practical advice

### Start simple
Before using:
- PIKAN
- exp features
- RBA
- adaptive weighting

first make sure the problem works with:
- MLP
- identity features
- moderate collocation count
- short training

### Good progression
1. MLP + identity  
2. add bounded parameter mapping  
3. add exp/sincos features if needed  
4. add RBA  
5. add adaptive loss weighting  
6. try PIKAN  

---

## Citation / acknowledgement

If you use PhINs in your research, please cite the associated tutorial / methodological papers from this project when available.

---

## Contributing

Issues, example contributions, and pull requests are welcome, especially for:
- pharmacometrics
- systems biology
- gray-box scientific ML
- improved examples
- training-speed improvements
