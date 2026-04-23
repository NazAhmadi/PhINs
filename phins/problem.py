from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import jax
import jax.numpy as jnp

from .config import PINNConfig
from .constraints import split_outputs, transform_constant_parameters
from .data import PINNDataBundle
from .features import apply_feature_expansion


ResidualFn = Callable[[dict, jnp.ndarray], Dict[str, jnp.ndarray]]
DataLossFn = Callable[[dict, PINNDataBundle], Dict[str, jnp.ndarray]]


def mse(a, b):
    return jnp.mean((a - b) ** 2)


@dataclass
class PINNProblem:
    config: PINNConfig
    residual_fn: ResidualFn
    data_bundle: PINNDataBundle
    extra_data_loss_fn: DataLossFn | None = None

    def make_context(self, trainer_state, t: jnp.ndarray, apply_net=None) -> dict:
        if apply_net is None:
            apply_net = trainer_state.get('apply_net')
        feats = apply_feature_expansion(t, self.config.feature)
        raw_output = apply_net(trainer_state['net_params'], feats)
        states, tv_params = split_outputs(raw_output, self.config.data.state_names, self.config.data.parameter_specs)
        const_params = transform_constant_parameters(trainer_state['raw_constants'], self.config.data.parameter_specs)

        def scalarized_output(x):
            f = apply_feature_expansion(x, self.config.feature)
            return apply_net(trainer_state['net_params'], f)

        def grad_of_output(idx, x):
            return jax.grad(lambda z: jnp.sum(scalarized_output(z)[:, idx:idx+1]))(x)

        state_grads = {
            name: grad_of_output(i, t)
            for i, name in enumerate(self.config.data.state_names)
        }

        return {
            't': t,
            'raw_output': raw_output,
            'states': states,
            'state_grads': state_grads,
            'time_varying_params': tv_params,
            'constant_params': const_params,
            'all_params': {**const_params, **tv_params},
            'data_bundle': self.data_bundle,
        }

    # THIS PART IS ADDED: expose raw residual arrays before reduction so RBA can weight pointwise residuals.
    def residual_arrays(self, trainer_state, apply_net=None) -> Dict[str, jnp.ndarray]:
        db = self.data_bundle
        if db.t_collocation is None:
            return {}
        ctx_col = self.make_context(trainer_state, db.t_collocation, apply_net=apply_net)
        res = self.residual_fn(ctx_col, db.t_collocation)
        out = {}
        for name, val in res.items():
            val = jnp.asarray(val)
            if val.ndim == 1:
                val = val[:, None]
            out[name] = val
        return out

    def loss_terms(self, trainer_state, apply_net=None, rba_weights: Dict[str, jnp.ndarray] | None = None) -> Dict[str, jnp.ndarray]:
        terms = {}
        db = self.data_bundle

        if db.t_data is not None and db.y_data is not None:
            ctx_data = self.make_context(trainer_state, db.t_data, apply_net=apply_net)
            pred_state_mat = jnp.concatenate([ctx_data['states'][name] for name in self.config.data.state_names], axis=1)
            terms['data'] = mse(pred_state_mat, db.y_data)
        else:
            terms['data'] = jnp.array(0.0)

        if db.t_ic is not None and db.y_ic is not None:
            ctx_ic = self.make_context(trainer_state, db.t_ic, apply_net=apply_net)
            pred_ic = jnp.concatenate([ctx_ic['states'][name] for name in self.config.data.state_names], axis=1)
            terms['ic'] = mse(pred_ic, db.y_ic)
        else:
            terms['ic'] = jnp.array(0.0)

        res_arrays = self.residual_arrays(trainer_state, apply_net=apply_net)
        if res_arrays:
            for name, val in res_arrays.items():
                if rba_weights is not None and name in rba_weights:
                    w = jnp.asarray(rba_weights[name])
                    if w.ndim == 1:
                        w = w[:, None]
                    terms[name] = jnp.mean(w * (val ** 2))
                else:
                    terms[name] = jnp.mean(val ** 2)
            terms['residual'] = jnp.sum(jnp.array([terms[name] for name in res_arrays.keys()]))
        else:
            terms['residual'] = jnp.array(0.0)

        if self.extra_data_loss_fn is not None and db.t_data is not None:
            ctx_extra = self.make_context(trainer_state, db.t_data, apply_net=apply_net)
            extra = self.extra_data_loss_fn(ctx_extra, db)
            terms.update(extra)

        return terms
