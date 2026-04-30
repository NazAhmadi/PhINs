from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import jax
import jax.numpy as jnp
import optax

from .config import PINNConfig, AdaptiveWeightConfig, RBAConfig
from .constraints import raw_constant_parameters
from .models import build_model
from .optim import build_optimizer
from .problem import PINNProblem
from .features import apply_feature_expansion
from .samplers import infer_domain_from_points, sample_uniform_collocation


@dataclass
class TrainResult:
    state: dict
    history: List[Dict[str, float]]


class PINNTrainer:
    def __init__(self, problem: PINNProblem):
        self.problem = problem
        self.config: PINNConfig = problem.config

        key = jax.random.PRNGKey(self.config.training.seed)
        dummy_x = jnp.zeros((1, self.config.data.input_dim))
        feat_dim = apply_feature_expansion(dummy_x, self.config.feature).shape[-1]

        output_dim = self._infer_output_dim()
        bundle = build_model(self.config.architecture, feat_dim, output_dim, key)

        self.optimizer = build_optimizer(self.config.training)
        self.apply_net = bundle.apply
        raw_consts = raw_constant_parameters(self.config.data.parameter_specs)
        self.state = {
            'net_params': bundle.params,
            'raw_constants': raw_consts,
            'opt_state_net': self.optimizer.init(bundle.params),
            'opt_state_const': self.optimizer.init(raw_consts),
        }
        self.current_t_collocation = None

        apply_net = self.apply_net
        optimizer = self.optimizer
        problem_ref = self.problem

        def step_fn(state, adaptive_term_weights, rba_weights, t_collocation):
            trainable = {'net_params': state['net_params'], 'raw_constants': state['raw_constants']}

            def loss_fn(tr):
                s = {**state, **tr}
                terms = problem_ref.loss_terms(
                    s,
                    apply_net=apply_net,
                    rba_weights=rba_weights,
                    t_collocation=t_collocation,
                )
                total = jnp.array(0.0)
                for k, v in terms.items():
                    base_w = self.config.training.loss_weights.get(k, 1.0)
                    adapt_w = adaptive_term_weights.get(k, 1.0)
                    total = total + base_w * adapt_w * v
                return total

            grads = jax.grad(loss_fn)(trainable)
            net_updates, opt_state_net = optimizer.update(grads['net_params'], state['opt_state_net'], state['net_params'])
            const_updates, opt_state_const = optimizer.update(grads['raw_constants'], state['opt_state_const'], state['raw_constants'])
            return {
                **state,
                'net_params': optax.apply_updates(state['net_params'], net_updates),
                'raw_constants': optax.apply_updates(state['raw_constants'], const_updates),
                'opt_state_net': opt_state_net,
                'opt_state_const': opt_state_const,
            }

        self._jit_step = jax.jit(step_fn)

    def _infer_output_dim(self) -> int:
        state_dim = len(self.config.data.state_names)
        tv_specs = [s for s in self.config.data.parameter_specs if s.mode == 'time_varying']
        tv_indices = [s.output_index for s in tv_specs if s.output_index is not None]
        if tv_indices:
            return max(state_dim, max(tv_indices) + 1)
        return state_dim + len(tv_specs)

    def _resolve_collocation_domain(self):
        col_cfg = self.config.data.collocation
        domain = col_cfg.domain
        if domain is not None:
            return domain
        domain = infer_domain_from_points(self.problem.data_bundle.t_collocation, self.config.data.input_dim)
        if domain is not None:
            return domain
        domain = infer_domain_from_points(self.problem.data_bundle.t_data, self.config.data.input_dim)
        if domain is not None:
            return domain
        raise ValueError(
            "collocation.sampling='uniform_random' needs collocation.domain, "
            "or t_collocation/t_data from which bounds can be inferred."
        )

    def _initial_collocation(self):
        col_cfg = self.config.data.collocation
        if col_cfg.sampling == 'fixed':
            return self.problem.data_bundle.t_collocation
        if col_cfg.sampling == 'uniform_random':
            key = jax.random.PRNGKey(self.config.training.seed)
            return sample_uniform_collocation(
                key,
                domain=self._resolve_collocation_domain(),
                n_points=col_cfg.n_points,
                input_dim=self.config.data.input_dim,
            )
        raise ValueError(f"Unknown collocation sampling mode: {col_cfg.sampling}")

    def _maybe_resample_collocation(self, epoch: int, key, current_t_col):
        col_cfg = self.config.data.collocation
        if col_cfg.sampling == 'fixed':
            return key, current_t_col
        if col_cfg.sampling != 'uniform_random':
            raise ValueError(f"Unknown collocation sampling mode: {col_cfg.sampling}")
        if epoch % max(int(col_cfg.resample_every), 1) != 0:
            return key, current_t_col
        key, subkey = jax.random.split(key)
        t_col = sample_uniform_collocation(
            subkey,
            domain=self._resolve_collocation_domain(),
            n_points=col_cfg.n_points,
            input_dim=self.config.data.input_dim,
        )
        return key, t_col

    def _init_adaptive_term_weights(self):
        aw_cfg: AdaptiveWeightConfig | None = getattr(self.config.training, 'adaptive_weights', None)
        if aw_cfg is None or not aw_cfg.enabled:
            return {}
        target_terms = aw_cfg.target_terms
        if target_terms is None:
            target_terms = list(self.config.training.loss_weights.keys())
        return {name: float(aw_cfg.init_value) for name in target_terms}

    def _init_rba_weights(self, t_collocation=None):
        rba_cfg: RBAConfig | None = getattr(self.config.training, 'rba', None)
        if rba_cfg is None or not rba_cfg.enabled:
            return {}
        res_arrays = self.problem.residual_arrays(self.state, apply_net=self.apply_net, t_collocation=t_collocation)
        if not res_arrays:
            return {}
        target_terms = rba_cfg.target_terms
        if target_terms is None:
            target_terms = list(res_arrays.keys())
        weights = {}
        for name in target_terms:
            if name in res_arrays:
                weights[name] = jnp.ones_like(res_arrays[name]) * rba_cfg.init_value
        return weights

    def _update_adaptive_term_weights(self, terms: Dict[str, float], adaptive_term_weights: Dict[str, float], epoch: int):
        aw_cfg: AdaptiveWeightConfig | None = getattr(self.config.training, 'adaptive_weights', None)
        if aw_cfg is None or not aw_cfg.enabled:
            return adaptive_term_weights
        if epoch % aw_cfg.update_every != 0:
            return adaptive_term_weights

        target_terms = aw_cfg.target_terms
        if target_terms is None:
            target_terms = list(adaptive_term_weights.keys())

        raw = {}
        for name in target_terms:
            Li = float(terms.get(name, 0.0))
            raw[name] = 1.0 / (Li + aw_cfg.eps)

        total_raw = sum(raw.values())
        if total_raw <= 0:
            return adaptive_term_weights

        normalized = {
            name: aw_cfg.normalize_sum_to * raw[name] / total_raw
            for name in raw
        }
        updated = {}
        for name in normalized:
            prev = adaptive_term_weights.get(name, aw_cfg.init_value)
            updated[name] = aw_cfg.gamma * prev + (1.0 - aw_cfg.gamma) * normalized[name]
        return updated

    def _update_rba_weights(self, rba_weights, t_collocation=None):
        rba_cfg: RBAConfig | None = getattr(self.config.training, 'rba', None)
        if rba_cfg is None or not rba_cfg.enabled:
            return rba_weights

        res_arrays = self.problem.residual_arrays(self.state, apply_net=self.apply_net, t_collocation=t_collocation)
        if not res_arrays:
            return rba_weights

        target_terms = rba_cfg.target_terms
        if target_terms is None:
            target_terms = list(res_arrays.keys())

        new_weights = dict(rba_weights)
        for name in target_terms:
            if name not in res_arrays:
                continue
            r = jnp.abs(res_arrays[name])
            if rba_cfg.normalize_by_max:
                denom = jnp.maximum(jnp.max(r), 1e-12)
                r_scaled = r / denom
            else:
                r_scaled = r
            prev = new_weights.get(name, jnp.ones_like(r) * rba_cfg.init_value)
            updated = rba_cfg.gamma * prev + rba_cfg.lr_lambda * r_scaled
            updated = jnp.maximum(updated, rba_cfg.lam_min)
            new_weights[name] = updated
        return new_weights

    def _eval_loss(self, state, adaptive_term_weights=None, rba_weights=None, t_collocation=None):
        adaptive_term_weights = adaptive_term_weights or {}
        rba_weights = rba_weights or {}
        terms = self.problem.loss_terms(
            state,
            apply_net=self.apply_net,
            rba_weights=rba_weights,
            t_collocation=t_collocation,
        )
        total = 0.0
        for k, v in terms.items():
            base_w = self.config.training.loss_weights.get(k, 1.0)
            adapt_w = adaptive_term_weights.get(k, 1.0)
            total += base_w * adapt_w * float(v)
        return float(total), {k: float(v) for k, v in terms.items()}

    def fit(self):
        history = []
        state = self.state
        adaptive_term_weights = self._init_adaptive_term_weights()
        t_collocation = self._initial_collocation()
        self.current_t_collocation = t_collocation
        rba_weights = self._init_rba_weights(t_collocation=t_collocation)
        key = jax.random.PRNGKey(self.config.training.seed + 1)

        for epoch in range(self.config.training.epochs + 1):
            key, t_collocation = self._maybe_resample_collocation(epoch, key, t_collocation)
            self.current_t_collocation = t_collocation

            rba_weights = self._update_rba_weights(rba_weights, t_collocation=t_collocation)
            state = self._jit_step(state, adaptive_term_weights, rba_weights, t_collocation)
            self.state = state

            if epoch % self.config.training.print_every == 0 or epoch == self.config.training.epochs:
                total, terms = self._eval_loss(state, adaptive_term_weights, rba_weights, t_collocation=t_collocation)
                adaptive_term_weights = self._update_adaptive_term_weights(terms, adaptive_term_weights, epoch)
                item = {
                    'epoch': float(epoch),
                    'total': total,
                    **terms,
                    **{f'adapt_{k}': float(v) for k, v in adaptive_term_weights.items()},
                }
                history.append(item)
                print(item)

        self.state = state
        self.current_t_collocation = t_collocation
        self.adaptive_term_weights = adaptive_term_weights
        self.rba_weights = rba_weights
        return TrainResult(state=state, history=history)

    def predict(self, t):
        return self.problem.make_context(self.state, t, apply_net=self.apply_net)
