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

        num_time_varying = sum(1 for s in self.config.data.parameter_specs if s.mode == 'time_varying')
        output_dim = len(self.config.data.state_names) + num_time_varying
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

        apply_net = self.apply_net
        optimizer = self.optimizer
        problem_ref = self.problem

        # THIS PART IS ADDED: step can now consume dynamic adaptive term weights and RBA weights.
        def step_fn(state, adaptive_term_weights, rba_weights):
            trainable = {'net_params': state['net_params'], 'raw_constants': state['raw_constants']}

            def loss_fn(tr):
                s = {**state, **tr}
                terms = problem_ref.loss_terms(s, apply_net=apply_net, rba_weights=rba_weights)
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

    # THIS PART IS ADDED: initialize adaptive term weights.
    def _init_adaptive_term_weights(self):
        aw_cfg: AdaptiveWeightConfig | None = getattr(self.config.training, 'adaptive_weights', None)
        if aw_cfg is None or not aw_cfg.enabled:
            return {}
        target_terms = aw_cfg.target_terms
        if target_terms is None:
            target_terms = list(self.config.training.loss_weights.keys())
        return {name: float(aw_cfg.init_value) for name in target_terms}

    # THIS PART IS ADDED: initialize per-point RBA weights.
    def _init_rba_weights(self):
        rba_cfg: RBAConfig | None = getattr(self.config.training, 'rba', None)
        if rba_cfg is None or not rba_cfg.enabled:
            return {}
        res_arrays = self.problem.residual_arrays(self.state, apply_net=self.apply_net)
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

    # THIS PART IS ADDED: adaptive weighting across terms by inverse loss magnitude.
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

    # THIS PART IS ADDED: residual-based attention update over collocation residual arrays.
    def _update_rba_weights(self, rba_weights):
        rba_cfg: RBAConfig | None = getattr(self.config.training, 'rba', None)
        if rba_cfg is None or not rba_cfg.enabled:
            return rba_weights

        res_arrays = self.problem.residual_arrays(self.state, apply_net=self.apply_net)
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

    def _eval_loss(self, state, adaptive_term_weights=None, rba_weights=None):
        adaptive_term_weights = adaptive_term_weights or {}
        rba_weights = rba_weights or {}
        terms = self.problem.loss_terms(state, apply_net=self.apply_net, rba_weights=rba_weights)
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
        rba_weights = self._init_rba_weights()

        for epoch in range(self.config.training.epochs + 1):
            # Update attention-like weights using current state before the gradient step.
            rba_weights = self._update_rba_weights(rba_weights)
            state = self._jit_step(state, adaptive_term_weights, rba_weights)
            self.state = state  # keep current state visible to the update helpers

            if epoch % self.config.training.print_every == 0 or epoch == self.config.training.epochs:
                total, terms = self._eval_loss(state, adaptive_term_weights, rba_weights)
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
        self.adaptive_term_weights = adaptive_term_weights
        self.rba_weights = rba_weights
        return TrainResult(state=state, history=history)

    def predict(self, t):
        return self.problem.make_context(self.state, t, apply_net=self.apply_net)
