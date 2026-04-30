from __future__ import annotations

import optax

from .config import TrainingConfig


def build_scheduler(cfg: TrainingConfig):
    if cfg.scheduler == 'none':
        return cfg.learning_rate
    if cfg.scheduler == 'cosine':
        decay_steps = int(cfg.scheduler_kwargs.get('decay_steps', cfg.epochs))
        alpha = float(cfg.scheduler_kwargs.get('alpha', 0.0))
        return optax.cosine_decay_schedule(cfg.learning_rate, decay_steps=decay_steps, alpha=alpha)
    if cfg.scheduler == 'exponential':
        transition_steps = int(cfg.scheduler_kwargs.get('transition_steps', max(cfg.epochs // 10, 1)))
        decay_rate = float(cfg.scheduler_kwargs.get('decay_rate', 0.95))
        return optax.exponential_decay(cfg.learning_rate, transition_steps=transition_steps, decay_rate=decay_rate)
    if cfg.scheduler == 'piecewise':
        boundaries_and_scales = cfg.scheduler_kwargs.get('boundaries_and_scales', {int(cfg.epochs * 0.5): 0.1})
        return optax.piecewise_constant_schedule(cfg.learning_rate, boundaries_and_scales)
    raise ValueError(f'Unknown scheduler: {cfg.scheduler}')


def build_optimizer(cfg: TrainingConfig):
    lr = build_scheduler(cfg)
    if cfg.optimizer == 'adam':
        return optax.adam(lr)
    if cfg.optimizer == 'adamw':
        weight_decay = float(cfg.scheduler_kwargs.get('weight_decay', 1e-4))
        return optax.adamw(lr, weight_decay=weight_decay)
    if cfg.optimizer == 'radam':
        return optax.radam(lr)
    if cfg.optimizer == 'adamax':
        return optax.adamax(lr)
    if cfg.optimizer == 'rmsprop':
        return optax.rmsprop(lr)
    if cfg.optimizer == 'sgd':
        momentum = float(cfg.scheduler_kwargs.get('momentum', 0.9))
        return optax.sgd(lr, momentum=momentum)
    raise ValueError(f'Unknown optimizer: {cfg.optimizer}')
