import optax


def learning_rate(cfg, batch_size: int):
    rate = cfg.optimizer.lr
    if not cfg.optimizer.get("anneal"):
        return rate
    updates = int(cfg.total_timesteps) // batch_size
    return optax.cosine_decay_schedule(
        rate,
        updates * cfg.algorithm.update_epochs * cfg.algorithm.num_minibatches,
        alpha=cfg.optimizer.get("min_lr_ratio", 0.0),
    )


def warmup_cosine(peak, steps: int, warmup: int = 0, start: float = 0.0, alpha: float = 0.0):
    return optax.warmup_cosine_decay_schedule(
        init_value=start,
        peak_value=peak,
        warmup_steps=warmup,
        decay_steps=max(steps, warmup + 1),
        end_value=peak * alpha,
    )
