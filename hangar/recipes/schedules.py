import optax


def learning_rate(cfg, batch_size: int):
    rate = cfg.optimizer.lr
    updates = (
        int(cfg.num_steps)
        // batch_size
        * cfg.algorithm.get("update_epochs", 1)
        * cfg.algorithm.get("num_minibatches", 1)
    )
    if cfg.optimizer.get("warmup_updates"):
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=rate,
            warmup_steps=cfg.optimizer.warmup_updates,
            decay_steps=updates,
            end_value=cfg.optimizer.get("final_lr", 0.0),
        )
    anneal = cfg.optimizer.get("anneal")
    if not anneal:
        return rate
    if anneal == "linear":
        return optax.linear_schedule(rate, 0.0, updates)
    return optax.cosine_decay_schedule(rate, updates, alpha=cfg.optimizer.get("alpha", 0.0))
