import optax


def learning_rate(cfg, batch_size: int):
    rate = cfg.optimizer.lr
    if not cfg.optimizer.get("anneal"):
        return rate
    updates = int(cfg.total_timesteps) // batch_size
    return optax.cosine_decay_schedule(
        rate,
        updates
        * cfg.algorithm.get("update_epochs", 1)
        * cfg.algorithm.get("num_minibatches", 1),
        alpha=cfg.optimizer.get("alpha", 0.0),
    )
