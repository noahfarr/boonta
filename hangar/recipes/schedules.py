import optax


def learning_rate(cfg, batch_size: int):
    rate = cfg.optimizer.lr
    anneal = cfg.optimizer.get("anneal")
    if not anneal:
        return rate
    steps = (
        int(cfg.num_steps)
        // batch_size
        * cfg.algorithm.get("update_epochs", 1)
        * cfg.algorithm.get("num_minibatches", 1)
    )
    if anneal == "linear":
        return optax.linear_schedule(rate, 0.0, steps)
    return optax.cosine_decay_schedule(rate, steps, alpha=cfg.optimizer.get("alpha", 0.0))
