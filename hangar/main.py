import time

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from hangar import resolvers  # noqa: F401


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):
    if cfg.num_processes > 1:
        jax.distributed.initialize(num_processes=cfg.num_processes)

    from boonta.artisans import Metrics
    from boonta.loggers import MultiLogger
    from boonta.utils import SystemMonitor, brief, load_checkpoint, newest
    from hangar import recipes

    start = time.monotonic()
    scores = []
    podracer = recipes.make(cfg)
    algorithm, environment = podracer.algorithm, podracer.environment

    num_updates = int(cfg.total_timesteps) // (
        podracer.batch_size * cfg.training.num_epochs
    )
    assert num_updates >= 1, (
        f"total_timesteps ({cfg.total_timesteps}) over {cfg.training.num_epochs} "
        f"epochs is less than one update ({podracer.batch_size} steps) per epoch. "
        f"Increase total_timesteps or decrease num_envs, num_steps or num_epochs."
    )
    num_steps = num_updates * podracer.batch_size

    config = OmegaConf.to_container(cfg, resolve=True)

    logger = MultiLogger(
        [
            instantiate(v, cfg=config, _recursive_=False, _convert_="all")
            for v in (cfg.loggers or {}).values()
        ]
    )

    monitor = SystemMonitor()

    key = jax.random.key(cfg.seed)
    init_key, train_key, evaluate_key = jax.random.split(key, 3)

    state = load_checkpoint(newest(cfg.get("resume")), podracer.init(init_key))

    if jax.process_index() == 0:
        brief(cfg, state)

    baseline_key, evaluate_key = jax.random.split(evaluate_key)
    train_keys = jax.random.split(train_key, cfg.training.num_epochs)
    evaluate_keys = jax.random.split(evaluate_key, cfg.training.num_epochs)

    artisans = [instantiate(v) for v in (cfg.artisans or {}).values()]
    scoring = instantiate(cfg.scoring)

    def reduce(logs, prefix):
        logs = jax.device_get({k: v for k, v in logs.items() if "/" in k})
        return {
            k.replace("episode_statistics/", prefix): np.nanmean(
                v.reshape(1, -1), axis=1, keepdims=True
            )
            for k, v in logs.items()
        }

    def craft(state, logs, step):
        metrics = {}
        for artisan in artisans:
            artifact = artisan.craft(algorithm, environment, state, logs)
            logger.log_artifact(artifact, step)
            if isinstance(artifact, Metrics):
                metrics |= {k: np.asarray(v).reshape(1, 1) for k, v in artifact.data.items()}
        return metrics

    try:
        data = monitor.metrics(step=0)
        if cfg.evaluation.num_steps:
            _, logs = podracer.evaluate(state, baseline_key, cfg.evaluation.num_steps)
            data |= reduce(logs, "evaluation/")
            data |= craft(state, logs, 0)
        logger.log(data, steps=jnp.array([0, 0]))

        for epoch in range(cfg.training.num_epochs):
            monitor.start()
            state, logs = podracer.train(state, train_keys[epoch], num_updates)
            data = reduce(logs, "training/")
            monitor.stop()

            steps = np.array([epoch, epoch + 1], dtype=np.int64) * num_steps
            if cfg.evaluation.num_steps:
                _, logs = podracer.evaluate(
                    state, evaluate_keys[epoch], cfg.evaluation.num_steps
                )
                data |= reduce(logs, "evaluation/")
            data |= craft(state, logs, int(steps.max()))

            data |= monitor.metrics(step=int(steps.max()))
            logger.log(data, steps=steps)

            returns = data.get(cfg.score)
            if returns is not None and np.isfinite(returns).any():
                scores.append(float(np.nanmean(returns)))

        if cfg.evaluation.num_steps and cfg.score in data:
            logger.log_summary({"score": data[cfg.score].reshape(-1)})
    finally:
        podracer.close(state)
        logger.finish()

    return {
        "score": scoring(scores) if scores else float("-inf"),
        "cost": time.monotonic() - start,
    }


if __name__ == "__main__":
    main()
