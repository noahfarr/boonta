import time
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from hangar import resolvers  # noqa: F401


@hydra.main(version_base=None, config_path=str(Path(__file__).parent / "config"), config_name="train")
def main(cfg):
    if cfg.num_processes > 1:
        jax.distributed.initialize(num_processes=cfg.num_processes)

    from boonta.loggers import MultiLogger
    from boonta.utils import SystemMonitor, load_checkpoint, resolve_checkpoint
    from hangar import recipes
    from hangar.brief import brief

    start = time.monotonic()
    scores = []
    podracer = recipes.make(cfg)
    algorithm, environment = podracer.algorithm, podracer.environment

    num_updates = int(cfg.num_steps) // (podracer.batch_size * cfg.num_epochs)
    assert num_updates >= 1, (
        f"num_steps ({cfg.num_steps}) over {cfg.num_epochs} "
        f"epochs is less than one update ({podracer.batch_size} steps) per epoch. "
        f"Increase num_steps or decrease num_envs, podracer.config.num_steps or num_epochs."
    )

    config = OmegaConf.to_container(cfg, resolve=True)

    logger = MultiLogger(
        [
            instantiate(v, cfg=config, _recursive_=False, _convert_="all")
            for v in (cfg.loggers or {}).values()
        ]
    )

    monitor = SystemMonitor()

    key = jax.random.key(cfg.seed)
    init_key, train_key, craft_key = jax.random.split(key, 3)

    state = load_checkpoint(resolve_checkpoint(cfg.checkpoint), podracer.init(init_key))

    epoch, remainder = divmod(
        int(jax.device_get(state.algorithm_state.step)),
        num_updates * podracer.batch_size,
    )
    assert remainder == 0, (
        f"the checkpoint stopped at step {int(state.algorithm_state.step)}, which is not "
        f"an epoch boundary of this run ({num_updates * podracer.batch_size} steps per epoch)"
    )

    if jax.process_index() == 0:
        brief(cfg, state)

    train_keys = jax.random.split(train_key, cfg.num_epochs)

    artisans = [instantiate(v) for v in (cfg.artisans or {}).values()]
    scoring = instantiate(cfg.scoring)

    def reduce(logs):
        logs = jax.device_get({k: v for k, v in logs.items() if "/" in k})
        return {
            k.replace("episode_statistics/", "training/"): np.nanmean(
                v.reshape(1, -1), axis=1, keepdims=True
            )
            for k, v in logs.items()
        }

    def craft(state, logs, step):
        keys = jax.random.split(jax.random.fold_in(craft_key, step), max(len(artisans), 1))
        for key, artisan in zip(keys, artisans):
            logger.log_artifact(artisan.craft(key, algorithm, environment, state, logs), step)

    try:
        data = {}
        if epoch == 0:
            data = monitor.metrics(step=0)
            logger.log(data, steps=jnp.array([0, 0]))

        for epoch in range(epoch, cfg.num_epochs):
            monitor.start()
            state, logs = podracer.train(state, train_keys[epoch], num_updates)
            data = reduce(logs)
            monitor.stop()

            steps = (
                np.array([epoch, epoch + 1], dtype=np.int64)
                * num_updates
                * podracer.batch_size
            )
            craft(state, logs, int(steps.max()))

            data |= monitor.metrics(step=int(steps.max()))
            logger.log(data, steps=steps)

            returns = data.get(cfg.score)
            if returns is not None and np.isfinite(returns).any():
                scores.append(float(np.nanmean(returns)))
    finally:
        podracer.close(state)
        logger.finish()

    return {
        "score": scoring(scores) if scores else float("-inf"),
        "cost": time.monotonic() - start,
    }


if __name__ == "__main__":
    main()
