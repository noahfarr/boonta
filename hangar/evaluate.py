from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from hangar import resolvers  # noqa: F401


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "config"),
    config_name="evaluate",
)
def main(cfg):
    from boonta.loggers import MultiLogger
    from boonta.utils import load_checkpoint, resolve_checkpoint
    from hangar import recipes

    assert cfg.checkpoint is not None
    podracer = recipes.make(cfg)
    algorithm, environment = podracer.algorithm, podracer.environment
    logger = MultiLogger(
        [
            instantiate(
                v,
                cfg=OmegaConf.to_container(cfg, resolve=True),
                _recursive_=False,
                _convert_="all",
            )
            for v in (cfg.loggers or {}).values()
        ]
    )
    num_steps = environment.time_limit() + 1
    num_rollouts = -(-int(cfg.num_episodes) // int(cfg.environment.num_envs))

    init_key, evaluate_key, craft_key = jax.random.split(jax.random.key(cfg.seed), 3)
    path = resolve_checkpoint(cfg.checkpoint)
    state = load_checkpoint(path, podracer.init(init_key))
    step = int(jax.device_get(state.algorithm_state.step))

    try:
        rollout_logs = [
            podracer.evaluate(state, rollout_key, num_steps)[1]
            for rollout_key in jax.random.split(evaluate_key, num_rollouts)
        ]
        data = {
            k.replace("episode_statistics/", "evaluation/"): np.nanmean(
                [jax.device_get(logs[k]) for logs in rollout_logs]
            ).reshape(1, 1)
            for k in rollout_logs[0]
            if "/" in k
        }
        artisans = [instantiate(v) for v in (cfg.artisans or {}).values()]
        for key, artisan in zip(
            jax.random.split(craft_key, max(len(artisans), 1)), artisans
        ):
            logger.log_artifact(
                artisan.craft(key, algorithm, environment, state, rollout_logs[-1]),
                step,
            )
        logger.log(data, steps=jnp.array([step, step]))
    finally:
        podracer.close(state)
        logger.finish()


if __name__ == "__main__":
    main()
