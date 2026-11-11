from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from hangar import resolvers  # noqa: F401


def load_evaluation_checkpoint(path, target):
    """Restore inference weights without loading SuperMimic's replay state."""
    algorithm_state = getattr(target, "algorithm_state", None)
    if algorithm_state is None or not hasattr(algorithm_state, "buffer_state"):
        from boonta.utils import load_checkpoint

        return load_checkpoint(path, target)

    path = Path(path)
    algorithm_directory = path / "algorithm_state"
    if not (algorithm_directory / "_METADATA").exists():
        from boonta.utils import load_checkpoint

        return load_checkpoint(path, target)

    from boonta.utils.checkpoint import restore_item

    ignored = (
        "actor_optimizer_state",
        "critic_optimizer_state",
        "world_model_optimizer_state",
        "buffer_state",
    )

    def empty_targets(tree):
        return jax.tree.map(
            lambda leaf: jnp.empty((0,), dtype=leaf.dtype)
            if hasattr(leaf, "dtype")
            else leaf,
            tree,
            is_leaf=lambda leaf: leaf is None,
        )

    partial_target = algorithm_state.replace(
        **{name: empty_targets(getattr(algorithm_state, name)) for name in ignored}
    )
    restored = restore_item(str(algorithm_directory), partial_target)
    algorithm_state = restored.replace(
        **{name: getattr(algorithm_state, name) for name in ignored}
    )
    return target.replace(algorithm_state=algorithm_state)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "config"),
    config_name="evaluate",
)
def main(cfg):
    from boonta.loggers import MultiLogger
    from boonta.utils import resolve_checkpoint
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
    state = load_evaluation_checkpoint(path, podracer.init(init_key))
    step = int(jax.device_get(state.algorithm_state.step))

    try:
        _, logs = zip(
            *(
                podracer.evaluate(state, rollout_key, num_steps)
                for rollout_key in jax.random.split(evaluate_key, num_rollouts)
            )
        )
        logs = jax.tree.map(lambda *values: np.stack(values), *logs)
        data = {
            name.replace("episode_statistics/", "evaluation/"): np.nanmean(
                value
            ).reshape(1, 1)
            for name, value in logs.items()
        }
        artisans = [instantiate(v) for v in (cfg.artisans or {}).values()]
        for key, artisan in zip(
            jax.random.split(craft_key, max(len(artisans), 1)), artisans
        ):
            logger.log_artifact(
                artisan.craft(key, algorithm, environment, state, logs), step
            )
        logger.log(data, steps=jnp.array([step, step]))
    finally:
        podracer.close(state)
        logger.finish()


if __name__ == "__main__":
    main()
