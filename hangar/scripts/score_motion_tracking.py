import os
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from rich.console import Console
from rich.table import Table

from hangar import resolvers  # noqa: F401


@struct.dataclass
class Policy:
    params: dict


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parents[1] / "config"),
    config_name="evaluate",
)
def main(cfg):
    import orbax.checkpoint as ocp

    from boonta.environments.mjlab import motion_tracking
    from boonta.utils import resolve_checkpoint
    from hangar import recipes

    assert cfg.checkpoint is not None
    assert cfg.environment.kwargs.play, "score with environment.kwargs.play=true"
    podracer = recipes.make(cfg)
    algorithm, environment = podracer.algorithm, podracer.environment
    simulator = environment.unwrapped.env
    library = np.load(simulator.cfg.commands["motion"].motion_file)
    starts, ends = library["clip_start"], library["clip_end"]
    num_clips, num_envs = len(starts), int(cfg.environment.num_envs)
    assert num_envs >= num_clips, f"set environment.num_envs to at least the {num_clips} clips"
    path = Path(resolve_checkpoint(cfg.checkpoint)) / "algorithm_state"
    params = ocp.StandardCheckpointer().restore(os.path.abspath(str(path)))["params"]
    policy = Policy(params=params)
    num_steps = int(np.max(ends - starts))

    @jax.jit
    def play(policy, key):
        reset_key, rollout_key = jax.random.split(key)
        environment_state, timestep = environment.init(reset_key)
        history = jnp.zeros((num_envs, algorithm.cfg.history, *jnp.shape(algorithm.observation_of(timestep))[1:]))

        def step_fn(carry, key):
            environment_state, timestep, history, returned, length, failed, playing = carry
            act_key, step_key = jax.random.split(key)
            _, history, action, _ = algorithm.step(policy, history, act_key, timestep, temperature=0.0)
            environment_state, timestep = environment.step(step_key, environment_state, action)
            returned = returned + playing * timestep.reward
            length = length + playing
            failed = failed | (playing & timestep.terminated)
            playing = playing & ~timestep.done
            return (environment_state, timestep, history, returned, length, failed, playing), None

        zeros = jnp.zeros(num_envs)
        carry = (environment_state, timestep, history, zeros, zeros, zeros.astype(bool), ~zeros.astype(bool))
        (_, _, _, returned, length, failed, _), _ = jax.lax.scan(
            step_fn, carry, jax.random.split(rollout_key, num_steps)
        )
        return returned, length, failed

    returned, length, failed = jax.device_get(play(policy, jax.random.key(cfg.seed)))
    ceiling = motion_tracking.attainable(simulator.cfg) * (ends - starts - 1)
    normalized = returned[:num_clips] / ceiling
    names = library["names"]
    table = Table("clip", "normalized return", "steps", "failed")
    for name, score, steps, fell in zip(names, normalized, length[:num_clips], failed[:num_clips]):
        table.add_row(name, f"{score:.3f}", f"{int(steps)}", "yes" if fell else "")
    console = Console()
    console.print(table)
    console.print(
        f"normalized return {np.mean(normalized):.3f} over {num_clips} clips, "
        f"{int(np.sum(failed[:num_clips]))} failed"
    )


if __name__ == "__main__":
    main()
