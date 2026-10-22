import argparse
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from render_checkpoint import load
from signal_map import measure

from hangar import resolvers  # noqa: F401


@dataclasses.dataclass
class Placement:
    cell_weights: jax.Array
    num_placed: int
    selector: object

    def init(self, archive_size, num_actions):
        return self.selector.init(archive_size, num_actions)

    @property
    def score(self):
        from boonta.curricula.prd import Diagonal

        return Diagonal()

    def update(self, state, key, transitions):
        return state

    def select(self, state, wrapper, archive_auto_reset_state, key, **context):
        archive_state = archive_auto_reset_state.archive_state
        cells = jnp.squeeze(archive_state.keys, axis=-1)
        weights = jnp.where(archive_state.mask, self.cell_weights[jnp.maximum(cells, 0)] + 1e-12, 0.0)
        return jax.random.categorical(key, jnp.log(weights), shape=(self.num_placed,))


def transplant(fresh, saved):
    from boonta.environments.wrappers.archive_auto_reset import ArchiveAutoResetState

    def is_archive(node):
        return isinstance(node, ArchiveAutoResetState)

    archive_state = next(
        node.archive_state for node in jax.tree.leaves(saved, is_leaf=is_archive) if is_archive(node)
    )
    return jax.tree.map(
        lambda node: node.replace(archive_state=archive_state) if is_archive(node) else node,
        fresh,
        is_leaf=is_archive,
    )


def burn_in(arm, state, key, num_rollouts):
    from functools import partial

    def rollout(state, key):
        rollout_key, call_key = jax.random.split(key)
        steps = jax.random.split(rollout_key, arm.config.num_steps)
        state, _ = jax.lax.scan(partial(arm.rollout, temperature=1.0), state, steps)
        return arm.pit(call_key, state), None

    state, _ = jax.lax.scan(rollout, state, jax.random.split(key, num_rollouts))
    return state


def interval(values):
    mean = float(np.mean(values))
    half = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values))
    return f"{mean:+.4f} ± {half:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--checkpoint", help="checkpoint to update from (default: the run's latest)")
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--rollouts", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=4096)
    parser.add_argument("--burn-in", type=int, help="rollouts each arm runs before the update (default: horizon / rollout length)")
    arguments = parser.parse_args()
    run = Path(arguments.run).resolve()
    trained_with = OmegaConf.load(run / ".hydra" / "config.yaml")
    cfg = load(run, [f"environment.num_envs={trained_with.environment.num_envs}", *arguments.overrides])

    from boonta.environments.dune_sea.hotspots import Scorecard
    from boonta.podracers import anakin
    from boonta.podracers.anakin import AnakinState
    from boonta.utils import load_checkpoint, resolve_checkpoint
    from hangar import recipes

    podracer = recipes.make(cfg)
    algorithm, environment = podracer.algorithm, podracer.environment
    game = environment.unwrapped
    measure_key, *repeat_keys = jax.random.split(jax.random.key(cfg.seed), arguments.repeats + 1)
    checkpoint = load_checkpoint(resolve_checkpoint(arguments.checkpoint or run), podracer.init(measure_key))
    print(f"checkpoint at step {int(checkpoint.algorithm_state.step):,}")

    signal_key, reach_key = jax.random.split(measure_key)
    maps = measure(
        algorithm,
        checkpoint.algorithm_state,
        checkpoint.algorithm_state,
        game,
        float(cfg.algorithm.gamma),
        arguments,
        signal_key,
        reach_key,
    )
    num_placed = podracer.pit.selector.num_placed(environment.num_envs)
    selector = podracer.pit.selector
    placements = {
        "rho": Placement(jnp.ones(game.num_cells), 0, selector),
        "prd": selector,
        "signal oracle": Placement(jnp.asarray(np.maximum(maps["signal"], 0.0)), num_placed, selector),
        "alignment oracle": Placement(jnp.asarray(np.maximum(maps["alignment"], 0.0)), num_placed, selector),
    }
    arms = {
        name: anakin.make(podracer.config, algorithm, environment, dataclasses.replace(podracer.pit, selector=selector))
        for name, selector in placements.items()
    }
    num_rollouts = arguments.burn_in or max(game.time_limit() // podracer.config.num_steps, 1)
    settle = {
        name: jax.jit(lambda state, key, arm=arm: burn_in(arm, arm.pit(key, state), jax.random.fold_in(key, 1), num_rollouts))
        for name, arm in arms.items()
    }
    print(f"{num_placed} of {environment.num_envs} environments restart per rollout in every arm but rho")
    print(f"each arm runs {num_rollouts} rollouts without updating before the measured update")

    scorecard = Scorecard()

    def score(algorithm_state, key):
        played = scorecard.craft(
            key, SimpleNamespace(algorithm=algorithm, environment=environment), SimpleNamespace(algorithm_state=algorithm_state), {}
        )
        return float(played.data["hotspots/return"])

    begin = jax.jit(environment.init)
    before = score(checkpoint.algorithm_state, measure_key)
    improvements = {name: [] for name in arms}
    for repeat_key in repeat_keys:
        environment_key, burn_key, train_key = jax.random.split(repeat_key, 3)
        environment_state, timestep = begin(environment_key)
        for name, arm in arms.items():
            state = AnakinState(
                timestep=timestep,
                environment_state=transplant(environment_state, checkpoint.environment_state),
                carry=None,
                algorithm_state=checkpoint.algorithm_state,
                pit_state=checkpoint.pit_state,
            )
            state = jax.tree.map(jnp.copy, settle[name](state, burn_key))
            state = state.replace(algorithm_state=jax.tree.map(jnp.copy, checkpoint.algorithm_state))
            state, _ = arm.train(state, train_key, 1)
            improvements[name].append(score(state.algorithm_state, measure_key) - before)

    print(f"J_rho before the update: {before:.4f} of {score_best(game):.4f} possible")
    print(f"{'arm':<18} {'J_rho change':>22} {'minus rho':>22}")
    rho = np.asarray(improvements["rho"])
    for name, values in improvements.items():
        values = np.asarray(values)
        print(f"{name:<18} {interval(values):>22} {'' if name == 'rho' else interval(values - rho):>22}")


def score_best(game):
    from boonta.environments.dune_sea.hotspots import reach

    return float(reach(game).sum())


if __name__ == "__main__":
    main()
