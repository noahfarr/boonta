import argparse
import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from render_checkpoint import load
from signal_map import measure, policy_table

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
        return self.selector.score

    def update(self, state, key, transitions):
        return state

    def select(self, state, wrapper, archive_auto_reset_state, key, **context):
        if self.num_placed == 0:
            return jnp.zeros(0, jnp.int32)
        archive_state = archive_auto_reset_state.archive_state
        cells = jnp.squeeze(archive_state.keys, axis=-1)
        weights = jnp.where(archive_state.mask, self.cell_weights[jnp.maximum(cells, 0)] + 1e-12, 0.0)
        return jax.random.categorical(key, jnp.log(weights), shape=(self.num_placed,))


@dataclasses.dataclass
class Exploring:
    num_placed: int
    selector: object

    def init(self, archive_size, num_actions):
        return self.selector.init(archive_size, num_actions)

    @property
    def score(self):
        return self.selector.score

    def update(self, state, key, transitions):
        return state

    def select(self, state, wrapper, archive_auto_reset_state, key, **context):
        from boonta.environments.wrappers.archive_auto_reset import pick

        eligible_mask = wrapper.archive.eligible(archive_auto_reset_state.archive_state)
        return pick(key, jnp.where(eligible_mask, 0.0, -jnp.inf), (self.num_placed,))


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


def episode_returns(algorithm, algorithm_state, game, num_episodes, key):
    reset_key, run_key = jax.random.split(key)
    states, timesteps = jax.vmap(game.init)(jax.random.split(reset_key, num_episodes))

    def body(carry, key):
        states, timesteps, alive, total = carry
        act_key, step_key = jax.random.split(key)
        _, _, actions, _ = algorithm.step(algorithm_state, None, act_key, timesteps, temperature=1.0)
        states, timesteps = jax.vmap(game.step)(jax.random.split(step_key, num_episodes), states, actions)
        total = total + alive * timesteps.reward
        return (states, timesteps, alive & ~timesteps.done, total), None

    carry = (states, timesteps, jnp.ones(num_episodes, bool), jnp.zeros(num_episodes))
    (*_, total), _ = jax.lax.scan(body, carry, jax.random.split(run_key, game.time_limit()))
    return jnp.mean(total)


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

    selector = podracer.pit.selector
    num_placed = selector.num_placed(environment.num_envs)
    placements = {"rho": Placement(jnp.ones(1), 0, selector), "prd": selector, "erd": Exploring(num_placed, selector)}
    if hasattr(game, "place"):
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
        for name in ("signal", "alignment"):
            weights = jnp.asarray(np.maximum(maps[name], 0.0))
            placements[f"{name} oracle"] = Placement(weights, num_placed, selector)
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

    if hasattr(game, "expected_return"):
        tabulate = jax.jit(lambda algorithm_state: policy_table(algorithm, algorithm_state, game, measure_key)[2])
        print("J_rho is exact: the game evaluates the policy itself")

        def score(algorithm_state):
            return float(game.expected_return(tabulate(algorithm_state)))

    else:
        play = jax.jit(lambda algorithm_state: episode_returns(algorithm, algorithm_state, game, arguments.episodes, measure_key))
        print(f"J_rho is the mean return of {arguments.episodes} episodes from rho, the same episodes for every arm")

        def score(algorithm_state):
            return float(play(algorithm_state))

    begin = jax.jit(environment.init)
    before = score(checkpoint.algorithm_state)
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
            improvements[name].append(score(state.algorithm_state) - before)

    print(f"J_rho before the update: {before:.4f}")
    print(f"{'arm':<18} {'J_rho change':>22} {'minus rho':>22}")
    rho = np.asarray(improvements["rho"])
    for name, values in improvements.items():
        values = np.asarray(values)
        print(f"{name:<18} {interval(values):>22} {'' if name == 'rho' else interval(values - rho):>22}")


if __name__ == "__main__":
    main()
