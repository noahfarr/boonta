import argparse
import os
import shutil
import subprocess
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from render_checkpoint import load

from hangar import resolvers  # noqa: F401

GAP = 20


def discounted_returns(algorithm, algorithm_state, game, gamma, states, actions, key):
    num_rollouts, *_ = actions.shape

    def body(carry, key):
        states, actions, alive, discount, total = carry
        step_key, act_key = jax.random.split(key)
        states, timesteps = jax.vmap(game.step)(jax.random.split(step_key, num_rollouts), states, actions)
        total = total + alive * discount * timesteps.reward
        alive = alive & ~timesteps.done
        _, _, actions, _ = algorithm.step(algorithm_state, None, act_key, timesteps, temperature=1.0)
        return (states, actions, alive, discount * gamma, total), None

    start = (states, actions, jnp.ones(num_rollouts, bool), jnp.float32(1.0), jnp.zeros(num_rollouts))
    (*_, total), _ = jax.lax.scan(body, start, jax.random.split(key, game.time_limit()))
    return total


def signal(algorithm, algorithm_state, game, gamma, num_actions, num_rollouts, key):
    start, _ = game.init(key)
    cells = jnp.arange(game.num_cells)
    placed = jax.vmap(game.place, in_axes=(None, 0))(start, cells)
    observed = jax.vmap(game.observe)(placed)
    timesteps = jax.vmap(lambda obs: game.init(key)[1].replace(obs=obs))(observed)
    _, _, _, aux = algorithm.step(algorithm_state, None, key, timesteps, temperature=1.0)
    policy = jax.nn.softmax(aux["logits"], axis=-1)

    repeat = num_actions * num_rollouts
    states = jax.tree.map(lambda leaf: jnp.repeat(leaf, repeat, axis=0), placed)
    actions = jnp.tile(jnp.repeat(jnp.arange(num_actions), num_rollouts), game.num_cells)
    returns = discounted_returns(algorithm, algorithm_state, game, gamma, states, actions, key)
    halves = returns.reshape(game.num_cells, num_actions, 2, num_rollouts // 2).mean(axis=-1)
    values = jnp.moveaxis(halves, -1, 0)
    first, second = values - jnp.sum(policy * values, axis=-1, keepdims=True)
    return jnp.sum(policy * first * second, axis=-1)


def reach(algorithm, algorithm_state, game, num_episodes, key):
    start, timestep = jax.vmap(game.init)(jax.random.split(key, num_episodes))

    def body(carry, key):
        states, timesteps, alive, visits = carry
        act_key, step_key = jax.random.split(key)
        visits = visits + jnp.sum(alive[:, None] * timesteps.obs, axis=0)
        _, _, actions, _ = algorithm.step(algorithm_state, None, act_key, timesteps, temperature=1.0)
        states, timesteps = jax.vmap(game.step)(jax.random.split(step_key, num_episodes), states, actions)
        return (states, timesteps, alive & ~timesteps.done, visits), None

    carry = (start, timestep, jnp.ones(num_episodes, bool), jnp.zeros(game.num_cells))
    (*_, visits), _ = jax.lax.scan(body, carry, jax.random.split(key, game.time_limit()))
    return visits / num_episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--target", help="checkpoint of the policy being updated, or init (default: the run's latest)")
    parser.add_argument("--behavior", help="checkpoint of the policy that collects the data, or init (default: the target)")
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=4096)
    parser.add_argument("--tile", type=int, default=16)
    arguments = parser.parse_args()
    run = Path(arguments.run).resolve()
    cfg = load(run, arguments.overrides)

    from boonta.artisans.concentration import lorenz
    from boonta.artisans.overlay import paint
    from boonta.utils import load_checkpoint, resolve_checkpoint
    from hangar import recipes

    podracer = recipes.make(cfg)
    algorithm, game = podracer.algorithm, podracer.environment.unwrapped
    init_key, signal_key, reach_key = jax.random.split(jax.random.key(cfg.seed), 3)
    template = podracer.init(init_key)

    def policy(checkpoint):
        if checkpoint == "init":
            return template.algorithm_state
        path = Path(resolve_checkpoint(checkpoint or run)) / "algorithm_state"
        return load_checkpoint(path, template).algorithm_state

    target = policy(arguments.target)
    behavior = policy(arguments.behavior or arguments.target)

    gamma = float(cfg.algorithm.gamma)
    num_actions = game.action_space().num_actions
    per_cell_signal = np.asarray(
        jax.jit(partial(signal, algorithm, target, game, gamma, num_actions, arguments.rollouts))(signal_key)
    )
    visits = np.asarray(
        jax.jit(partial(reach, algorithm, behavior, game, arguments.episodes))(reach_key)
    )
    seen = visits * per_cell_signal

    print(f"target signal on behavior-reached states: {seen.sum():.4g} of {per_cell_signal.sum():.4g} unweighted")
    if np.maximum(seen, 0.0).sum() > 1e-3 * visits.sum():
        steps, carried = lorenz(visits, np.maximum(seen, 0.0))
        print(f"share of behavior steps carrying half the signal: {np.interp(0.5, carried, steps):.4f}")
        print(f"share carrying ninety percent: {np.interp(0.9, carried, steps):.4f}")
    else:
        print("no signal left above the estimation noise")
    for cell in np.argsort(-per_cell_signal)[:5]:
        print(f"  cell {cell}: signal {per_cell_signal[cell]:.4g}, visits per episode {visits[cell]:.4g}")

    tile = arguments.tile
    start, _ = game.init(init_key)
    render = np.asarray(game.render(start, tile=tile), np.uint8)
    cell_grid = np.asarray(game.cell_grid)
    num_rows, *_ = render.shape
    gap = np.full((num_rows, tile // 2, 3), GAP, np.uint8)
    panels = [render, *(paint(values, cell_grid, tile) for values in (visits, per_cell_signal, seen))]
    image = np.concatenate([part for panel in panels for part in (gap, panel)][1:], axis=1)
    path = run / "signal_map.png"
    Image.fromarray(image).save(path)
    print(f"render | behavior reach | target signal | reach x signal  ->  {path}")
    if shutil.which("xdg-open") and os.environ.get("DISPLAY"):
        subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
