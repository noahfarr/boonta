import argparse
import json
import time

import jax
import jax.numpy as jnp

from boonta.environments import dune_sea
from boonta.environments.wrappers import OptimisticAutoReset, Vectorize


def milliseconds(function, state, steps, key):
    @jax.jit
    def run(state, key):
        def body(total, key):
            nudge = jax.random.uniform(key) * 1e-6
            moved = state.replace(agent=state.agent.replace(energy=state.agent.energy + nudge))
            return total + function(moved, key), None

        return jax.lax.scan(body, 0.0, jax.random.split(key, steps))[0]

    jax.block_until_ready(run(state, key))
    start = time.perf_counter()
    jax.block_until_ready(run(state, key))
    return (time.perf_counter() - start) / steps * 1e3


def components(env, num_envs, ratio):
    batch = Vectorize(env, num_envs)
    pooled = OptimisticAutoReset(env, num_envs, ratio)
    observe = jax.vmap(env.observe)
    move = jax.vmap(env.move)
    interact = jax.vmap(env.interact)
    sight = jax.vmap(lambda state, positions: env.sight(state, positions, state.cells.wall, state.cells.bush))

    def actions(key):
        return jax.random.randint(key, (num_envs, env.num_agents), 0, env.num_actions)

    def order(key):
        return jax.vmap(lambda key: jax.random.permutation(key, env.num_agents))(jax.random.split(key, num_envs))

    def moved(state, key):
        return move(state, actions(key), order(key))

    def stepped(wrapper):
        def run(state, key):
            state, timestep = wrapper.step(key, state, actions(key))
            return timestep.obs["view"].sum() + state.cells.wall.sum()

        return run

    def interacted(state, key):
        cells, energy, inventory = interact(state, actions(key), state.agent.position, order(key))
        return energy.sum() + cells.wall.sum()

    def generated(state, key):
        state, timestep = batch.init(key)
        return timestep.obs["view"].sum() + state.cells.wall.sum()

    parts = {
        "observe": lambda state, key: observe(state)["view"].sum(),
        "move": lambda state, key: moved(state, key).sum(),
        "interact": interacted,
        "map generation (all envs)": generated,
        "step": stepped(batch),
        f"step with pooled reset (ratio {ratio})": stepped(pooled),
    }
    if env.sight_drain > 0:
        parts["sight"] = lambda state, key: sight(state, state.agent.position).sum()
    return parts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="hide_and_seek")
    parser.add_argument("--kwargs", default="{}")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--ratio", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    env = dune_sea.make(args.env, **json.loads(args.kwargs))
    state, _ = Vectorize(env, args.num_envs).init(jax.random.key(0))
    print(f"{args.env} on {jax.devices()[0].platform}, {args.num_envs} envs, {env.num_agents} agents")
    results = {
        name: milliseconds(function, state, args.steps, jax.random.key(1))
        for name, function in components(env, args.num_envs, args.ratio).items()
    }
    plain = results["step"]
    for name, duration in results.items():
        print(f"{duration:9.2f} ms  {duration / plain:6.2f}x step  {name}")
