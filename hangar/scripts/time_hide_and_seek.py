import argparse
import json
import time

import jax

from boonta.environments import dune_sea
from boonta.environments.dune_sea.hide_and_seek import NUM_ACTIONS
from boonta.environments.wrappers import Vectorize


def milliseconds(function, state, steps, key):
    @jax.jit
    def run(state, key):
        return jax.lax.scan(lambda state, key: (function(state, key), None), state, jax.random.split(key, steps))[0]

    jax.block_until_ready(run(state, key))
    start = time.perf_counter()
    jax.block_until_ready(run(state, key))
    return (time.perf_counter() - start) / steps * 1e3


def components(env, num_envs):
    batch = Vectorize(env, num_envs)
    simulate = jax.vmap(env.simulate)
    sight = jax.vmap(env.sight)
    observe = jax.vmap(env.observe)

    def actions(key):
        return jax.random.randint(key, (num_envs, env.num_agents), 0, NUM_ACTIONS)

    def nudge(state, key):
        circle = state.sim.circle
        moved = circle.replace(position=circle.position + jax.random.uniform(key) * 1e-6)
        return state.replace(sim=state.sim.replace(circle=moved))

    def touch(state, value):
        total = sum(leaf.sum() for leaf in jax.tree.leaves(value))
        return state.replace(time=state.time + (total > 1e9))

    return {
        "physics": lambda state, key: state.replace(sim=simulate(state.sim)[0]),
        "sight": lambda state, key: touch(nudge(state, key), sight(nudge(state, key))),
        "observe": lambda state, key: touch(nudge(state, key), observe(nudge(state, key))),
        "step": lambda state, key: batch.step(key, state, actions(key))[0],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kwargs", default="{}")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    env = dune_sea.make("hide_and_seek", **json.loads(args.kwargs))
    state, _ = Vectorize(env, args.num_envs).init(jax.random.key(0))
    print(
        f"hide_and_seek on {jax.devices()[0].platform}, {args.num_envs} envs, "
        f"{env.num_agents} agents, {env.num_objects} objects, {env.substeps} substeps"
    )
    results = {
        name: milliseconds(function, state, args.steps, jax.random.key(1))
        for name, function in components(env, args.num_envs).items()
    }
    plain = results["step"]
    for name, duration in results.items():
        print(f"{duration:9.2f} ms  {duration / plain:6.2f}x step  {name}")
    print(f"{args.num_envs / plain * 1e3:,.0f} environment steps per second")
