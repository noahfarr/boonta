from collections.abc import Callable

import jax
import jax.numpy as jnp
import lox

from boonta.algorithms import Algorithm
from boonta.podracers.podracer import Lap, Pit
from boonta.utils import Key, PyTree
from boonta.utils.typing import Environment

from .plr import Graded, LevelBuffer, LevelBufferState, plr


def accel(
    algorithm: Algorithm,
    environment: Environment,
    num_edits: int,
    mutate: Callable[[Key, PyTree, int], PyTree],
    **kwargs,
) -> tuple[Graded, LevelBuffer, Pit, Lap]:
    algorithm, environment, curate, lap = plr(algorithm, environment, **kwargs)
    capacity = environment.capacity
    staging = environment.staging

    def pit(state):
        curated = curate(state)
        environment_state = jax.tree.map(
            mutate_replayed,
            curated.environment_state,
            state.environment_state,
            is_leaf=lambda node: isinstance(node, LevelBufferState),
        )
        return curated.replace(environment_state=environment_state)

    def mutate_replayed(buffer, finished):
        if not isinstance(buffer, LevelBufferState):
            return buffer
        replayed = ~finished.exploring & jnp.all(finished.playing >= 0)
        lox.log({"accel/levels/mutating": replayed.astype(jnp.float32)})
        return jax.lax.cond(
            replayed,
            lambda: stage_children(buffer, finished),
            lambda: buffer,
        )

    def stage_children(buffer, finished):
        parents = jax.tree.map(lambda leaf: leaf[finished.playing], buffer.theta)
        keys = jax.random.split(jax.random.fold_in(finished.key, 1), staging)
        children = jax.vmap(mutate, (0, 0, None))(keys, parents, num_edits)
        levels = jax.tree.map(
            lambda leaf, child: leaf.at[capacity:].set(child), buffer.theta, children
        )
        return environment.update(
            buffer.replace(
                exploring=jnp.bool_(True),
                timestamps=finished.timestamps,
                episodes=finished.episodes,
            ),
            theta=levels,
            assign=capacity + jnp.arange(staging, dtype=jnp.int32),
        )

    return algorithm, environment, pit, lap
