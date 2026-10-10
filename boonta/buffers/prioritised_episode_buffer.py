import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flashbax import utils
from flashbax.buffers import sum_tree
from flashbax.buffers.prioritised_trajectory_buffer import (
    SET_BATCH_FN,
    PrioritisedTrajectoryBuffer,
    PrioritisedTrajectoryBufferSample,
    PrioritisedTrajectoryBufferState,
    prioritised_init,
    set_priorities,
    validate_device,
    validate_priority_exponent,
)
from flashbax.buffers.trajectory_buffer import Experience, can_sample

from boonta.utils import Array, Key

from .episode_buffer import (candidates, episode_starts, expand, startable,
                             validate, window)


def importance_weights(probabilities: Array, size: int, beta: float = 0.4) -> Array:
    weights = (size * jnp.maximum(probabilities, 1e-10)) ** (-beta)
    return weights / jnp.maximum(weights.max(), 1e-10)


def sample(
    state: PrioritisedTrajectoryBufferState,
    key: Key,
    sample_batch_size: int,
    sample_sequence_length: int,
    get_start_flags: Callable[[Experience], Array],
) -> PrioritisedTrajectoryBufferSample:
    size, time = utils.get_tree_shape_prefix(state.experience, n_axes=2)
    windows = candidates(
        get_start_flags(state.experience), startable(state, sample_sequence_length)
    )
    priorities = sum_tree.get(state.sum_tree_state, jnp.arange(size * time))
    weights = priorities * windows
    weights = jnp.where(weights.sum() > 0, weights, windows)
    probabilities = weights / weights.sum()
    flat = jax.random.choice(key, size * time, (sample_batch_size,), p=probabilities)
    return PrioritisedTrajectoryBufferSample(
        experience=window(
            state.experience, flat // time, flat % time, sample_sequence_length
        ),
        indices=flat,
        probabilities=jnp.take(probabilities, flat),
    )


def add(
    state: PrioritisedTrajectoryBufferState, batch: Experience, device: str
) -> PrioritisedTrajectoryBufferState:
    _, length = utils.get_tree_shape_prefix(batch, n_axes=2)
    size, time = utils.get_tree_shape_prefix(state.experience, n_axes=2)
    assert length <= time, (
        f"cannot add {length} steps at once to rows that hold {time}: the write "
        f"would overwrite itself"
    )
    positions = (jnp.arange(length) + state.current_index) % time
    experience = jax.tree.map(
        lambda stored, added: stored.at[:, positions].set(added),
        state.experience,
        batch,
    )

    fresh = (positions[None, :] + jnp.arange(size)[:, None] * time).reshape(-1)
    tree = SET_BATCH_FN[device](
        state.sum_tree_state,
        fresh,
        jnp.full(fresh.shape, state.sum_tree_state.max_recorded_priority),
    )

    index = state.current_index + length
    return state.replace(
        experience=experience,
        current_index=index % time,
        is_full=state.is_full | (index >= time),
        running_index=state.running_index + length,
        sum_tree_state=tree,
    )


def make_prioritised_episode_buffer(
    max_length: int,
    min_length: int,
    sample_batch_size: int,
    sample_sequence_length: int,
    get_start_flags: Callable[[Experience], Array] = episode_starts,
    add_sequences: bool = False,
    add_batch_size: int | None = None,
    priority_exponent: float = 0.6,
    device: str = "cpu",
) -> PrioritisedTrajectoryBuffer:
    add_batches = add_batch_size is not None
    rows = add_batch_size or 1
    validate(max_length, min_length, sample_sequence_length, rows)
    validate_priority_exponent(priority_exponent)
    if not validate_device(device):
        device = "cpu"
    time = max_length // rows

    return PrioritisedTrajectoryBuffer(
        init=functools.partial(
            prioritised_init, add_batch_size=rows, max_length_time_axis=time, period=1
        ),
        add=expand(functools.partial(add, device=device), add_sequences, add_batches),
        sample=functools.partial(
            sample,
            sample_batch_size=sample_batch_size,
            sample_sequence_length=sample_sequence_length,
            get_start_flags=get_start_flags,
        ),
        can_sample=functools.partial(
            can_sample,
            min_length_time_axis=max(min_length // rows, sample_sequence_length),
        ),
        set_priorities=functools.partial(
            set_priorities, priority_exponent=priority_exponent, device=device
        ),
    )
