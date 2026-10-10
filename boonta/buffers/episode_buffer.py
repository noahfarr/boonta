from collections.abc import Callable

import jax
import jax.numpy as jnp
from flashbax import utils
from flashbax.buffers.trajectory_buffer import (
    Experience,
    TrajectoryBuffer,
    TrajectoryBufferSample,
    TrajectoryBufferState,
    make_trajectory_buffer,
)
from flashbax.utils import add_dim_to_args

from boonta.utils import Array, Key


def episode_starts(experience: Experience) -> Array:
    return experience.first.done


def every_step(experience: Experience) -> Array:
    return jnp.ones_like(experience.first.done)


def validate(
    max_length: int,
    min_length: int,
    sample_sequence_length: int,
    add_batch_size: int,
) -> None:
    time = max_length // add_batch_size
    assert time >= 2, (
        f"max_length // add_batch_size is {max_length} // {add_batch_size} = {time}; "
        f"each row needs at least 2 steps"
    )
    assert min_length // add_batch_size <= time, (
        f"min_length ({min_length}) does not fit in rows of {time} steps"
    )
    assert sample_sequence_length <= time, (
        f"sample_sequence_length ({sample_sequence_length}) exceeds the {time} steps "
        f"each row holds"
    )


def startable(state: TrajectoryBufferState, sample_sequence_length: int) -> Array:
    _, time = utils.get_tree_shape_prefix(state.experience, n_axes=2)
    positions = jnp.arange(time)
    filled = positions + sample_sequence_length <= state.current_index
    wrapped = (positions - state.current_index) % time <= time - sample_sequence_length
    return jnp.where(state.is_full, wrapped, filled)


def candidates(flags: Array, valid: Array) -> Array:
    flagged = flags.astype(bool) & valid
    windows = jnp.where(flagged.any(), flagged, valid)
    return jnp.where(windows.any(), windows, True).reshape(-1).astype(jnp.float32)


def window(
    experience: Experience, rows: Array, starts: Array, sample_sequence_length: int
) -> Experience:
    def gather(leaf):
        size, time, *rest = leaf.shape
        positions = (starts[:, None] + jnp.arange(sample_sequence_length)) % time
        flat = leaf.reshape(size * time, *rest)
        return jnp.take(flat, rows[:, None] * time + positions, axis=0)

    return jax.tree.map(gather, experience)


def expand(add, add_sequences: bool, add_batches: bool):
    if not add_batches:
        add = add_dim_to_args(add, axis=0, starting_arg_index=1, ending_arg_index=2)
    if not add_sequences:
        time_axis = 0
        if add_batches:
            time_axis = 1
        add = add_dim_to_args(
            add, axis=time_axis, starting_arg_index=1, ending_arg_index=2
        )
    return add


def make_episode_buffer(
    max_length: int,
    min_length: int,
    sample_batch_size: int,
    sample_sequence_length: int,
    get_start_flags: Callable[[Experience], Array] = episode_starts,
    add_sequences: bool = False,
    add_batch_size: int | None = None,
) -> TrajectoryBuffer:
    add_batches = add_batch_size is not None
    rows = add_batch_size or 1
    validate(max_length, min_length, sample_sequence_length, rows)

    buffer = make_trajectory_buffer(
        max_length_time_axis=max_length // rows,
        min_length_time_axis=max(min_length // rows, sample_sequence_length),
        add_batch_size=rows,
        sample_batch_size=sample_batch_size,
        sample_sequence_length=sample_sequence_length,
        period=1,
        max_size=None,
    )

    def sample(state: TrajectoryBufferState, key: Key) -> TrajectoryBufferSample:
        size, time = utils.get_tree_shape_prefix(state.experience, n_axes=2)
        weights = candidates(
            get_start_flags(state.experience), startable(state, sample_sequence_length)
        )
        flat = jax.random.choice(
            key, size * time, (sample_batch_size,), p=weights / weights.sum()
        )
        return TrajectoryBufferSample(
            experience=window(
                state.experience, flat // time, flat % time, sample_sequence_length
            )
        )

    return buffer.replace(
        add=expand(buffer.add, add_sequences, add_batches), sample=sample
    )
