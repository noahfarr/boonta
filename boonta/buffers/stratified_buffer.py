import math
from functools import partial

import flashbax as fbx
import jax
from flashbax.buffers.trajectory_buffer import (TrajectoryBuffer,
                                                TrajectoryBufferSample,
                                                TrajectoryBufferState)
from flashbax.buffers.trajectory_buffer import sample as sample_rows

from boonta.utils.typing import Key


def sample(
    state: TrajectoryBufferState,
    key: Key,
    batch_size: int,
    sequence_length: int,
    period: int,
) -> TrajectoryBufferSample:
    rows = jax.tree.leaves(state.experience)[0].shape[0]
    blocks = math.gcd(batch_size, rows)
    experience = jax.tree.map(
        lambda leaf: leaf.reshape(blocks, rows // blocks, *leaf.shape[1:]),
        state.experience,
    )

    def block(experience, key):
        return sample_rows(
            state.replace(experience=experience),
            key,
            batch_size // blocks,
            sequence_length,
            period,
        ).experience

    experience = jax.vmap(block)(experience, jax.random.split(key, blocks))
    return TrajectoryBufferSample(
        experience=jax.tree.map(
            lambda leaf: leaf.reshape(batch_size, *leaf.shape[2:]), experience
        )
    )


def make_stratified_buffer(**kwargs) -> TrajectoryBuffer:
    buffer = fbx.make_trajectory_buffer(**kwargs)
    return buffer.replace(sample=partial(sample, **buffer.sample.keywords))
