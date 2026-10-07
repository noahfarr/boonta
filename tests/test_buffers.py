import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from boonta.buffers import (every_step, importance_weights, make_episode_buffer,
                            make_prioritised_episode_buffer)

ROWS, TIME, LENGTH, BATCH, EPISODE = 2, 32, 4, 256, 5


@struct.dataclass
class Step:
    done: jnp.ndarray
    value: jnp.ndarray


@struct.dataclass
class Item:
    first: Step


def build(make, get_start_flags=None):
    flags = {} if get_start_flags is None else {"get_start_flags": get_start_flags}
    return make(
        max_length=ROWS * TIME,
        min_length=LENGTH,
        sample_batch_size=BATCH,
        sample_sequence_length=LENGTH,
        add_sequences=True,
        add_batch_size=ROWS,
        **flags,
    )


def fill(buffer, steps):
    state = buffer.init(Item(Step(jnp.bool_(True), jnp.float32(0))))
    for step in range(steps):
        state = buffer.add(
            state,
            Item(
                Step(
                    jnp.full((ROWS, 1), step % EPISODE == 0),
                    jnp.full((ROWS, 1), float(step)),
                )
            ),
        )
    return state


BUFFERS = [
    pytest.param(make_episode_buffer, id="episode"),
    pytest.param(make_prioritised_episode_buffer, id="prioritised"),
]


@pytest.mark.parametrize("make", BUFFERS)
def test_sequences_start_at_episode_starts(make):
    buffer = build(make)
    sampled = buffer.sample(fill(buffer, 23), jax.random.key(0)).experience.first
    assert np.all(np.asarray(sampled.done)[:, 0])
    np.testing.assert_array_equal(np.asarray(sampled.value)[:, 0] % EPISODE, 0)


@pytest.mark.parametrize("make", BUFFERS)
def test_a_full_buffer_never_wraps_past_the_write_head(make):
    buffer = build(make, every_step)
    values = np.asarray(
        buffer.sample(fill(buffer, TIME + 13), jax.random.key(0)).experience.first.value
    )
    np.testing.assert_array_equal(np.diff(values, axis=1), 1.0)
    assert values.min() >= 13


@pytest.mark.parametrize("make", BUFFERS)
def test_without_episode_starts_any_valid_window_is_drawn_uniformly(make):
    buffer = build(make, lambda experience: jnp.zeros_like(experience.first.done))
    sampled = buffer.sample(fill(buffer, TIME + 13), jax.random.key(0))
    values = np.asarray(sampled.experience.first.value)

    np.testing.assert_array_equal(np.diff(values, axis=1), 1.0)
    assert values.min() >= 13
    assert len(np.unique(values[:, 0])) > TIME // 2
    if hasattr(sampled, "probabilities"):
        windows = ROWS * (TIME - LENGTH + 1)
        np.testing.assert_allclose(np.asarray(sampled.probabilities), 1 / windows)


def test_every_new_start_gets_the_maximum_priority():
    buffer = build(make_prioritised_episode_buffer, every_step)
    sampled = buffer.sample(fill(buffer, 20), jax.random.key(0))
    starts = np.unique(np.asarray(sampled.experience.first.value)[:, 0])
    np.testing.assert_array_equal(starts, np.arange(20 - LENGTH + 1))


def test_set_priorities_biases_sampling():
    buffer = build(make_prioritised_episode_buffer, every_step)
    state = buffer.set_priorities(fill(buffer, 20), jnp.array([3]), jnp.array([1000.0]))
    sampled = buffer.sample(state, jax.random.key(0))
    starts = np.asarray(sampled.experience.first.value)[:, 0]
    assert np.mean(starts == 3.0) > 0.5
    np.testing.assert_array_equal(np.asarray(sampled.indices) % TIME, starts.astype(int))


def test_importance_weights_are_largest_for_the_rarest_samples():
    weights = np.asarray(importance_weights(jnp.array([0.5, 0.1, 0.01]), size=100))
    assert weights.max() == 1.0
    assert weights[2] > weights[1] > weights[0]
