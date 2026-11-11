import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from boonta import environments
from boonta.environments import craftaxcu
from boonta.environments.wrappers import Batched

SOURCE = Path(craftaxcu.__file__).parent / 'ffi/vendor/craftax.cu'


def test_achievement_names_follow_the_vendored_kernel_order():
    text = SOURCE.read_text()
    table = re.search(r'CU_ACH_NAMES\[CRAFTAX_NUM_ACHIEVEMENTS\] = \{(.*?)\};', text, re.S)
    assert tuple(re.findall(r'"(\w+)"', table.group(1))) == craftaxcu.ACHIEVEMENTS
    enum = re.findall(r'CRAFTAX_ACH_(\w+) = (\d+),', text)
    assert len(enum) == len(craftaxcu.ACHIEVEMENTS)
    assert all(craftaxcu.ACHIEVEMENTS[int(index)] == name.lower().replace('solider', 'soldier')
               for name, index in enum)


def test_vendor_pin_and_license_are_retained():
    vendor = SOURCE.parent
    assert (vendor / '.commit').read_text().strip() == '13202363e1f7a4bb0e7b95255b6425c5eab16f07'
    assert 'LLVM' in (vendor / 'LICENSE').read_text()
    assert 'Elliot Arledge' in (vendor / 'NOTICE').read_text()
    assert environments.registry['craftaxcu'] is craftaxcu.make


@pytest.fixture(scope='module')
def env():
    if not any(device.platform == 'gpu' for device in jax.devices()):
        pytest.skip('requires a CUDA-enabled JAX GPU and nvcc')
    return environments.make('craftaxcu', 'Craftax-Symbolic-Cu', kwargs={'num_envs': 8, 'fixed': 7, 'coarse': 4})


def assert_observation_equal(actual, expected):
    np.testing.assert_array_equal(actual['view'], expected['view'])
    np.testing.assert_array_equal(actual['mask'], expected['mask'])


def test_native_batch_steps_and_reobserves_under_jit(env):
    state, first = jax.jit(env.init)(jax.random.key(0))
    assert first.obs['view'].shape == (8, env.obs_size)
    assert first.obs['mask'].shape == (8, env.num_actions)
    assert env.observation_space()['view'].shape == (env.obs_size,)
    assert env.action_space().shape == ()
    assert first.terminated.all()
    after, timestep = jax.jit(env.step)(jax.random.key(1), state, jnp.zeros(8, jnp.int32))
    assert jnp.isfinite(timestep.obs['view']).all()
    assert jnp.isfinite(timestep.reward).all()
    assert_observation_equal(jax.jit(env.observe)(after), timestep.obs)
    np.testing.assert_array_equal(jax.jit(env.action_mask)(after), timestep.obs['mask'])
    assert jax.tree.structure(after) == jax.tree.structure(state)
    # An input snapshot must remain usable after a step through aliased FFI buffers.
    assert_observation_equal(jax.jit(env.observe)(state), first.obs)


def test_restored_snapshot_replays_the_same_step(env):
    state, _ = jax.jit(env.init)(jax.random.key(2))
    actions = jnp.zeros(8, jnp.int32)
    step = jax.jit(env.step)
    after, first = step(jax.random.key(3), state, actions)
    replay, second = step(jax.random.key(3), state, actions)
    assert_observation_equal(first.obs, second.obs)
    np.testing.assert_array_equal(first.reward, second.reward)
    for actual, expected in zip(after, replay):
        np.testing.assert_array_equal(actual, expected)


def test_reseed_changes_only_the_saved_rng_key(env):
    state, first = jax.jit(env.init)(jax.random.key(4))
    refreshed = jax.jit(env.reseed)(state, jax.random.key(5))
    np.testing.assert_array_equal(refreshed[0], state[0])
    np.testing.assert_array_equal(refreshed[1], state[1])
    offset = env.tail_key
    np.testing.assert_array_equal(refreshed[2][:, :offset], state[2][:, :offset])
    np.testing.assert_array_equal(refreshed[2][:, offset + 8:], state[2][:, offset + 8:])
    assert not np.array_equal(refreshed[2][:, offset:offset + 8], state[2][:, offset:offset + 8])
    assert_observation_equal(jax.jit(env.observe)(refreshed), first.obs)


def test_native_state_descriptions_and_batched_wrapper(env):
    wrapped = Batched(env, num_envs=8)
    state, _ = jax.jit(wrapped.init)(jax.random.key(6))
    assert jax.jit(env.cells)(state).shape == (8,)
    assert jax.jit(env.worlds)(state).shape == (8,)
    assert jax.jit(env.portray)(state).shape == (8, 9)
    held = tuple(leaf[:1] for leaf in state)
    assert jax.jit(env.portray)(held).shape == (1, 9)


def craftax_config():
    from hydra import compose, initialize_config_dir
    from hangar import resolvers  # noqa: F401 -- registers config resolvers

    config = Path(__file__).parents[1] / 'hangar/config'
    with initialize_config_dir(config_dir=str(config), version_base=None):
        return compose('train', overrides=[
            'environment=craftaxcu/symbolic', 'environment.num_envs=8', 'seed=11',
        ])


def test_config_selects_native_batching_and_resolves_environment_parameters():
    cfg = craftax_config()
    assert cfg.environment.namespace == 'craftaxcu'
    assert cfg.environment.auto_reset._target_ == 'boonta.environments.wrappers.Batched'
    assert cfg.environment.auto_reset.num_envs == 8
    assert cfg.environment.kwargs.num_envs == 8
    assert cfg.environment.kwargs.seed == 11
    assert cfg.environment.kwargs.coarse == 4


def test_config_builds_native_batched_environment(env):
    from hydra.utils import instantiate

    cfg = craftax_config()
    raw = environments.make(**cfg.environment)
    wrapped = instantiate(cfg.environment.auto_reset, raw)
    state, first = jax.jit(wrapped.init)(jax.random.key(11))
    assert first.obs['view'].shape == (8, raw.obs_size)
    _, timestep = jax.jit(wrapped.step)(jax.random.key(12), state, jnp.zeros(8, jnp.int32))
    assert timestep.reward.shape == (8,)
    assert jnp.isfinite(timestep.obs['view']).all()
