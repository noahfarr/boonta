import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("jumanji")

from boonta import environments  # noqa: E402
from boonta.environments.jumanji import FRIGHTENED, pac_man, reseed  # noqa: E402

FLOOR, PLAYER, GHOSTS, POWER_UPS, PELLETS, SCARED = range(6)


@pytest.fixture(scope="module")
def env():
    return environments.make("jumanji", "PacMan-v1", kwargs={})


def walk(env, num_steps, seed=0):
    state, timestep = env.init(jax.random.key(seed))
    step = jax.jit(env.step)
    for t in range(num_steps):
        state, timestep = step(jax.random.key(t), state, jnp.int32(t // 5 % 4))
        if bool(timestep.terminated | timestep.truncated):
            break
    return state, timestep


def test_the_planes_put_every_piece_where_the_state_says_it_is(env):
    state, timestep = walk(env, 30)
    planes = np.asarray(timestep.obs)
    assert planes.shape == env.observation_space().shape == (31, 28, 6)
    np.testing.assert_array_equal(planes, np.asarray(env.observe(state)))
    np.testing.assert_array_equal(planes[..., FLOOR], np.asarray(state.grid))
    player = state.player_locations
    assert planes[..., PLAYER].sum() == 1 and planes[int(player.x), int(player.y), PLAYER] == 1
    for column, row in np.asarray(state.ghost_locations):
        assert planes[row, column, GHOSTS] >= 1
    assert planes[..., GHOSTS].sum() == 4
    assert planes[..., PELLETS].sum() == int(state.pellets)
    assert planes[..., POWER_UPS].sum() == int(np.any(np.asarray(state.power_up_locations) != 0, axis=-1).sum())
    for plane in (PLAYER, POWER_UPS, PELLETS):
        assert (planes[..., plane] * (1 - planes[..., FLOOR])).sum() == 0
    np.testing.assert_allclose(planes[..., SCARED], max(int(state.frightened_state_time), 0) / FRIGHTENED)


def test_an_eaten_pellet_leaves_its_plane(env):
    start, first = env.init(jax.random.key(0))
    state, timestep = walk(env, 30)
    eaten = int(start.pellets) - int(state.pellets)
    assert eaten > 0
    assert np.asarray(first.obs)[..., PELLETS].sum() - np.asarray(timestep.obs)[..., PELLETS].sum() == eaten


def test_the_cell_follows_pacman_and_the_pellets_left(env):
    cell = pac_man(env, bucket=4)
    start, _ = env.init(jax.random.key(0))
    moved, _ = walk(env, 30)
    keys = np.asarray(cell(jax.tree.map(lambda a, b: jnp.stack([a, b]), start, moved)))
    assert keys[0] != keys[1]
    def with_pellets(count):
        return moved.replace(env_state=moved.env_state.replace(pellets=count))

    same = with_pellets(moved.pellets - moved.pellets % 4)
    assert int(cell(moved)) == int(cell(same))
    fewer = with_pellets(moved.pellets - 4)
    assert int(cell(moved)) != int(cell(fewer))


def test_reseeding_gives_each_restored_copy_its_own_key(env):
    batch, _ = jax.vmap(env.init)(jax.random.split(jax.random.key(1), 4))
    fresh = reseed(batch, jax.random.key(2))
    assert fresh.key.dtype == batch.key.dtype and fresh.key.shape == batch.key.shape
    data = np.asarray(jax.random.key_data(fresh.key))
    assert len({tuple(row) for row in data}) == 4
    np.testing.assert_array_equal(np.asarray(fresh.grid), np.asarray(batch.grid))


def test_the_local_view_puts_pacman_in_the_middle_and_wraps_like_the_board():
    from hangar.recipes.ppo_pac_man import centred

    planes = np.zeros((31, 28, 6), np.float32)
    planes[10, 0, PLAYER] = 1.0
    planes[12, 3, GHOSTS] = 1.0  # two rows down, three columns right of PacMan
    planes[10, 27, PELLETS] = 1.0  # one column left of PacMan, across the wrapped edge
    view = np.asarray(centred(jnp.asarray(planes), radius=7))
    assert view.shape == (15, 15, 6)
    assert view[7, 7, PLAYER] == 1.0 and view[..., PLAYER].sum() == 1.0
    assert view[9, 10, GHOSTS] == 1.0 and view[..., GHOSTS].sum() == 1.0
    assert view[7, 6, PELLETS] == 1.0 and view[..., PELLETS].sum() == 1.0


def test_every_piece_only_ever_stands_on_a_walkable_cell(env):
    from hangar.recipes.ppo_pac_man import walkable

    floor = np.zeros(31 * 28, bool)
    floor[list(walkable(env))] = True
    assert floor.sum() == np.asarray(env.init(jax.random.key(0))[1].obs)[..., FLOOR].sum()
    state, timestep = env.init(jax.random.key(3))
    step = jax.jit(env.step)
    for t in range(300):
        planes = np.asarray(timestep.obs).reshape(-1, 6)
        assert not planes[~floor][:, [PLAYER, GHOSTS, POWER_UPS, PELLETS]].any()
        state, timestep = step(jax.random.key(t), state, jnp.int32((t // 7) % 4))


def test_the_egocentric_encoder_keeps_the_leading_axes(env):
    from hangar.recipes.ppo_pac_man import Egocentric

    _, timestep = walk(env, 3)
    obs = jnp.broadcast_to(timestep.obs, (2, 5, *timestep.obs.shape))
    from hangar.recipes.ppo_pac_man import walkable

    net = Egocentric(floor=walkable(env), features=32)
    params = net.init(jax.random.key(0), obs)
    assert net.apply(params, obs).shape == (2, 5, 32)


def test_pacman_renders_from_the_numpy_state_the_render_artisan_hands_it(env):
    state, _ = walk(env, 10)
    image = env.render(jax.device_get(state))
    assert image.ndim == 3 and image.shape[-1] == 3 and image.dtype == np.uint8
    assert len(np.unique(image.reshape(-1, 3), axis=0)) > 5


def test_running_out_of_time_truncates_and_a_restore_restarts_the_clock():
    from boonta.environments.wrappers.archive_auto_reset import reset_time_limit

    env = environments.make("jumanji", "PacMan-v1", kwargs={"time_limit": 6})
    assert env.time_limit() == 6
    state, _ = env.init(jax.random.key(0))
    step = jax.jit(env.step)
    ends = []
    for t in range(6):
        state, timestep = step(jax.random.key(t), state, jnp.int32(4))
        ends.append((bool(timestep.terminated), bool(timestep.truncated)))
    assert ends == [(False, False)] * 5 + [(False, True)]
    assert int(reset_time_limit(state).time) == 0
