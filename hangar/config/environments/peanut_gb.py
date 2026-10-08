from ..sections import Environment
from . import environment

environment(
    Environment(
        namespace="peanut_gb",
        suite="pokemon_red",
        env_id="pokemon_red",
        num_envs=1024,
        kwargs=dict(
            num_envs="${environment.num_envs}",
            frame_skip=24,
            hold=8,
            num_threads=16,
            stack=4,
            horizon=16384,
            flag_reward=1.0,
            level_reward=0.25,
            experience_reward=0.003,
            heal_reward=0.2,
            faint_penalty=0.2,
            catch_reward=0.2,
            map_reward=0.125,
            tile_reward=0.005,
            menu_penalty=0.001,
        ),
    ),
    name="peanut_gb/pokemon_red",
)
