from hydra_zen import make_config
from omegaconf import MISSING

from .sections import Craftax, Environment, Kinetix, Libero, Minatar
from .store import place


def environment(node, name):
    place(node, f"environment/{name}", package="environment")


environment(
    Environment(
        namespace="ale",
        suite="ale",
        env_id="montezuma_revenge",
        num_envs=128,
        kwargs=dict(
            num_envs="${environment.num_envs}",
            frame_skip=4,
            num_threads=28,
            fraction=0.0,
            capacity=4096,
            depth=8,
            warmup_episodes=1024,
            novelty=0.0,
            snapshot_every=0,
            snapshot="",
        ),
    ),
    name="ale/montezuma",
)

for env_id in (
    "ant",
    "halfcheetah",
    "hopper",
    "humanoidstandup",
    "humanoid",
    "inverted_double_pendulum",
    "inverted_pendulum",
    "pusher",
    "reacher",
    "swimmer",
    "walker2d",
):
    environment(
        Environment(namespace="brax", suite="mujoco", env_id=env_id, kwargs=dict(backend="generalized")),
        name=f"brax/mujoco/{env_id}",
    )

environment(
    Environment(
        namespace="connectx",
        suite="connectx",
        env_id="connectx",
        num_envs=1024,
        kwargs=dict(rows=6, columns=7, inarow=4),
    ),
    name="connectx/connectx",
)

environment(
    Craftax(namespace="craftax", suite="craftax_classic", env_id="Craftax-Classic-Symbolic-v1", reset_ratio=64),
    name="craftax/craftax_classic/symbolic",
)
environment(
    Craftax(namespace="craftax", suite="craftax", env_id="Craftax-Symbolic-v1", reset_ratio=64),
    name="craftax/craftax/symbolic",
)

environment(
    Environment(
        namespace="gymnasium",
        suite="gymnasium",
        env_id="CartPole-v1",
        num_envs=16,
        kwargs=dict(num_envs="${environment.num_envs}", vectorization_mode="async"),
    ),
    name="gymnasium/cartpole",
)

deep_sea = Environment(namespace="gymnax", suite="bsuite", env_id="DeepSea-bsuite", kwargs=dict(size=20))
environment(deep_sea, name="gymnax/bsuite")
environment(deep_sea, name="gymnax/bsuite/deep_sea")
environment(
    Environment(namespace="gymnax", suite="classic_control", env_id="CartPole-v1"),
    name="gymnax/classic_control/cartpole",
)
for name, env_id in (
    ("asterix", "Asterix-MinAtar"),
    ("breakout", "Breakout-MinAtar"),
    ("freeway", "Freeway-MinAtar"),
    ("seaquest", "Seaquest-MinAtar"),
    ("space_invaders", "SpaceInvaders-MinAtar"),
):
    environment(Minatar(namespace="gymnax", suite="minatar", env_id=env_id), name=f"gymnax/minatar/{name}")


def isaaclab(suite, env_id, num_envs=4096):
    return Environment(
        namespace="isaaclab",
        suite=suite,
        env_id=env_id,
        num_envs=num_envs,
        kwargs=dict(num_envs="${environment.num_envs}", device="cuda:0", headless=True),
    )


environment(isaaclab("isaaclab", MISSING), name="isaaclab/isaaclab")
environment(isaaclab("classic", "Isaac-Ant-v0"), name="isaaclab/classic/ant")
environment(isaaclab("classic", "Isaac-Cartpole-v0"), name="isaaclab/classic/cartpole")
environment(isaaclab("classic", "Isaac-Humanoid-v0"), name="isaaclab/classic/humanoid")
environment(
    isaaclab("locomotion", "Isaac-Velocity-Flat-Anymal-C-v0"), name="isaaclab/locomotion/anymal_c_flat"
)
environment(
    isaaclab("locomotion", "Isaac-Velocity-Rough-Anymal-C-v0"), name="isaaclab/locomotion/anymal_c_rough"
)
environment(
    isaaclab("manipulation", "Isaac-Open-Drawer-Franka-v0"), name="isaaclab/manipulation/franka_cabinet"
)
environment(isaaclab("manipulation", "Isaac-Lift-Cube-Franka-v0"), name="isaaclab/manipulation/franka_lift")
environment(isaaclab("manipulation", "Isaac-Reach-Franka-v0"), name="isaaclab/manipulation/franka_reach")
environment(
    isaaclab("manipulation", "Isaac-Repose-Cube-Shadow-Direct-v0", num_envs=8192),
    name="isaaclab/manipulation/shadow_hand",
)
environment(isaaclab("multirotor", "Isaac-Quadcopter-Direct-v0"), name="isaaclab/multirotor/quadcopter")

for scenario in (
    "10m_vs_11m",
    "25m",
    "27m_vs_30m",
    "2s3z",
    "3m",
    "3s5z_vs_3s6z",
    "3s5z",
    "3s_vs_5z",
    "5m_vs_6m",
    "6h_vs_8z",
    "8m",
    "smacv2_10_units",
    "smacv2_20_units",
    "smacv2_5_units",
):
    environment(
        Environment(
            namespace="jaxmarl", suite="smax", env_id="HeuristicEnemySMAX", kwargs=dict(scenario=scenario)
        ),
        name=f"jaxmarl/smax/{scenario}",
    )

environment(Environment(namespace="jumanji", suite="sokoban", env_id="Sokoban-v0"), name="jumanji/sokoban")

kinetix = dict(action_type="multi_discrete", observation_type="symbolic_entity")
environment(
    Kinetix(namespace="kinetix", suite="kinetix", env_id=None, num_envs=32, kwargs=kinetix),
    name="kinetix/kinetix",
)
environment(
    Kinetix(
        namespace="kinetix", suite="kinetix", env_id=None, holdout_levels=True, num_envs=256, kwargs=kinetix
    ),
    name="kinetix/holdout_levels",
)
environment(
    Kinetix(
        namespace="kinetix",
        suite="kinetix",
        env_id=[
            "m/h0_unicycle",
            "m/h1_car_left",
            "m/h2_car_right",
            "m/h3_car_thrust",
            "m/h4_thrust_the_needle",
            "m/h5_angry_birds",
            "m/h6_thrust_over",
            "m/h7_car_flip",
            "m/h8_weird_vehicle",
            "m/h9_spin_the_right_way",
            "m/h10_thrust_right_easy",
            "m/h11_thrust_left_easy",
            "m/h12_thrustfall_left",
            "m/h13_thrustfall_right",
            "m/h14_thrustblock",
            "m/h15_thrustshoot",
            "m/h16_thrustcontrol_right",
            "m/h17_thrustcontrol_left",
            "m/h18_thrust_right_very_easy",
            "m/h19_thrust_left_very_easy",
            "m/arm_up",
            "m/arm_left",
            "m/arm_right",
            "m/arm_hard",
        ],
        num_envs=256,
        kwargs=kinetix,
    ),
    name="kinetix/holdout_m",
)
environment(
    Kinetix(
        namespace="kinetix",
        suite="kinetix",
        env_id=[
            "s/h0_weak_thrust",
            "s/h1_thrust_over_ball",
            "s/h2_one_wheel_car",
            "s/h3_point_the_thruster",
            "s/h4_thrust_aim",
            "s/h5_rotate_fall",
            "s/h6_unicycle_right",
            "s/h7_unicycle_left",
            "s/h8_unicycle_balance",
            "s/h9_explode_then_thrust_over",
        ],
        num_envs=256,
        kwargs=kinetix,
    ),
    name="kinetix/holdout_s",
)

for name, suite in (
    ("libero", MISSING),
    ("libero_10/task_0", "libero_10"),
    ("libero_90/task_0", "libero_90"),
    ("libero_goal/task_0", "libero_goal"),
    ("libero_object/task_0", "libero_object"),
    ("libero_spatial/task_0", "libero_spatial"),
):
    place(
        dict(
            environment=Libero(
                namespace="libero",
                suite=suite,
                env_id=0,
                num_envs=8,
                horizon=300,
                kwargs=dict(task_suite_name="${environment.suite}", image_size=224, num_envs="${environment.num_envs}"),
            ),
            algorithm=dict(action_horizon=50),
            dataloader=dict(path=MISSING),
        ),
        f"environment/libero/{name}",
        package="_global_",
    )

for name in ("find_return", "king_hill", "prey", "scouts", "snake", "traveling_salesman"):
    environment(Environment(namespace="mapox", suite="mapox", env_id=name), name=f"mapox/{name}")

for name, env_id in (
    ("cartpole_balance_sparse", "CartpoleBalanceSparse"),
    ("cartpole_balance", "CartpoleBalance"),
    ("cartpole_swingup_sparse", "CartpoleSwingupSparse"),
    ("cartpole_swingup", "CartpoleSwingup"),
):
    environment(
        Environment(
            namespace="mujoco_playground",
            suite="dm_control_suite",
            env_id=env_id,
            kwargs=dict(config_overrides=dict(impl="jax")),
        ),
        name=f"mujoco_playground/dm_control_suite/{name}",
    )

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

environment(
    make_config(
        hydra_defaults=["/artisan/transcript", "_self_"],
        namespace="wordle",
        suite="wordle",
        env_id="Wordle-v0",
        bases=(Environment,),
    ),
    name="wordle/wordle",
)

environment(
    Environment(namespace="xland_minigrid", suite="minigrid", env_id="MiniGrid-Empty-5x5"),
    name="xland_minigrid/minigrid/empty_5x5",
)
environment(
    Environment(
        namespace="xland_minigrid", suite="xland", env_id="XLand-MiniGrid-R1-9x9", kwargs=dict(benchmark="trivial-21k")
    ),
    name="xland_minigrid/xland/r1_9x9",
)
