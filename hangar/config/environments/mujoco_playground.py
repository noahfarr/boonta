from ..sections import Environment
from . import environment

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
