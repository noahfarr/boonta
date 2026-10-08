from omegaconf import MISSING

from ..sections import Libero
from ..store import place

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
