import flashbax
from omegaconf import MISSING

from boonta.buffers import make_episode_buffer, make_prioritised_episode_buffer

from .sections import Dataset
from .store import fbuilds, place, store

buffer = store(group="buffer", package="buffer")


def dataset(node, name):
    place(node, f"dataset/{name}", package="dataset")


buffer(
    fbuilds(
        flashbax.make_trajectory_buffer,
        max_length_time_axis="${eval:'${replay.capacity} // ${environment.num_envs}'}",
        min_length_time_axis="${eval:'max(1, ${buffer.sample_batch_size} // ${environment.num_envs})'}",
        sample_batch_size=512,
        add_batch_size="${environment.num_envs}",
        sample_sequence_length=1,
        period=1,
    ),
    name="transition",
)
buffer(
    fbuilds(
        flashbax.make_trajectory_buffer,
        max_length_time_axis="${eval:'${replay.capacity} // ${environment.num_envs}'}",
        min_length_time_axis="${buffer.sample_sequence_length}",
        sample_batch_size=256,
        add_batch_size="${environment.num_envs}",
        sample_sequence_length=16,
        period=1,
    ),
    name="trajectory",
)
buffer(
    fbuilds(
        make_episode_buffer,
        max_length="${replay.capacity}",
        min_length="${buffer.sample_sequence_length}",
        sample_batch_size=256,
        sample_sequence_length=MISSING,
        add_batch_size="${environment.num_envs}",
        add_sequences=True,
        zen_exclude=("get_start_flags",),
    ),
    name="episode",
)
buffer(
    fbuilds(
        make_prioritised_episode_buffer,
        max_length="${replay.capacity}",
        min_length="${buffer.sample_sequence_length}",
        sample_batch_size=256,
        sample_sequence_length=MISSING,
        add_batch_size="${environment.num_envs}",
        add_sequences=True,
        priority_exponent=0.6,
        zen_exclude=("get_start_flags",),
    ),
    name="prioritised_episode",
)

devices = "${podracer.config.mesh.count}"
dataset(
    Dataset(namespace="disk", dataset_id=MISSING, kwargs=dict(pool_size=0, num_devices=devices)),
    name="disk",
)
dataset(
    Dataset(
        namespace="minari",
        dataset_id="mujoco/${environment.env_id}/expert-v0",
        kwargs=dict(num_devices=devices),
    ),
    name="minari/mujoco/expert",
)
for name, batch_size, polygons, circles, joints, thrusters in (
    ("offline_m", 32, 6, 3, 2, 2),
    ("offline_s", 256, 5, 2, 1, 1),
):
    dataset(
        Dataset(
            namespace="kinetix",
            dataset_id=MISSING,
            kwargs=dict(
                batch_size=batch_size,
                num_polygons=polygons,
                num_circles=circles,
                num_joints=joints,
                num_thrusters=thrusters,
                frame_skip=2,
                observation="symbolic_entity",
                shards=-1,
                seed=42,
            ),
        ),
        name=f"kinetix/{name}",
    )
