from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.experimental import io_callback

from boonta.utils import Key, Timestep, Transition


@struct.dataclass(frozen=True)
class KinetixState:
    pass


class Kinetix:
    def __init__(
        self,
        manager,
        render: Callable | None = None,
        batch_size: int | None = None,
    ):
        self.manager = manager
        self.render = render
        self.batch_size = batch_size or manager.batch_size

        probe = self.gather()
        self.length = probe["action"].shape[1]
        leaves, self.treedef = jax.tree.flatten(probe)
        self.spec = [(leaf.shape, leaf.dtype) for leaf in leaves]
        self.lanes = {}
        for index, leaf in enumerate(leaves):
            lane = self.lanes.setdefault(np.dtype(leaf.dtype).str, [])
            lane.append(index)
        self.shapes = {
            name: jax.ShapeDtypeStruct(
                (sum(int(np.prod(self.spec[i][0])) for i in lane),),
                self.spec[lane[0]][1],
            )
            for name, lane in self.lanes.items()
        }

    def gather(self) -> dict:
        batch = self.manager.load_next_batch()
        major = np.asarray
        return {
            "action": major(batch.action),
            "action_mask": major(batch.action_mask),
            "done": major(batch.done),
            "env_state": jax.tree.map(major, batch.env_state),
        }

    def serve(self) -> dict:
        leaves = jax.tree.leaves(self.gather())
        return {
            name: np.concatenate([leaves[i].reshape(-1) for i in lane])
            for name, lane in self.lanes.items()
        }

    def unpack(self, packed: dict) -> dict:
        leaves = [None] * len(self.spec)
        for name, lane in self.lanes.items():
            buffer, offset = packed[name], 0
            for index in lane:
                shape, _ = self.spec[index]
                size = int(np.prod(shape))
                leaves[index] = buffer[offset : offset + size].reshape(shape)
                offset += size
        return jax.tree.unflatten(self.treedef, leaves)

    def weave(self, batch: dict) -> Transition:
        env_state = batch["env_state"]
        features = self.render(env_state) if self.render else env_state

        done, action = batch["done"], batch["action"]
        return Transition(
            first=Timestep(
                obs={"entities": features, "mask": batch["action_mask"]},
                action=jnp.concatenate(
                    [jnp.zeros_like(action[:, :1]), action[:, :-1]], axis=1
                ),
                terminated=jnp.concatenate(
                    [jnp.ones_like(done[:, :1]), done[:, :-1]], axis=1
                ),
                truncated=jnp.zeros_like(done),
            ),
            second=Timestep(obs=None, action=action),
            aux={"weight": batch["action_mask"]},
        )

    def init(self) -> KinetixState:
        return KinetixState()

    def update(self, state: KinetixState, key: Key, sharding) -> KinetixState:
        return state

    def close(self) -> None:
        pass

    def sample(
        self, state: KinetixState, key: Key, batch_shape: tuple[int, ...]
    ) -> Transition:
        assert tuple(batch_shape) == (self.batch_size, self.length), (
            f"quadinaros asks for batch_shape {tuple(batch_shape)} but the "
            f"manager yields ({self.batch_size}, {self.length}) trajectories "
            f"of fixed length. Set batch_shape to match."
        )
        del state, key
        return self.weave(self.unpack(io_callback(self.serve, self.shapes, ordered=False)))


def make(
    dataset_id: str,
    batch_size: int,
    num_polygons: int = 12,
    num_circles: int = 12,
    num_joints: int = 12,
    num_thrusters: int = 12,
    observation: str = "symbolic_flat",
    shards: int = -1,
    seed: int = 42,
    **kwargs,
) -> Kinetix:
    from kinetix.data.data_utils import TrajectoryDatasetManager
    from kinetix.environment import EnvParams, StaticEnvParams

    static = StaticEnvParams(
        num_polygons=num_polygons,
        num_circles=num_circles,
        num_joints=num_joints,
        num_thrusters=num_thrusters,
    )
    manager = TrajectoryDatasetManager(
        dataset_dir=dataset_id,
        batch_size=batch_size,
        static_env_params=static,
        seed=seed,
        maximum_number_of_shards=shards,
        should_expand_static_env_params=False,
    )
    return Kinetix(manager, render=renderer(observation, static), batch_size=batch_size)


def renderer(observation: str, static) -> Callable:
    from kinetix.environment import EnvParams
    from kinetix.environment.spaces import (EntityObservations,
                                            PixelObservations,
                                            SymbolicObservations,
                                            SymbolicPaddedObservations)

    kinds = {
        "symbolic_flat": SymbolicObservations,
        "symbolic_entity": EntityObservations,
        "symbolic_flat_padded": SymbolicPaddedObservations,
        "pixels": PixelObservations,
    }
    if observation not in kinds:
        raise ValueError(
            f"unknown observation type {observation}; expected one of {sorted(kinds)}"
        )
    return jax.vmap(jax.vmap(kinds[observation](EnvParams(), static).get_obs))
