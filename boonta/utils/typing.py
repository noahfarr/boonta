from typing import Any, TypeAlias

import flashbax
import flax.typing
import jax

Array: TypeAlias = jax.Array
Carry: TypeAlias = Any
Dtype: TypeAlias = flax.typing.Dtype
PyTree: TypeAlias = Any
Key: TypeAlias = jax.Array

Environment: TypeAlias = Any
EnvState: TypeAlias = PyTree

Buffer: TypeAlias = flashbax.buffers.trajectory_buffer.TrajectoryBuffer
BufferState: TypeAlias = flashbax.buffers.trajectory_buffer.TrajectoryBufferState
