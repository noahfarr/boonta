import flax.linen as nn
import jax.numpy as jnp

from boonta.utils.typing import Array


def _product(left: Array, right: Array) -> Array:
    lw, lx, ly, lz = jnp.moveaxis(left, -1, 0)
    rw, rx, ry, rz = jnp.moveaxis(right, -1, 0)
    return jnp.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )


def _matrix(quaternion: Array) -> Array:
    w, x, y, z = jnp.moveaxis(quaternion, -1, 0)
    rows = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    return jnp.stack([jnp.stack(row, axis=-1) for row in rows], axis=-2)


def _heading(quaternion: Array) -> Array:
    w, x, y, z = jnp.moveaxis(quaternion, -1, 0)
    angle = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    zero = jnp.zeros_like(angle)
    return jnp.stack([jnp.cos(angle / 2.0), zero, zero, jnp.sin(angle / 2.0)], axis=-1)


def _exponential(rotation: Array) -> Array:
    angle = jnp.sqrt(jnp.sum(rotation**2, axis=-1, keepdims=True) + 1e-12)
    return jnp.concatenate([jnp.cos(angle / 2.0), jnp.sin(angle / 2.0) * rotation / angle], axis=-1)


def _apply(matrix: Array, vector: Array) -> Array:
    return jnp.einsum("...ij,...j->...i", matrix, vector)


class FloatingBase(nn.Module):
    network: nn.Module
    num_joints: int

    def __call__(self, state: Array, action: Array, context: Array) -> Array:
        n = self.num_joints
        *batch, _ = jnp.shape(state)
        position, orientation, joints = state[..., :3], state[..., 3:7], state[..., 7 : 7 + n]
        velocity, turning = state[..., 7 + n : 10 + n], state[..., 10 + n : 13 + n]
        joint_velocity = state[..., 13 + n : 13 + 2 * n]
        previous = state[..., 13 + 2 * n :]
        heading = _matrix(_heading(orientation))
        tilt = jnp.einsum("...ji,...jk->...ik", heading, _matrix(orientation))[..., :, :2]
        features = jnp.concatenate(
            [
                position[..., 2:],
                jnp.reshape(tilt, (*batch, 6)),
                jnp.einsum("...ji,...j->...i", heading, velocity),
                turning,
                joints,
                joint_velocity,
                previous,
                action,
                context,
            ],
            axis=-1,
        )
        change = self.network(features)
        shift, rotation, acceleration, spin = (change[..., 3 * i : 3 * (i + 1)] for i in range(4))
        joint_change, joint_acceleration = change[..., 12 : 12 + n], change[..., 12 + n : 12 + 2 * n]
        following = _product(orientation, _exponential(rotation))
        following = following / jnp.linalg.norm(following, axis=-1, keepdims=True)
        return jnp.concatenate(
            [
                position + _apply(heading, shift),
                following,
                joints + joint_change,
                velocity + _apply(heading, acceleration),
                turning + spin,
                joint_velocity + joint_acceleration,
                action,
            ],
            axis=-1,
        )
