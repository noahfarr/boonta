import jax.numpy as jnp

from .axis import add_feature_axis, remove_feature_axis
from .typing import Array


def digest(cell: Array) -> Array:
    *_, width = cell.shape
    if width == 1:
        return cell[..., 0]
    seed = jnp.full(cell.shape[:-1], 2166136261, jnp.int32)
    for axis in range(width):
        seed = (seed ^ cell[..., axis]) * jnp.int32(16777619)
    return seed


def probe(cells: Array, cell: Array, num_probes: int) -> Array:
    archive_size, *_ = cells.shape
    return jnp.mod(
        add_feature_axis(digest(cell)) + jnp.arange(num_probes, dtype=jnp.int32),
        archive_size,
    )


def first(indices: Array, found: Array) -> Array:
    return remove_feature_axis(
        jnp.take_along_axis(indices, add_feature_axis(jnp.argmax(found, axis=-1)), axis=-1)
    )


def find(cells: Array, mask: Array, cell: Array, num_probes: int = 8) -> Array:
    indices = probe(cells, cell, num_probes)
    matching = mask[indices] & jnp.all(cells[indices] == cell[..., None, :], axis=-1)
    return jnp.where(jnp.any(matching, axis=-1), first(indices, matching), -1)


def claim(
    cells: Array,
    mask: Array,
    cell: Array,
    done: Array,
    num_probes: int = 8,
    priority: Array | None = None,
) -> tuple[Array, Array, Array, Array]:
    archive_size, *_ = cells.shape
    indices = probe(cells, cell, num_probes)
    occupied = mask[indices]
    matching = occupied & jnp.all(cells[indices] == cell[..., None, :], axis=-1)
    vacant = ~occupied

    known = jnp.any(matching, axis=-1)
    vacant_index = first(indices, vacant)
    insert = done & ~known & jnp.any(vacant, axis=-1)
    crowded = done & ~known & ~jnp.any(vacant, axis=-1)

    priority = jnp.full(archive_size, jnp.inf) if priority is None else priority
    lowest = jnp.where(occupied, priority[indices], jnp.inf)
    reuse_index = remove_feature_axis(
        jnp.take_along_axis(indices, add_feature_axis(jnp.argmin(lowest, axis=-1)), axis=-1)
    )
    reuse = crowded & jnp.isfinite(jnp.min(lowest, axis=-1))

    write_index = jnp.where(insert, vacant_index, jnp.where(reuse, reuse_index, archive_size))
    cells = cells.at[write_index].set(cell, mode="drop")
    mask = mask.at[write_index].set(True, mode="drop")

    index = jnp.where(
        known,
        first(indices, matching),
        jnp.where(insert, vacant_index, jnp.where(reuse, reuse_index, -1)),
    )
    safe_index = jnp.clip(index, 0, archive_size - 1)
    stored = done & (index >= 0) & mask[safe_index] & jnp.all(cells[safe_index] == cell, axis=-1)
    return cells, mask, jnp.where(stored, index, -1), (insert | reuse) & stored
