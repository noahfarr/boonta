import jax
import numpy as np

from boonta.environments.wrappers.archive_auto_reset import ArchiveAutoResetState

from .artifact import Image

POSITIVE = np.array([120.0, 220.0, 120.0])
NEGATIVE = np.array([180.0, 60.0, 220.0])
EDGE = 40
GAP = 20


def paint(cell_values: np.ndarray, cell_grid: np.ndarray, tile: int, peak: float | None = None) -> np.ndarray:
    peak = max(float(np.abs(cell_values).max()) if peak is None else peak, 1e-30)
    normalized = np.append(cell_values, 0.0)[cell_grid] / peak
    intensity = np.kron(normalized, np.ones((tile, tile)))
    canvas = (
        np.multiply.outer(np.maximum(intensity, 0.0), POSITIVE)
        + np.multiply.outer(np.maximum(-intensity, 0.0), NEGATIVE)
    ).astype(np.uint8)
    canvas[::tile] = EDGE
    canvas[:, ::tile] = EDGE
    return canvas


class Overlay:
    def __init__(self, name: str = "overlay", tile: int = 16):
        self.name = name
        self.tile = tile

    def craft(self, key, podracer, podracer_state, logs) -> Image:
        archive_auto_reset, selector = podracer.pit.environment, podracer.pit.selector
        ledger = podracer_state.pit_state
        archive_auto_reset_state = next(
            node
            for node in jax.tree.leaves(
                podracer_state.environment_state,
                is_leaf=lambda node: isinstance(node, ArchiveAutoResetState),
            )
            if isinstance(node, ArchiveAutoResetState)
        )
        worth = selector.score(
            selector,
            ledger,
            archive_auto_reset,
            archive_auto_reset_state,
            algorithm=podracer.algorithm,
            algorithm_state=podracer_state.algorithm_state,
        )
        value, archive_state = jax.device_get(
            (selector.value(ledger, worth), archive_auto_reset_state.archive_state)
        )
        occupied_slots = archive_state.mask
        cells = np.squeeze(archive_state.keys, axis=-1)

        game = podracer.environment.unwrapped
        cell_values = np.zeros(game.num_cells)
        cell_values[cells[occupied_slots]] = value[occupied_slots]
        start, _ = game.init(key)
        render = np.asarray(game.render(start, tile=self.tile), np.uint8)
        num_rows, *_ = render.shape
        gap = np.full((num_rows, self.tile // 2, 3), GAP, np.uint8)
        heatmap = paint(cell_values, np.asarray(game.cell_grid), self.tile)
        return Image(self.name, np.concatenate([render, gap, heatmap], axis=1))
