import jax
import numpy as np

from .artifact import Metrics


def lorenz(visits: np.ndarray, mass: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-(mass / np.maximum(visits, 1e-30)))
    steps = np.concatenate([[0.0], np.cumsum(visits[order])]) / max(float(visits.sum()), 1e-30)
    signal = np.concatenate([[0.0], np.cumsum(mass[order])]) / max(float(mass.sum()), 1e-30)
    return steps, signal


class Concentration:
    def __init__(self, name: str = "concentration", min_visits: float = 20.0):
        self.name = name
        self.min_visits = min_visits

    def craft(self, key, podracer, podracer_state, logs) -> Metrics:
        from boonta.curricula.prd import signal

        ledger = jax.device_get(podracer_state.pit_state)
        visits = np.asarray(ledger.reach, np.float64)
        per_step = np.asarray(signal(ledger.gain), np.float64)
        counted = np.asarray(ledger.gain.cell_visits, np.float64) >= self.min_visits
        visits = np.where(counted, visits, 0.0)
        mass = visits * np.where(counted, per_step, 0.0)
        if mass.sum() <= 0.0 or visits.sum() <= 0.0:
            return Metrics(self.name, {})
        steps, carried = lorenz(visits, mass)
        return Metrics(
            self.name,
            {
                f"{self.name}/steps_for_half": float(np.interp(0.5, carried, steps)),
                f"{self.name}/steps_for_ninety": float(np.interp(0.9, carried, steps)),
                f"{self.name}/index": float(2.0 * np.trapezoid(carried - steps, steps)),
                f"{self.name}/counted_steps": float(visits.sum() / max(float(np.asarray(ledger.reach).sum()), 1e-30)),
            },
        )
