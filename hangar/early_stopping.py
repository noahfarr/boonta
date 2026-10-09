import numpy as np


def default(epoch, step, data):
    return False


def epochs(epoch, step, data, at):
    return epoch >= at


def steps(epoch, step, data, at):
    return step >= at


def plateau(patience, score):
    best = -np.inf
    waited = 0

    def early_stopping(epoch, step, data):
        nonlocal best, waited
        returns = data.get(score)
        value = -np.inf
        if returns is not None and np.isfinite(returns).any():
            value = float(np.nanmean(returns))
        if value > best:
            best, waited = value, 0
            return False
        waited += 1
        return waited >= patience

    return early_stopping
