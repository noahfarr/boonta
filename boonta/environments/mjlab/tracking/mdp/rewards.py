def attainable(cfg) -> float:
    weights = sum(term.weight for term in cfg.rewards.values() if term.weight > 0)
    return weights * cfg.sim.mujoco.timestep * cfg.decimation
