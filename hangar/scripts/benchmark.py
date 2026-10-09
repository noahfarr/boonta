import performax as px

px.enable_device_profiling()
px.enable_barriers()

import time
from dataclasses import dataclass

import hydra
import jax
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from hangar import resolvers  # noqa: F401


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    from boonta.algorithms.wrappers import Wrapper as AlgorithmWrapper
    from boonta.environments.wrappers import Wrapper as EnvironmentWrapper
    from boonta.utils import brief
    from hangar import recipes

    @dataclass
    class TrackedAlgorithm(AlgorithmWrapper):
        def step(self, state, key, timestep, temperature=1.0):
            return px.track(name="algorithm/step")(self.algorithm.step)(
                state, key, timestep, temperature
            )

        def update(self, state, key, transitions):
            return px.track(name="algorithm/update")(self.algorithm.update)(
                state, key, transitions
            )

    class TrackedEnvironment(EnvironmentWrapper):
        def step(self, key, state, action):
            return px.track(name="environment/step")(self._env.step)(
                key, state, action
            )

        def update(self, state, **kwargs):
            return self._env.update(state, **kwargs)

        def action_mask(self, state):
            return self._env.action_mask(state)

        def observe(self, state):
            return self._env.observe(state)

        def render(self, state):
            return self._env.render(state)

        def close(self, state):
            return self._env.close(state)

    def identity(state):
        return state

    namespace = cfg.environment.namespace
    suite = cfg.environment.get("suite", namespace)
    name = HydraConfig.get().runtime.choices["algorithm"]
    components = recipes.register[(name, namespace, suite)](cfg)

    components["algorithm"] = TrackedAlgorithm(components["algorithm"])
    if components.get("environment") is not None:
        components["environment"] = TrackedEnvironment(components["environment"])
    components["pit"] = px.track(name="pit")(components.get("pit", identity))
    components["lap"] = px.track(name="lap")(components.get("lap", identity))
    podracer = instantiate(cfg.podracer)(**components)

    num_updates = int(cfg.total_timesteps) // (
        podracer.batch_size * cfg.training.num_epochs
    )
    assert num_updates >= 1, (
        f"total_timesteps ({cfg.total_timesteps}) over {cfg.training.num_epochs} "
        f"epochs is less than one update ({podracer.batch_size} steps) per epoch."
    )

    init_key, profile_key, *warmup_keys = jax.random.split(
        jax.random.key(cfg.seed), 4
    )
    state = podracer.init(init_key)
    brief(cfg, state)

    for warmup_key in warmup_keys:
        state, _ = jax.block_until_ready(
            podracer.train(state, warmup_key, num_updates)
        )

    clock = {}

    def race(state):
        start = time.perf_counter()
        result = jax.block_until_ready(podracer.train(state, profile_key, num_updates))
        clock["seconds"] = time.perf_counter() - start
        return result

    (state, _), stats = px.profile(race)(state)
    podracer.close(state)

    batch_size = podracer.batch_size
    times = {
        region.name: region.total_duration_ms / 1000 / num_updates
        for region in stats.device.stats
    }
    acting = sum(
        times.get(region, 0.0)
        for region in ("algorithm/step", "environment/step", "lap")
    )

    metrics = {}
    if times.get("environment/step"):
        metrics["environment/SPS"] = batch_size / times["environment/step"]
    if acting:
        metrics["rollout/SPS"] = batch_size / acting
    if times.get("algorithm/update"):
        metrics["update/SPS"] = batch_size / times["algorithm/update"]
    metrics["training/SPS"] = num_updates * batch_size / clock["seconds"]
    for region, seconds in times.items():
        metrics[f"{region}/time"] = seconds

    for name, value in metrics.items():
        if name.endswith("/time"):
            print(f"{name}: {value:.6f}")
        else:
            print(f"{name}: {value:,.0f}")


if __name__ == "__main__":
    main()
