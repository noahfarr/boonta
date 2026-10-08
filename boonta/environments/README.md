# environments

An environment is a game the agent plays. Every environment, whether it is written in JAX, runs in Gymnasium or calls C++ code, has the same interface, defined in [`environment.py`](environment.py).

| Method | Does |
| --- | --- |
| `init(key)` | starts an episode and returns `(state, timestep)` |
| `step(key, state, action)` | advances one step and returns `(state, timestep)` |
| `update(state, **kwargs)` | changes settings inside a running state |
| `close(state)` | frees anything the state holds outside JAX |
| `observation_space()`, `action_space()` | describe the shapes, as a [`Space`](spaces.py) |
| `action_mask(state)` | the legal actions (all of them by default) |
| `time_limit()` | the longest episode, for code that needs it |

`num_agents` is 1 unless the game seats several agents.

## Timesteps

A step returns a [`Timestep`](../utils/timestep.py) with `obs`, `action`, `reward`, `terminated`, `truncated` and `info`. `action` is the action that led to this timestep, and `done` is `terminated | truncated`.

- `terminated` means the episode really ended. Nothing follows, so the value of what comes next is zero.
- `truncated` means the episode was cut short, usually by a time limit. The targets stop there too, but the state is not treated as an end of the game.
- The timestep from `init` sets `terminated=True`. It marks an episode start, so recurrent networks reset their memory there. Gymnax, ALE, Jumanji and the test dummies all do this.

## Auto-reset

Podracers never call `init` in the middle of training, so a finished episode must restart by itself. boonta's convention is the same-step reset of [`SameStepAutoReset`](wrappers/same_step_auto_reset.py). On the last step of an episode, `reward`, `terminated` and `truncated` describe the end, while `obs` is already the first observation of the next episode. [`NextStepAutoReset`](wrappers/next_step_auto_reset.py) instead spends one extra empty step on the reset.

## Wrappers

A wrapper is an environment around another one. [`wrappers/`](wrappers) holds them all, for example `TimeLimit`, `RecordEpisodeStatistics` (logs `episode_statistics/episode_return`), `NormalizeObservation` and `StickyAction`. A wrapper that keeps state of its own stores it in a subclass of `WrapperState`, as [`TimeLimit`](wrappers/time_limit.py) does.

`update` must reach the game through every wrapper. The default `Wrapper.update` and `Wrapper.action_mask` assume the wrapper owns a `WrapperState`. So a wrapper with no state of its own must pass the state straight through, as [`TransformReward`](wrappers/transform_reward.py) does:

```python
class TransformReward(Wrapper):
    def __init__(self, env, fn: Callable):
        super().__init__(env)
        self.fn = fn

    def step(self, key, state, action):
        state, timestep = self._env.step(key, state, action)
        return state, timestep.replace(reward=self.fn(timestep.reward))

    def update(self, state, **kwargs):
        return self._env.update(state, **kwargs)

    def action_mask(self, state):
        return self._env.action_mask(state)
```

A stateless wrapper that forgets this raises as soon as anything reconfigures the stack.

### Many copies at once

[`Vectorize(env, num_envs)`](wrappers/vectorize.py) runs `num_envs` copies of a single environment with `jax.vmap`. A podracer always gets a batched environment. [`Batched`](wrappers/batched.py) is for an environment that already steps a whole batch itself, such as the Gymnasium adapter. It has the interface of `Vectorize` without the `vmap`.

A typical stack, from [`hangar/recipes/ppo_minatar.py`](../../hangar/recipes/ppo_minatar.py):

```python
env = environments.make(**cfg.environment)
env = SameStepAutoReset(env)
env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
env = Vectorize(env, num_envs=cfg.environment.num_envs)
```

## Add an environment

There are three ways in.

### Pure JAX

Subclass `Environment` and keep the state in a `flax.struct.dataclass`. This is the fastest kind, since it compiles into the training step. The `Corridor` dummy in [`tests/dummies.py`](../../tests/dummies.py) is a complete example:

```python
class Corridor(Environment):
    def __init__(self, length: int = 4, horizon: int = 12, solved: float = 0.9):
        self.length = length
        self.horizon = horizon
        self.solved = solved
        self.shortest = length - 1

    def observation_space(self) -> Space:
        return Space((self.length,), jnp.float32, 0.0, 1.0)

    def action_space(self) -> Space:
        return Space((), jnp.int32, 0, 1)

    def init(self, key):
        state = CorridorState(position=jnp.int32(0), clock=jnp.int32(0))
        return state, Timestep(
            obs=self.observe(state),
            action=jnp.int32(0),
            reward=jnp.float32(0.0),
            terminated=jnp.bool_(True),
            truncated=jnp.bool_(False),
        )
```

Its `step` moves the agent, sets `terminated` when it arrives and `truncated` when the clock passes `horizon`. A discrete action space is an integer `Space` whose `low` and `high` bound the actions. An adapter for a JAX library looks the same: [`gymnax.py`](gymnax.py) wraps a Gymnax environment in about 90 lines.

### Gymnasium

[`gymnasium.py`](gymnasium.py) runs any Gymnasium vector environment, so a simulator written in Python needs no port. Every step crosses to the host through `io_callback`, which costs about 200 µs, so it suits simulators that take milliseconds per step. The vector environment must reset in `AutoresetMode.SAME_STEP`. `make` sets that up and returns a `Batched` environment:

```bash
python hangar/main.py algorithm=ppo environment=gymnasium/cartpole
```

### C or C++ through FFI

A native environment steps a whole batch in C++ and is called with `jax.ffi.ffi_call`. [`ale/`](ale) and [`peanut_gb/`](peanut_gb) work this way. Each has an `ffi/` folder with the sources and a `build.sh`. On construction, `build` in [`__init__.py`](__init__.py) rebuilds the `.so` when any source listed is newer than it, then `register_targets` registers the FFI targets for CPU and CUDA. From [`ale/__init__.py`](ale/__init__.py):

```python
build(DIRECTORY, "libale_vec.so", ["vectorize.cpp", "ffi.cc"])
lib = load(DIRECTORY, "libale_vec.so")
register_targets(
    lib,
    {
        "ale_init": "ale_ffi_init",
        "ale_step": "ale_ffi_step",
        "ale_close": "ale_ffi_close",
    },
)
```

`build` hides the compiler output, so a failed build shows only the exit status. Run `build.sh` by hand to see the error. ALE does not ship its sources; run `ale/ffi/vendor.sh` once first.

## Register and configure

1. Add a `make(env_id, **kwargs)` function to your module and put it in `registry` in [`__init__.py`](__init__.py). `environments.make(namespace, env_id, kwargs=...)` looks up the namespace and passes `kwargs` on.
2. Add a config file under [`hangar/config/environment/`](../../hangar/config/environment). Its path is what you pass on the command line. [`gymnax/minatar/breakout.yaml`](../../hangar/config/environment/gymnax/minatar/breakout.yaml) is

   ```yaml
   namespace: gymnax
   suite: minatar
   env_id: Breakout-MinAtar
   ```

   `suite` is optional and defaults to the namespace. Recipes are chosen by `(algorithm, namespace, suite)`. `num_envs` usually comes from the algorithm config, and anything under `kwargs` goes to `make`.
3. Write or reuse a recipe for it. See [recipes](../../hangar/recipes/README.md).

## Tests

[`tests/test_environments.py`](../../tests/test_environments.py) builds a stack for every wrapper in `STACKS`, around the `Dial` dummy. `test_the_stacks_cover_every_wrapper` fails until a new wrapper is in that list. The stacks then check that `update` reaches the game and that the action mask shows through. A new environment gets a smoke case in `CASES` in [`tests/test_recipes.py`](../../tests/test_recipes.py). Mark it with `installed(...)` if it needs an optional package, so it skips where the package is missing.
