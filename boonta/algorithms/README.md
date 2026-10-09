# algorithms

An algorithm decides what to do and learns from what happened. It never touches the environment or the devices. A [podracer](../podracers/README.md) runs the loop and calls three methods, defined in [`algorithm.py`](algorithm.py).

| Method | Called | Returns |
| --- | --- | --- |
| `init(key, timestep)` | once, with the first timestep | the algorithm state |
| `step(state, key, timestep, temperature)` | every environment step | `(state, action, aux)` |
| `update(state, key, transitions)` | once per batch | the new state |

Everything is pure JAX, so all three run inside `jax.jit`.

## What an algorithm receives

The network and the optimizer are passed in, not built inside. A recipe builds them and hands them over, so the same `PPO` runs a CNN on MinAtar or an MLP on CartPole. Algorithms never import environments.

The network is the policy. Its head returns a distrax distribution, so `step` samples from what the network returns. There is no separate policy object. See [networks](../networks/README.md) for the heads.

`temperature` reaches the head. `step` passes it on, and `update` always uses `temperature=1.0`. Podracers train at 1.0 and evaluate at 0.0, which makes the policy greedy. A head called without a temperature raises.

## The state

The state is a frozen `flax.struct.dataclass`. It needs a `step` field, because the podracer adds the number of environment steps (or samples, offline) to it. From [`ppo.py`](ppo.py):

```python
@struct.dataclass(frozen=True)
class PPOState:
    step: Array
    params: PyTree
    optimizer_state: optax.OptState
```

A field that holds one row per environment is marked `struct.field(metadata={"axis": "data"})`. It is then sharded with the environments across devices. Everything else is replicated. The recurrent carry and the replay buffers in [`recurrent_dqn.py`](recurrent_dqn.py) are marked this way.

## What update receives

`transitions` is a `Transition(first, second, aux)`. `first` is the timestep the agent acted on. `second` is the timestep that followed, and it holds the action taken and the reward. `aux` is the dict `step` returned, such as `log_prob` and `value`. On `anakin` and `sebulba` every leaf has shape `(num_steps, num_envs, ...)`. On `quadinaros` the leading shape is the batch shape the dataset samples.

On-policy advantages come from [`advantage_estimators.generalized_advantage_estimation`](advantage_estimators.py). It accumulates in fp32 whatever the critic's dtype.

## Auxiliary losses

Every algorithm takes `auxiliary_losses: tuple[Callable, ...]` and adds each one to its loss. Each loss is called with keyword arguments.

- Every algorithm passes `params`, `apply`, `transitions`, `dist` and `variables`.
- Those with a critic pass `value`. Those with Q-values pass `q_values`.
- Recurrent algorithms pass `carry`.
- SAC, REPPO and IQL pass their actor's values.

A loss takes `**kwargs` and reads only what it needs. `variables` holds every collection the forward pass returned, so `variables["intermediates"]["features"]` is what the head saw. [`auxiliary_losses.py`](auxiliary_losses.py) has two real ones, `DR3` and `Anchor`. A minimal loss, from [`tests/test_auxiliary_losses.py`](../../tests/test_auxiliary_losses.py):

```python
def energy(weight):
    def loss(variables, **kwargs):
        features = variables["intermediates"]["features"].astype(jnp.float32)
        return weight * jnp.mean(features**2)

    return loss
```

## Recurrent algorithms

Recurrent algorithms (`recurrent_*.py`) keep two carries. `carry` is the live one that `step` advances. `rollout_carry` is the carry the current rollout started from. `update` replays the rollout through the network from `rollout_carry`, then sets `rollout_carry` to `carry`, where the next rollout starts. See the end of `RecurrentPPO.update` in [`recurrent_ppo.py`](recurrent_ppo.py).

## Add an algorithm

The steps below follow `BC`, the smallest algorithm, in [`bc.py`](bc.py).

1. Write `boonta/algorithms/<name>.py` with a config dataclass, a state dataclass and the algorithm class. The class takes its config, network and optimizer as fields:

   ```python
   @dataclass
   class BC:
       cfg: BCConfig
       network: nn.Module
       optimizer: optax.GradientTransformation
       auxiliary_losses: tuple[Callable, ...] = ()

       def init(self, key: Key, timestep: Timestep) -> BCState:
           params = self.network.init(key, timestep.obs, temperature=1.0)
           return BCState(
               step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
               params=params,
               optimizer_state=self.optimizer.init(params["params"]),
           )

       def step(
           self, state: BCState, key: Key, timestep: Timestep, temperature: float = 1.0
       ) -> tuple[BCState, Array, PyTree]:
           dist = self.network.apply(state.params, timestep.obs, temperature=temperature)
           return state, dist.sample(seed=key), {}
   ```

   In `update`, apply the network with `mutable=True` and pass the variables it returns to every auxiliary loss as `variables`. After the optimizer step, store back only the collections the network had at init, so a layer that keeps state, such as BatchNorm's `batch_stats`, carries it forward and nothing sown is kept:

   ```python
   variables = {name: variables.get(name, value) for name, value in state.params.items()}
   params = {**variables, "params": optax.apply_updates(state.params["params"], updates)}
   ```

   Acting and target computations apply the network without `mutable`. A target network averages or copies `"params"` as before and copies every other collection from the online network. Log with `lox.log({...})`.

2. Export the class in [`__init__.py`](__init__.py).

3. Add `hangar/config/algorithm/<name>.yaml`. It is a `# @package _global_` file. `algorithm._target_` names the config dataclass, and the file also sets the batch keys the podracer reads (`rollout.num_steps`, `environment.num_envs`) and `optimizer`. An offline algorithm picks its dataset and podracer here, as [`bc.yaml`](../../hangar/config/algorithm/bc.yaml) does:

   ```yaml
   # @package _global_

   defaults:
     - /dataset: minari/mujoco/expert
     - override /podracer: quadinaros

   algorithm:
     _target_: boonta.algorithms.bc.BCConfig
     batch_size: 256
     entropy_coefficient: 0.0

   environment:
     num_envs: 32

   optimizer:
     lr: 3e-4
   ```

4. Write a recipe that builds the network and the environment and registers the pair. See [recipes](../../hangar/recipes/README.md).

5. Add the tests.
   - Add a builder to [`tests/zoo.py`](../../tests/zoo.py), next to `zoo.bc`. It builds the algorithm small, on a dummy environment from [`tests/dummies.py`](../../tests/dummies.py).
   - Add a line to `LEARNERS` in [`tests/test_algorithms.py`](../../tests/test_algorithms.py), for example `pytest.param(zoo.bc, corridor, 200, id="bc-corridor")`. `test_learns` trains for that many updates and checks the return reaches the task's `solved`. Each dummy task fails untrained, so a broken algorithm cannot pass.
   - Add it to `EVERY` in [`tests/test_auxiliary_losses.py`](../../tests/test_auxiliary_losses.py), with the extra keywords it passes (`{"value"}`, `{"q_values", "carry"}` and so on).
   - Add it to the other lists that apply in `test_algorithms.py`: `RECURRENT` for a carry, `REPLAYS` for a replay buffer, `BOUNDED` for targets that must stop at episode ends, `REPLAYERS` for on-policy algorithms whose update must recompute the log-probs or Q-values the rollout acted on.
   - Add a smoke case for the recipe to `CASES` in [`tests/test_recipes.py`](../../tests/test_recipes.py).

Run one case with

```bash
.venv/bin/python -m pytest "tests/test_algorithms.py::test_learns[bc-corridor]" -q
```

## Wrappers

[`wrappers/`](wrappers) wrap an algorithm in the same interface. `Population` runs N copies on slices of the batch. `PSRO` wraps a `Population` and keeps a pool, payoffs and opponent choice for self-play leagues. `sebulba` rejects both.
