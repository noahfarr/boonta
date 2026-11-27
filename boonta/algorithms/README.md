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

A field that holds one row per environment is marked `struct.field(metadata={"axis": "batch"})`. It is then sharded with the environments across devices. Everything else is replicated. The replay buffers in [`recurrent_dqn.py`](recurrent_dqn.py) are marked this way. The recurrent carry is not part of the state. It belongs to the rollout, and the podracer holds it.

## What update receives

`transitions` is a `Transition(first, second, aux)`. `first` is the timestep the agent acted on. `second` is the timestep that followed, and it holds the action taken and the reward. `aux` is the dict `step` returned, such as `log_prob` and `value`. On `anakin` and `sebulba` every leaf has shape `(num_steps, num_envs, ...)`. On `quadinaros` the leading shape is the batch shape the dataset samples.

On-policy advantages come from [`advantage_estimators.generalized_advantage_estimator`](advantage_estimators.py). It accumulates in fp32 whatever the critic's dtype.

## Minibatches across devices

On several devices, an update should learn from the data each device already holds. If a minibatch mixes entries from every device, XLA copies the batch to every device and each device computes the whole update.

Each algorithm therefore cuts its batch into blocks of `num_minibatches` consecutive entries and gives each minibatch one entry of every block, drawn by a random permutation per block. PPO, PQN, GRPO, MMD and REPPO lay the batch out environment by environment, so a block is `num_minibatches` consecutive steps of one environment. Recurrent PPO, PQN and GRPO cut the environments into blocks and keep whole trajectories together with their initial carries. When every device holds whole blocks, each device holds its share of every minibatch, and XLA only exchanges gradients. A run then gives the same result on any number of devices, up to floating-point rounding, which `test_every_device_learns_from_its_share_of_the_batch` checks.

Replay buffers from [`boonta.buffers.make_blockwise_buffer`](../buffers/blockwise_buffer.py) sample in the same spirit. It is flashbax's trajectory buffer with a different `sample`. The environment rows are cut into `gcd(sample_batch_size, num_envs)` contiguous blocks, and each block samples its share of the batch with flashbax's own sampler. The episode buffers still sample over the whole buffer, so they compute their update on every device.

Each device holds whole blocks when

- `num_envs * num_steps / devices` is a multiple of `num_minibatches` for PPO, PQN, GRPO, MMD and REPPO,
- `num_envs / devices` is a multiple of `num_minibatches` for the recurrent algorithms,
- `gcd(sample_batch_size, num_envs)` is a multiple of the number of devices for replay buffers.

Otherwise training is still correct, but XLA moves data between devices.

## Auxiliary losses

Every algorithm takes `auxiliary_losses: tuple[Callable, ...]` and adds each one to its loss. Each loss is called with keyword arguments.

- Every algorithm passes `params`, `apply`, `transitions`, `dist` and `variables`.
- Those with a critic pass `value`. Those with Q-values pass `q_values`.
- Recurrent algorithms pass `carry`, the carry that was live when the loss ran.
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

The recurrent carry is rollout state, like the environment state, so the podracer owns it and not the algorithm. `init(key, timestep)` returns `(state, carry)`, where the carry is what a fresh rollout starts from, and `None` for an algorithm without one. `step(state, carry, key, timestep, temperature)` returns `(state, carry, action, aux)`. The podracer records the carry a rollout starts from and the carry it ends with in `Transition.carry`, so a rollout holds two carries however long it is. A recurrent `update` replays the rollout from `transitions.start`, and `transitions.tail()` returns the final timestep and the carry the rollout ended with, which is enough to bootstrap the final value. Because the carry is not in `algorithm_state`, a checkpoint holds only what was learned, and an evaluation starts every episode from a fresh carry. See `RecurrentPPO` in [`recurrent_ppo.py`](recurrent_ppo.py).

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

3. Add `hangar/config/algorithm/<name>.yaml`. It is a `# @package _global_` file. `algorithm._target_` names the config dataclass, and the file also sets the batch keys the podracer reads (`podracer.config.num_steps`, `environment.num_envs`) and `optimizer`. An offline algorithm picks its dataset and podracer here, as [`bc.yaml`](../../hangar/config/algorithm/bc.yaml) does:

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

[`wrappers/`](wrappers) wrap an algorithm in the same interface. `Population` runs N copies on slices of the batch. `Roster` is a `Population` that can start its copies from given `initial_params`, keeps their initial states and can anchor each one near them. The [`psro`](../curricula/psro.py) curriculum trains a `Roster`, and its `PSRO` pit keeps the pool, the payoffs and the opponent choice. `sebulba` rejects both. `RND` adds a random network distillation bonus, scored by its own critic and added to the advantage through the `advantage_estimator` of `PPO` or `RecurrentPPO`. It wraps the algorithm directly, inside any other wrapper.

## EGGROLL

[EGGROLL](https://arxiv.org/abs/2511.16652) trains a policy by scoring
perturbed weights on rollouts. Each update is one generation. The state keeps
the generation's `noise_key`, from which `step` draws the mirrored low-rank
perturbations and `update` redraws them; `update` then stores its own key as the
next generation's. Fitness is the first episode of each rollout, so EGGROLL
needs every rollout to start fresh episodes: wrap the environment in
`Group(num_steps=<rollout length>)` after its auto-reset, as the recipe does.

The network is the policy, as for every algorithm: a head such as `Categorical`
or `SquashedGaussian` around a `LowRankPerturbedMLP`, whose dense layers share
one projection and add each pair's perturbation as two low-rank projections.
EGGROLL samples the head at `algorithm.temperature`, so the default of 0 plays
the deterministic policy and 1 a stochastic one. The podracer's temperature
scales both the policy temperature and the perturbation, so evaluation at
temperature 0 plays the greedy, unperturbed policy.

```sh
python -m hangar.train algorithm=eggroll environment=gymnax/classic_control/cartpole
```

The recipe covers single-agent Gymnax classic control, MinAtar and Brax. It
picks `Categorical` for discrete actions and `SquashedGaussian` for continuous
ones, whose actions it rescales from [-1, 1] to the environment's bounds.
`algorithm.population_size` must be even and divide `environment.num_envs`.
Each candidate gets `num_envs / population_size` rollouts, and fitness
averages their first episode returns up to `podracer.config.num_steps`. For
Brax, set that horizon to 1000 to score full-length episodes under the default
environment limit.
