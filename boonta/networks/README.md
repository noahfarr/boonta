# networks

Networks are [Flax](https://flax.readthedocs.io) modules. A recipe builds one and hands it to an [algorithm](../algorithms/README.md), which only calls `init` and `apply`.

## Network

[`Network`](network.py) has three parts.

| Field | Does |
| --- | --- |
| `feature_extractor` | turns the observation (and optionally the last action, reward and done flag) into features |
| `torso` | an optional recurrent part, which takes and returns a carry |
| `head` | turns features into the output, usually a distribution |

Without a torso, `network.apply(params, obs, temperature=t)` returns the head's output. With a torso it returns `(carry, output)`, and its inputs are sequences shaped `(batch, time, ...)`.

## The network is the policy

There is no separate policy object. An actor ends in a head from [`heads.py`](heads.py) that wraps the output layer and returns a distrax distribution.

| Head | Distribution | Temperature means |
| --- | --- | --- |
| `Categorical` | categorical over discrete actions | divides the logits; 0 is greedy |
| `EpsilonGreedy` | ε-greedy over Q-values | ε |
| `Gaussian` | diagonal Gaussian | scales the standard deviation |
| `SquashedGaussian` | Gaussian through `tanh`, bounded | scales the standard deviation |

A head wraps one layer, for example `Categorical(nn.Dense(num_actions))`, and shares that layer's scope. So wrapping a layer keeps its parameter names, and old checkpoints still load.

[`ActorCritic`](actor_critic.py) pairs an actor head with a critic and returns `(distribution, value)`. The value is cast to fp32. From [`hangar/recipes/ppo_minatar.py`](../../hangar/recipes/ppo_minatar.py):

```python
head=ActorCritic(
    actor=Categorical(nn.Dense(num_actions)),
    critic=nn.Dense(1),
),
```

Load-bearing details:

- **Temperature goes to the head only.** `Network` passes `temperature` to its head, and `ActorCritic` passes it to the actor only. A module placed between them must forward it too. One that drops it fails silently: evaluation then runs at temperature 1 and reports the training policy as its greedy score.
- **Heads take no default temperature.** A call that leaves it out raises.
- **`dist.logits` is normalised.** It is the `log_softmax` of the head output. The raw output is `dist.preferences`. DQN and PQN read their Q-values from `preferences`.
- **Distribution parameters are fp32.** The constructors in [`distributions.py`](distributions.py) cast the head output before building the distribution, whatever the network's dtype.

## Torsos, cells and stacks

A recurrent model is built from three config groups in [`hangar/config/`](../../hangar/config).

| Group | Is | Choices |
| --- | --- | --- |
| `cell` | one recurrent step | `gru`, `min_gru`, `rtu`, `self_attention` |
| `torso` | runs a cell over a sequence | set by the cell: `rnn`, `ssm`, or `default` (the cell itself) |
| `stack` | layers of torsos | `default` (one layer), `highway`, `llama`, `repeat` |

Each cell file picks its torso. `gru` and `rtu` run in an [`RNN`](blocks/rnn.py), a sequential scan. `min_gru` runs in an [`SSM`](blocks/ssm.py), a parallel scan. `self_attention` is its own torso. For example:

```bash
uv run boonta algorithm=recurrent_ppo cell=min_gru stack=highway num_layers=2
```

Every torso and stack is a [`Block`](blocks/block.py): `__call__(carry, x, done)` returns `(carry, x)`, and `initialize_carry(key, input_shape)` returns a blank carry. Where `done` is set the block starts again from the blank carry, so episodes never leak into each other.

## Parameter identity

A Flax module instance is a parameter identity. Pass the same block object to two places and they share weights, with no error and no change of shape. Stacks avoid this in one of two ways.

- [`repeat`](stacks/repeat.py) and [`llama`](stacks/llama.py) `copy.deepcopy` the block once per layer.
- [`Tower`](blocks/tower.py) runs one block `num_layers` times with `nn.scan` and `variable_axes={"params": 0}`, which gives every layer its own weights. The layer loop is fully unrolled, and the carry keeps environments first, `(batch, layers, ...)`.

Count the parameters when you build a stack. Every run prints them before training, and a tied layer shows up only there.

## Mixed precision

`dtype` sets the compute dtype and `param_dtype` stays fp32. Use bf16 for anything that enters a matrix multiply and fp32 for anything accumulated or reduced.

Every block that owns an `nn.Dense` must take `dtype` and `param_dtype` and pass them to it, as [`Highway`](blocks/highway.py) and [`MinGRUCell`](blocks/min_gru.py) do. Flax defaults to `dtype=None`, which runs in the fp32 parameter dtype. A block that forgets runs fp32 among bf16 blocks, with no error and no wrong numbers, only lost speed. Stack configs pass it on as `dtype: ${oc.select:network.dtype,null}`.

Attention is the exception. `get_attention_implementation` in [`boonta/utils`](../utils/__init__.py) picks cuDNN attention in bf16 on an NVIDIA GPU of compute capability 8 or more when the head size fits, and XLA attention in fp32 everywhere else, whatever the module's dtype.

## Sown features

`Network` sows the features its head sees as `features`:

```python
self.sow("intermediates", "features", x, reduce_fn=lambda _, value: value)
```

An algorithm's update applies the network with `mutable=True` and hands the sown `intermediates` to its auxiliary losses, which read `intermediates["intermediates"]["features"]`. Acting calls the network without `mutable`, so sowing costs nothing there, and nothing sown reaches the parameters or a checkpoint. Any module can sow more values the same way.

## Stateful layers

Every collection a layer writes during the update's forward pass, such as BatchNorm's `batch_stats`, is stored beside `params` and carried to the next update. Only `params` reaches the optimizer. Acting and evaluation apply the network without `mutable`, so a layer that checks `self.is_mutable_collection(...)` sees training only in the update.

## Add a network part

- **A feature extractor or head** is a plain Flax module. Build it in the recipe. A head should wrap its output layer in one of the heads above.
- **A cell** for `RNN` subclasses `RNNCellBase`: `__call__(carry, x)` returns `(carry, y)`, plus `initialize_carry`. Flax's own `nn.GRUCell` fits as is. A cell for `SSM` subclasses `SSMCellBase`: `__call__(x)` returns the scan's `(a, b)` and `output(carries, x)` reads the result. Add a config file to `hangar/config/cell/` that names the torso it runs in.
- **A torso or stack** subclasses `Block`. Export it in [`__init__.py`](__init__.py).

Then test it.

1. Add a builder to `TORSOS` in [`tests/zoo.py`](../../tests/zoo.py). It takes `dtype=None`.
2. [`tests/test_networks.py`](../../tests/test_networks.py) then checks every torso. Chunked and step-by-step runs must agree, an episode start must forget what came before, bf16 must reach every dense layer while the carry keeps its dtype, and `Network` must forward the temperature past it.
3. [`tests/test_algorithms.py`](../../tests/test_algorithms.py) checks that every recurrent algorithm, run with each torso in `TORSOS`, replays exactly the log-probs and Q-values it acted on.

A new stack also belongs in `test_every_layer_of_a_stack_owns_its_weights`, which checks that each extra layer adds the same number of parameters.
