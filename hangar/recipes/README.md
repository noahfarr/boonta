# recipes

A recipe turns a config into the parts of a run. It is the only place where the packages meet. Algorithms never import environments, environments never import networks, and podracers only call the methods of what they are given. A recipe builds an environment, a network, an optimizer and an algorithm, and hands them to a podracer.

There is one file per algorithm and suite, named `<algorithm>_<suite>.py`, such as [`ppo_minatar.py`](ppo_minatar.py). Each file is plain and readable on its own. There is no shared helper layer.

## How a recipe is picked

[`__init__.py`](__init__.py) holds `register`, a dict from `(algorithm, namespace, suite)` to a recipe's `make`:

```python
("ppo", "gymnax", "minatar"): ppo_minatar.make,
```

- `algorithm` is the Hydra group choice, the name you pass as `algorithm=ppo`. It is read from `HydraConfig`, not from the config body.
- `namespace` and `suite` come from the environment config. `suite` defaults to `namespace`.

`recipes.make(cfg)` looks up the recipe, calls it, and passes what it returns to the podracer:

```python
def make(cfg):
    namespace = cfg.environment.namespace
    suite = cfg.environment.get("suite", namespace)
    name = HydraConfig.get().runtime.choices["algorithm"]
    components = register[(name, namespace, suite)](cfg)
    return instantiate(cfg.podracer)(**components)
```

## What make(cfg) returns

A dict that is splatted into the podracer's `make`. `algorithm`, `environment`, `pit` and `lap` are always there; `pit` and `lap` come from the curriculum. Any other key becomes a podracer argument, such as `dataset` for `quadinaros`. See [podracers](../../boonta/podracers/README.md).

## A recipe, step by step

[`ppo_minatar.py`](ppo_minatar.py) is a whole recipe in 50 lines.

1. Build the environment from its config, then wrap it in the auto-reset its config chooses, which also vectorizes it. `environment/auto_reset` defaults to `same_step`.

   ```python
   env = environments.make(**cfg.environment)
   num_actions = env.action_space().num_actions
   env = instantiate(cfg.environment.auto_reset, env)
   ```

2. Build the network. The head is the policy, here a `Categorical` over the actions next to a value head.

   ```python
   network = Network(
       feature_extractor=FeatureExtractor(
           observation_extractor=nn.Sequential(
               [
                   lambda obs: obs.astype(jnp.float32),
                   nn.Conv(16, (3, 3), padding="VALID"),
                   nn.relu,
                   Flatten(start_dim=-3),
                   nn.Dense(128),
                   nn.relu,
               ]
           ),
       ),
       head=ActorCritic(
           actor=Categorical(nn.Dense(num_actions)),
           critic=nn.Dense(1),
       ),
   )
   ```

3. Build the algorithm. `instantiate(cfg.algorithm)` makes the config dataclass named in `hangar/config/algorithm/ppo.yaml`. The optimizer is built here from `cfg.optimizer`.

   ```python
   algorithm = PPO(
       cfg=instantiate(cfg.algorithm),
       network=network,
       optimizer=optax.chain(
           optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
           optax.adam(cfg.optimizer.lr),
       ),
   )
   ```

4. Apply the curriculum, then the wrappers that record or reshape what the agent experiences. The curriculum always gets a vectorized, auto-resetting environment and only wraps the outside, so the statistics see the levels it chose.

   ```python
   algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
   env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
   return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
   ```

A learning rate that anneals comes from `learning_rate(cfg, batch_size)` in [`schedules.py`](schedules.py). Recipes do not build their own cosine schedule; a test checks this.

## Add a recipe

1. Copy the closest recipe to `<algorithm>_<suite>.py` and change what differs.
2. Import it in [`__init__.py`](__init__.py) and add its `(algorithm, namespace, suite)` key to `register`. One recipe can serve several suites, as `ppo_isaac_lab.make` does.
3. Add a smoke case to `CASES` in [`tests/test_recipes.py`](../../tests/test_recipes.py):

   ```python
   ("ppo", "gymnax/minatar/breakout", None, False),
   ```

   The third field is a skip reason, such as `installed("brax")`, and the fourth builds a dataset for offline recipes. `test_every_recipe_builds_and_runs_one_update` builds the recipe with 32 environments and runs one update. `test_every_recipe_has_a_smoke_case` fails if a registered recipe has no case.

## Hyperparameters

`hangar/config/hyperparameters/<algorithm>/<environment path>.yaml` is applied on top of the algorithm, environment and curriculum configs. Each file starts with `# @package _global_`, so it can set any key. The `cascade` resolver in [`resolvers.py`](../resolvers.py) walks up the environment path until it finds a file, and falls back to `hyperparameters/<algorithm>.yaml`. With `environment=gymnax/minatar/breakout` it tries

1. `hyperparameters/ppo/gymnax/minatar/breakout.yaml`
2. `hyperparameters/ppo/gymnax/minatar.yaml`
3. `hyperparameters/ppo/gymnax.yaml`
4. `hyperparameters/ppo.yaml`

and uses the first that exists. Here it is the second, which sets `num_steps: 20_000_000` for every MinAtar game. A missing file is not an error.

Before each environment file it tries the same path with the curriculum appended, so `curriculum=plr` with `environment=kinetix/kinetix` finds `hyperparameters/ppo/kinetix/plr.yaml`. That file pulls in `hyperparameters/ppo/kinetix.yaml` through its defaults and sets what PLR needs on Kinetix, the level `generate_level` factory. A recipe never passes curriculum arguments: it calls `instantiate(cfg.curriculum)(algorithm, env)`, and Hydra builds the factories when the curriculum is instantiated. The default curriculum has no such files, so it resolves exactly as before.

Run the recipe with

```bash
uv run boonta train algorithm=ppo environment=gymnax/minatar/breakout
```

## Sokoban PPO encoder

`algorithm=ppo environment=jumanji/sokoban` defaults to
`network.encoder=one_hot`. It one-hot encodes the variable layer (5 tile IDs) and
fixed layer (3 tile IDs), then flattens the 10×10 grid into 800 values. A single
clock scalar, divided by the configured puzzle time limit, gives 801 inputs.
The clock is not one-hot encoded or repeated across the board. A two-layer,
256-wide ReLU MLP feeds the policy and value heads. There is no recurrence or
convolution in this option. PPO settings, rewards, and sampling are unchanged.

Start a fresh run because encoder choices change checkpoint shapes:

```bash
boonta train algorithm=ppo environment=jumanji/sokoban network.encoder=one_hot
```

Available comparison encoders:

- `network.encoder=cnn`: four binary board planes plus a normalized clock plane,
  two 3×3 convolutions with 32 and 64 channels and ReLU, followed by a
  `network.encoder_features=256` feature projection with ReLU and the MLP body.
- `network.encoder=flat`: the original raw flattened tile IDs and repeated clock.
- `network.encoder=puffer network.encoder_features=1024`: four binary planes
  flattened in channel-first order and a bias-free 400-to-1,024 linear projection,
  without the clock, followed by the MLP body. This reproduces PufferLib 5.0's
  input encoder without its MinGRU core.

`network.encoder_features` sets the projection width for `cnn` and `puffer`;
`one_hot` and `flat` feed the fixed-width MLP directly. See PufferLib's
[observation representation](https://github.com/PufferAI/PufferLib/blob/5.0/ocean/boxoban/boxoban.h)
and [linear encoder](https://github.com/PufferAI/PufferLib/blob/5.0/src/algo.cu).
