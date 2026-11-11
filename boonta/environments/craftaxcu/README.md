# Craftax CUDA bindings

Ported from `/home/farr/watto`, branch `optimal-restart-distributions`, into
Boonta's current environment interface. The pinned CUDA source, upstream license
and notice, local patch, build script, and vendor update script are retained.

Requires CUDA-enabled JAX, an NVIDIA GPU, and `nvcc`. The first construction builds
`ffi/libcraftaxcu.so` for the local GPU. Set `CRAFTAXCU_ARCH` to override its
compute capability (for example `120` for sm_120).

```python
import jax
import jax.numpy as jnp
from boonta import environments

env = environments.make(
    'craftaxcu', 'Craftax-Symbolic-Cu', kwargs={'num_envs': 64, 'seed': 0}
)
state, first = jax.jit(env.init)(jax.random.key(0))
state, timestep = jax.jit(env.step)(
    jax.random.key(1), state, jnp.zeros(64, jnp.int32)
)
```

The environment is already batched and the CUDA kernel auto-resets finished
episodes. Do not vmap it or wrap it in `SameStepAutoReset`. Use the existing
`Batched(env, num_envs=...)` wrapper when an API expects a vectorized environment.
Spaces describe one environment; runtime observations have a leading `num_envs`
axis. Observations are `{'view': packed_float32_observations, 'mask': legal_actions}`.
`action_mask(state)` returns the same legal-action mask.

State is a tuple of caller-owned CUDA byte arrays containing world, SoA, and tail
snapshots, including RNG state. `observe`, `legal`, `cells`, `worlds`, `portray`,
`bank`, `revive`, `save`, `restore`, `archive`, and `reseed` retain the source
binding API. FFI calls that operate on rows require the configured batch size;
`portray` also supports smaller or larger batches by chunking. `reseed` refreshes
the RNG bytes of restored snapshots. Native pool lifetime is managed by the
adapter; `close(state)` preserves the source implementation's no-op behavior.

This port contains bindings and environment registration only. No training
recipe, curriculum integration, or renderer has been added. Existing JAX Craftax
support remains available separately as `craftax`.

Run metadata tests on CPU, and CUDA binding tests explicitly on GPU:

```bash
.venv/bin/python -m pytest tests/test_craftaxcu.py -q
JAX_PLATFORMS=cuda .venv/bin/python -m pytest tests/test_craftaxcu.py -q
```

## Environment configuration

The Hydra group `environment=craftaxcu/symbolic` configures 4,096 native CUDA
environments, uses the run's seed, sets `coarse=4`, and selects
`environment/auto_reset=batched` automatically. Override `environment.num_envs`
to change both the CUDA batch and its wrapper. The config is at
`hangar/config/environment/craftaxcu/symbolic.yaml`.

This selects and instantiates the environment bindings. Training still requires
a recipe registered for `(algorithm, craftaxcu, craftaxcu)`; this bindings-only
port does not register one, so selecting this config alone is not a complete
`boonta train` command.
