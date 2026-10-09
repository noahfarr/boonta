from types import SimpleNamespace

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import zoo
from boonta.networks import (RNN, ActorCritic, Categorical, EpsilonGreedy,
                             FeatureExtractor, Gaussian, Highway,
                             GatedDeltaNet, LinearAttention,
                             LinearAttentionInputs, Network, RTUCell,
                             SquashedGaussian, Tower, distributions, llama,
                             repeat)
from boonta.networks.blocks.linear_attention import chunkwise, recurrent
from boonta.networks.blocks.ssm import binary_operator
from boonta.networks.kernels import get_scan_implementation
from boonta.utils import get_attention_implementation, load_qwen3_5

BATCH, LENGTH, CHUNK = 3, 8, 4
KEY = jax.random.key(0)


def sequence(dtype=jnp.float32):
    return jax.random.normal(jax.random.key(1), (BATCH, LENGTH, zoo.WIDTH)).astype(dtype)


def starts(*steps):
    done = jnp.zeros((BATCH, LENGTH), bool).at[:, 0].set(True)
    for step in steps:
        done = done.at[:, step].set(True)
    return done


def run(torso, params, x, done, size):
    carry = torso.initialize_carry(KEY, (BATCH, zoo.WIDTH))
    outputs = []
    for start in range(0, x.shape[1], size):
        carry, output = torso.apply(
            params, carry, x[:, start : start + size], done[:, start : start + size]
        )
        outputs.append(output)
    return carry, jnp.concatenate(outputs, axis=1)


def build(name, dtype=None):
    torso = zoo.TORSOS[name](dtype=dtype)
    x, done = sequence(dtype or jnp.float32), starts()
    carry = torso.initialize_carry(KEY, (BATCH, zoo.WIDTH))
    return torso, torso.init(jax.random.key(2), carry, x[:, :CHUNK], done[:, :CHUNK])


@pytest.mark.parametrize("name", zoo.TORSOS)
def test_chunks_and_single_steps_give_the_same_outputs(name):
    torso, params = build(name)
    x, done = sequence(), starts()
    _, chunked = run(torso, params, x, done, CHUNK)
    _, stepped = run(torso, params, x, done, 1)
    np.testing.assert_allclose(chunked, stepped, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("name", zoo.TORSOS)
def test_an_episode_start_forgets_everything_before_it(name):
    torso, params = build(name)
    x = sequence()
    _, outputs = run(torso, params, x, starts(5), 1)
    _, fresh = run(torso, params, x[:, 5:], starts()[:, : LENGTH - 5], 1)
    np.testing.assert_allclose(outputs[:, 5:], fresh, rtol=1e-4, atol=1e-5)


def dense_dtypes(torso, params, *inputs):
    seen = []

    def record(call, args, kwargs, context):
        if isinstance(context.module, (nn.Dense, nn.DenseGeneral)):
            seen.append(context.module.dtype)
        return call(*args, **kwargs)

    with nn.intercept_methods(record):
        torso.apply(params, *inputs)
    return seen


@pytest.mark.parametrize("name", zoo.TORSOS)
def test_half_precision_runs_every_dense_in_half_and_keeps_the_carry_dtype(name):
    torso, params = build(name, jnp.bfloat16)
    x, done = sequence(jnp.bfloat16), starts()
    initial = torso.initialize_carry(KEY, (BATCH, zoo.WIDTH))

    dtypes = dense_dtypes(torso, params, initial, x[:, :CHUNK], done[:, :CHUNK])
    assert dtypes and all(dtype == jnp.bfloat16 for dtype in dtypes)
    carry, _ = run(torso, params, x, done, CHUNK)
    assert jax.tree.map(lambda leaf: leaf.dtype, carry) == jax.tree.map(
        lambda leaf: leaf.dtype, initial
    )


@pytest.mark.xfail(
    reason="a GRU returns fp32 activations, which Tower cannot scan as a bf16 carry",
    strict=True,
)
def test_a_highway_of_grus_runs_in_half_precision():
    torso = Tower(Highway(zoo.gru(jnp.bfloat16), dtype=jnp.bfloat16), num_layers=2)
    x, done = sequence(jnp.bfloat16), starts()
    carry = torso.initialize_carry(KEY, (BATCH, zoo.WIDTH))
    params = torso.init(jax.random.key(2), carry, x[:, :CHUNK], done[:, :CHUNK])
    run(torso, params, x, done, CHUNK)


def count(stack):
    x, done = sequence()[:, :CHUNK], starts()[:, :CHUNK]
    carry = stack.initialize_carry(KEY, (BATCH, zoo.WIDTH))
    params = stack.init(jax.random.key(1), carry, x, done)
    return sum(leaf.size for leaf in jax.tree.leaves(params))


@pytest.mark.parametrize(
    "stack",
    [
        lambda num_layers: repeat(zoo.gru(), num_layers),
        lambda num_layers: llama((zoo.gru(),), num_layers, zoo.WIDTH),
        lambda num_layers: llama((zoo.gru(),) * 2, num_layers, zoo.WIDTH),
        lambda num_layers: Tower(Highway(zoo.gru()), num_layers),
    ],
    ids=["repeat", "llama", "llama-pattern", "tower"],
)
def test_every_layer_of_a_stack_owns_its_weights(stack):
    one, two, three = (count(stack(num_layers)) for num_layers in (1, 2, 3))
    assert two - one == three - two > 0


def reference(a, b, init):
    cumulative_a, cumulative_b = jax.lax.associative_scan(binary_operator, (a, b), axis=1)
    return cumulative_b + cumulative_a * init[:, None]


def scans():
    names = ["associative", "sequential"]
    if jax.default_backend() == "gpu":
        names.append("pallas")
    return names


@pytest.mark.parametrize("name", scans())
def test_every_scan_matches_the_reference(name):
    keys = jax.random.split(jax.random.key(3), 3)
    a = jax.random.uniform(keys[0], (4, 16, 128)) * 0.9 + 0.05
    b = jax.random.uniform(keys[1], (4, 16, 128))
    init = jax.random.normal(keys[2], (4, 128))
    scan = get_scan_implementation(name)

    np.testing.assert_allclose(scan(a, b, init), reference(a, b, init), atol=1e-5)
    weight = jax.random.normal(jax.random.key(4), a.shape)
    mine = jax.grad(lambda *x: (scan(*x) * weight).sum(), argnums=(0, 1, 2))(a, b, init)
    want = jax.grad(lambda *x: (reference(*x) * weight).sum(), argnums=(0, 1, 2))(a, b, init)
    for got, expected in zip(mine, want):
        np.testing.assert_allclose(got, expected, atol=1e-4)


def test_the_pallas_scan_hands_widths_its_blocks_cannot_tile_to_the_associative_scan():
    keys = jax.random.split(jax.random.key(6), 3)
    a = jax.random.uniform(keys[0], (2, 8, 100), jnp.bfloat16)
    b = jax.random.uniform(keys[1], (2, 8, 100), jnp.bfloat16)
    init = jax.random.normal(keys[2], (2, 100), jnp.bfloat16)
    out = get_scan_implementation("pallas")(a, b, init)
    assert out.dtype == jnp.bfloat16
    exact = reference(*(x.astype(jnp.float32) for x in (a, b, init)))
    np.testing.assert_allclose(out.astype(jnp.float32), exact, rtol=1e-2, atol=1e-2)


def half_problem():
    keys = jax.random.split(jax.random.key(5), 3)
    a = jax.random.uniform(keys[0], (8, 64, 256)) * 0.9 + 0.05
    b = jax.random.uniform(keys[1], (8, 64, 256))
    init = jax.random.normal(keys[2], (8, 256))
    half = tuple(x.astype(jnp.bfloat16) for x in (a, b, init))
    return half, reference(*(x.astype(jnp.float32) for x in half))


def accumulated_in_half(a, b, init):
    def step(carry, inputs):
        decay, value = inputs
        carry = decay * carry + value
        return carry, carry

    _, outputs = jax.lax.scan(step, init, (jnp.moveaxis(a, 1, 0), jnp.moveaxis(b, 1, 0)))
    return jnp.moveaxis(outputs, 0, 1)


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="the pallas scan needs a gpu")
def test_the_pallas_scan_stores_half_but_accumulates_in_single_precision():
    half, exact = half_problem()
    pallas = get_scan_implementation("pallas")(*half).astype(jnp.float32)
    drifted = accumulated_in_half(*half).astype(jnp.float32)
    assert np.max(np.abs(pallas - exact) / np.abs(exact)) <= float(jnp.finfo(jnp.bfloat16).eps)
    assert np.abs(pallas - exact).max() < np.abs(drifted - exact).max()


def test_an_rtu_remembers_an_impulse_for_many_steps():
    rnn = RNN(cell=RTUCell(features=8, r_min=0.9, r_max=0.999))
    impulse = jnp.zeros((1, 64, 3)).at[:, 0].set(1.0)
    done = jnp.zeros((1, 64), bool)
    params = rnn.init(jax.random.key(1), None, impulse, done)
    _, outputs = rnn.apply(params, None, impulse, done)
    assert float(jnp.abs(outputs[:, -1]).max()) > 1e-3


@pytest.mark.parametrize("name", [name for name in scans() if name != "associative"])
def test_a_scan_that_stores_half_returns_half(name):
    half = jnp.full((4, 16, 128), 0.5, jnp.bfloat16)
    init = jnp.zeros((4, 128), jnp.bfloat16)
    assert get_scan_implementation(name)(half, half, init).dtype == jnp.bfloat16


def logits():
    return jax.random.normal(jax.random.key(1), (256, 5)) * 2.0


def test_categorical_divides_by_temperature_and_is_greedy_at_zero():
    raw = logits()
    np.testing.assert_allclose(
        distributions.categorical(raw, 0.25).logits,
        jax.nn.log_softmax(raw / 0.25),
        rtol=1e-6,
        atol=1e-6,
    )
    greedy = distributions.categorical(raw, 0.0)
    action, log_prob = greedy.sample_and_log_prob(seed=KEY)
    np.testing.assert_array_equal(action, jnp.argmax(raw, axis=-1))
    np.testing.assert_array_equal(log_prob, 0.0)
    np.testing.assert_array_equal(greedy.entropy(), 0.0)


def test_categorical_keeps_the_raw_output_as_preferences():
    raw = logits()
    distribution = distributions.categorical(raw, 0.5)
    np.testing.assert_array_equal(distribution.preferences, raw)
    np.testing.assert_array_equal(
        distributions.categorical(raw).logits, jax.nn.log_softmax(raw)
    )


def test_categorical_accepts_a_traced_temperature():
    raw = logits()
    sample = jax.jit(lambda temperature: distributions.categorical(raw, temperature).sample(seed=KEY))
    np.testing.assert_array_equal(sample(0.0), jnp.argmax(raw, axis=-1))


def test_distributions_compute_in_single_precision():
    half = logits().astype(jnp.bfloat16)
    assert distributions.categorical(half).logits.dtype == jnp.float32
    assert distributions.gaussian(half[:, :4]).mean().dtype == jnp.float32


def test_independent_categoricals_sum_over_the_reinterpreted_axis():
    raw = logits().reshape(64, 4, 5)
    distribution = distributions.categorical(raw, reinterpreted_batch_ndims=1)
    action = distribution.sample(seed=KEY)
    assert action.shape == (64, 4)
    expected = jnp.take_along_axis(jax.nn.log_softmax(raw), action[..., None], -1)
    np.testing.assert_allclose(distribution.log_prob(action), expected[..., 0].sum(-1), rtol=1e-6)


def test_epsilon_greedy_temperature_is_epsilon():
    q_values = logits()
    greedy = distributions.epsilon_greedy(q_values, 0.0)
    np.testing.assert_array_equal(greedy.preferences, q_values)
    np.testing.assert_array_equal(greedy.sample(seed=KEY), jnp.argmax(q_values, -1))
    np.testing.assert_allclose(distributions.epsilon_greedy(q_values, 1.0).probs, 0.2)


def test_gaussian_temperature_scales_the_deviation():
    outputs = jax.random.normal(jax.random.key(2), (32, 6))
    mean, scale = jnp.split(outputs, 2, axis=-1)
    distribution = distributions.gaussian(outputs)
    np.testing.assert_allclose(distribution.scale_diag, jax.nn.softplus(scale) + 1e-3)
    np.testing.assert_allclose(
        distributions.gaussian(outputs, 0.5).scale_diag, 0.5 * distribution.scale_diag
    )
    np.testing.assert_array_equal(distributions.gaussian(outputs, 0.0).sample(seed=KEY), mean)


def test_squashed_gaussian_is_bounded_and_greedy_at_zero():
    outputs = jax.random.normal(jax.random.key(3), (32, 6))
    mean, _ = jnp.split(outputs, 2, axis=-1)
    assert jnp.all(jnp.abs(distributions.squashed_gaussian(outputs).sample(seed=KEY)) < 1.0)
    np.testing.assert_array_equal(
        distributions.squashed_gaussian(outputs, 0.0).sample(seed=KEY), jnp.tanh(mean)
    )


@pytest.mark.parametrize(
    "head, width",
    [(Categorical, 5), (EpsilonGreedy, 5), (Gaussian, 6), (SquashedGaussian, 6)],
    ids=["categorical", "epsilon_greedy", "gaussian", "squashed_gaussian"],
)
def test_a_head_keeps_its_layer_parameters(head, width):
    x = jax.random.normal(jax.random.key(5), (3, 8))
    plain = Network(head=nn.Dense(width)).init(jax.random.key(4), x)
    wrapped = Network(head=head(nn.Dense(width))).init(jax.random.key(4), x, temperature=1.0)
    jax.tree.map(np.testing.assert_array_equal, plain, wrapped)


def test_actor_critic_tempers_only_the_actor():
    x = jax.random.normal(jax.random.key(6), (3, 8))
    network = Network(head=ActorCritic(actor=Categorical(nn.Dense(5)), critic=nn.Dense(1)))
    params = network.init(jax.random.key(4), x, temperature=1.0)
    distribution, value = network.apply(params, x, temperature=1.0)
    greedy, greedy_value = network.apply(params, x, temperature=0.0)
    np.testing.assert_array_equal(greedy.sample(seed=KEY), jnp.argmax(distribution.logits, axis=-1))
    np.testing.assert_array_equal(value, greedy_value)


def test_a_half_precision_critic_reports_its_value_in_single_precision():
    x = jax.random.normal(jax.random.key(6), (3, 8))
    critic = nn.Dense(1, dtype=jnp.bfloat16)
    network = Network(head=ActorCritic(actor=Categorical(nn.Dense(5)), critic=critic))
    params = network.init(jax.random.key(4), x, temperature=1.0)
    _, value = network.apply(params, x, temperature=1.0)
    assert value.dtype == jnp.float32


@pytest.mark.parametrize("name", zoo.TORSOS)
def test_the_network_forwards_temperature_past_every_torso(name):
    network = Network(
        feature_extractor=FeatureExtractor(nn.Dense(zoo.WIDTH)),
        torso=zoo.TORSOS[name](),
        head=Categorical(nn.Dense(5)),
    )
    x = sequence()[:, :CHUNK]
    action = jnp.zeros((BATCH, CHUNK), jnp.int32)
    reward = jnp.zeros((BATCH, CHUNK))
    done = starts()[:, :CHUNK]
    carry = network.initialize_carry(KEY, (BATCH, zoo.WIDTH))
    inputs = (x, action, reward, done)
    params = network.init(jax.random.key(9), *inputs, carry=carry, temperature=1.0)
    _, distribution = network.apply(params, *inputs, carry=carry, temperature=1.0)
    _, greedy = network.apply(params, *inputs, carry=carry, temperature=0.0)
    np.testing.assert_array_equal(
        greedy.sample(seed=KEY), jnp.argmax(distribution.logits, axis=-1)
    )


def plain():
    return Network(feature_extractor=FeatureExtractor(nn.Dense(8)), head=nn.Dense(2))


def test_init_keeps_only_parameters():
    assert list(plain().init(jax.random.key(0), jnp.ones((3, 4)))) == ["params"]


def test_the_forward_pass_sows_the_features_the_head_sees():
    obs = jnp.ones((3, 4))
    network = plain()
    variables = network.init(jax.random.key(0), obs)
    output, intermediates = network.apply(variables, obs, mutable="intermediates")
    features = intermediates["intermediates"]["features"]
    head = variables["params"]["head"]
    np.testing.assert_allclose(output, features @ head["kernel"] + head["bias"], rtol=1e-6)
    np.testing.assert_array_equal(network.apply(variables, obs), output)


@pytest.fixture
def gpu(monkeypatch):
    def install(capability):
        device = SimpleNamespace(
            platform="gpu", device_kind="NVIDIA GeForce", compute_capability=capability
        )
        monkeypatch.setattr(jax, "local_devices", lambda: [device])

    return install


def test_attention_runs_on_xla_off_nvidia():
    implementation, _ = get_attention_implementation(64, 16)
    assert implementation == "xla"


@pytest.mark.parametrize(
    "capability, head_dim, lengths, expected",
    [
        ("8.6", 64, (16, 16), "cudnn"),
        ("8.6", 64, (15, 15), "xla"),
        ("8.6", 12, (16, 16), "xla"),
        ("8.6", 192, (16, 16), "xla"),
        ("9.0", 192, (16, 16), "cudnn"),
        ("7.5", 64, (16, 16), "xla"),
    ],
)
def test_attention_picks_cudnn_only_where_it_runs(gpu, capability, head_dim, lengths, expected):
    gpu(capability)
    implementation, _ = get_attention_implementation(head_dim, *lengths)
    assert implementation == expected


def linear_inputs(decay, delta_rule, batch=2, length=37, heads=3, key_dim=8, value_dim=5):
    keys = jax.random.split(jax.random.key(7), 5)
    key = jax.random.normal(keys[1], (batch, length, heads, key_dim))
    return LinearAttentionInputs(
        query=jax.random.normal(keys[0], (batch, length, heads, key_dim)),
        key=key / jnp.linalg.norm(key, axis=-1, keepdims=True),
        value=jax.random.normal(keys[2], (batch, length, heads, value_dim)),
        log_decay=-2.0 * jax.random.uniform(keys[3], (batch, length, heads)) if decay else None,
        beta=jax.random.uniform(keys[4], (batch, length, heads)) if delta_rule else None,
    )


def per_token(inputs, done, state, delta_rule):
    query, key, value = (
        np.asarray(x, np.float64) for x in (inputs.query, inputs.key, inputs.value)
    )
    batch, length, heads, _ = value.shape
    log_decay = np.zeros((batch, length, heads))
    if inputs.log_decay is not None:
        log_decay = np.asarray(inputs.log_decay, np.float64)
    beta = np.ones((batch, length, heads))
    if inputs.beta is not None:
        beta = np.asarray(inputs.beta, np.float64)
    state = np.asarray(state, np.float64)
    outputs = np.zeros_like(value)
    for step in range(length):
        state = np.where(np.asarray(done)[:, step, None, None, None], 0.0, state)
        state = state * np.exp(log_decay[:, step])[..., None, None]
        written = value[:, step]
        if delta_rule:
            written = written - np.einsum("bhk,bhkv->bhv", key[:, step], state)
        state = state + np.einsum(
            "bhk,bhv->bhkv", key[:, step], written * beta[:, step, :, None]
        )
        outputs[:, step] = np.einsum("bhk,bhkv->bhv", query[:, step], state)
    return outputs, state


@pytest.mark.parametrize("delta_rule", [False, True], ids=["linear", "delta"])
@pytest.mark.parametrize("decay", [False, True], ids=["constant", "decayed"])
@pytest.mark.parametrize("chunk_size", [0, 1, 4, 16, 64], ids=lambda size: f"chunk{size}")
def test_linear_attention_matches_a_per_token_recurrence(decay, delta_rule, chunk_size):
    inputs = linear_inputs(decay, delta_rule)
    done = jax.random.uniform(jax.random.key(8), inputs.value.shape[:2]) < 0.1
    state = jax.random.normal(jax.random.key(9), (2, 3, 8, 5))
    expected, expected_state = per_token(inputs, done, state, delta_rule)
    if chunk_size:
        outputs, final = chunkwise(inputs, done, state, chunk_size)
    else:
        outputs, final = recurrent(inputs, done, state)
    np.testing.assert_allclose(outputs, expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(final, expected_state, rtol=1e-4, atol=1e-4)


def test_a_gated_delta_net_forgets_everything_before_an_episode_start_inside_a_chunk():
    torso, params = build("gated_delta_net")
    x = sequence()
    carry = torso.initialize_carry(KEY, (BATCH, zoo.WIDTH))
    _, outputs = torso.apply(params, carry, x, starts(5))
    _, fresh = torso.apply(params, carry, x[:, 5:], starts()[:, : LENGTH - 5])
    np.testing.assert_allclose(outputs[:, 5:], fresh, rtol=1e-4, atol=1e-5)


def reference_qwen3_5():
    torch = pytest.importorskip("torch")
    modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    configuration = pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5")
    torch.manual_seed(0)
    config = configuration.Qwen3_5TextConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [1, 1, 0],
            "mrope_interleaved": True,
        },
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    model = modeling.Qwen3_5ForCausalLM(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * (0.5 if parameter.ndim == 1 else 0.2))
    state = {name: jnp.asarray(value.numpy()) for name, value in model.state_dict().items()}
    return torch, config, model, state


def qwen3_5_from(config, context_length):
    assert config.rms_norm_eps == 1e-6
    delta = LinearAttention(
        cell=GatedDeltaNet(
            features=config.hidden_size,
            num_key_heads=config.linear_num_key_heads,
            num_value_heads=config.linear_num_value_heads,
            key_dim=config.linear_key_head_dim,
            value_dim=config.linear_value_head_dim,
            kernel_size=config.linear_conv_kernel_dim,
            epsilon=config.rms_norm_eps,
        ),
        chunk_size=4,
    )
    attention = zoo.qwen3_5_attention(
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        int(config.head_dim * config.rope_parameters["partial_rotary_factor"]),
        config.rope_parameters["rope_theta"],
        context_length,
    )
    blocks = tuple(attention if kind == "full_attention" else delta for kind in config.layer_types)
    return llama(
        blocks,
        num_layers=len(blocks),
        features=config.hidden_size,
        hidden_dim=config.intermediate_size,
    )


def qwen3_5_logits(stack, params, state, carry, tokens, done):
    embedded = state["model.embed_tokens.weight"][tokens]
    carry, hidden = stack.apply(params, carry, embedded, done)
    return carry, hidden @ state["lm_head.weight"].T


def test_qwen3_5_matches_transformers_over_a_sequence_and_token_by_token():
    torch, config, model, state = reference_qwen3_5()
    batch, length, prefill = 2, 11, 6
    tokens = np.random.default_rng(0).integers(0, config.vocab_size, (batch, length))
    with torch.no_grad():
        whole = model(torch.as_tensor(tokens)).logits.numpy()
        output = model(torch.as_tensor(tokens[:, :prefill]), use_cache=True)
        stepped = [output.logits.numpy()]
        for step in range(prefill, length):
            output = model(
                torch.as_tensor(tokens[:, step : step + 1]),
                past_key_values=output.past_key_values,
                use_cache=True,
            )
            stepped.append(output.logits.numpy())
    stepped = np.concatenate(stepped, axis=1)

    stack = qwen3_5_from(config, length)
    params = load_qwen3_5(stack, state, prefix="model")
    done = jnp.zeros((batch, length), bool).at[:, 0].set(True)
    carry = stack.initialize_carry(KEY, (batch, config.hidden_size))
    _, logits = qwen3_5_logits(stack, params, state, carry, jnp.asarray(tokens), done)
    np.testing.assert_allclose(logits, whole, rtol=1e-4, atol=1e-4)

    carry, first = qwen3_5_logits(
        stack, params, state, carry, jnp.asarray(tokens[:, :prefill]), done[:, :prefill]
    )
    pieces = [first]
    for step in range(prefill, length):
        carry, piece = qwen3_5_logits(
            stack,
            params,
            state,
            carry,
            jnp.asarray(tokens[:, step : step + 1]),
            done[:, step : step + 1],
        )
        pieces.append(piece)
    decoded = jnp.concatenate(pieces, axis=1)
    np.testing.assert_allclose(decoded, stepped, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(decoded, logits, rtol=1e-4, atol=1e-4)
