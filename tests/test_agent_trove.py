from types import SimpleNamespace

import jax
import numpy as np
import pytest

from boonta.datasets.agent_trove import AgentTrove, tokenize, windows


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[ord(char) for char in text])

    def token_to_id(self, token):
        return 0


ROW = {"conversations": [
    {"from": "human", "value": "question"},
    {"from": "gpt", "value": "answer"},
]}


def test_chatml_masks_only_assistant_body_and_end_marker():
    tokens, weights = tokenize(ROW, Tokenizer())
    text = "".join(map(chr, tokens))
    assert text == "<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\nanswer<|im_end|>\n"
    boundary = text.index("answer")
    np.testing.assert_array_equal(weights[:boundary], 0)
    np.testing.assert_array_equal(weights[boundary:], 1)
    _, all_weights = tokenize(ROW, Tokenizer(), assistant_only=False)
    assert all_weights.all()


def test_windows_preserve_shifted_targets_tail_and_padding():
    tokens, _ = tokenize(ROW, Tokenizer(), assistant_only=False)
    chunks = list(windows(ROW, Tokenizer(), 128, 0, assistant_only=False))
    assert len(chunks) == 1
    targets = np.concatenate([
        chunk.second.action[chunk.aux["weight"].astype(bool)] for chunk in chunks
    ])
    np.testing.assert_array_equal(targets, tokens[1:])
    for chunk in chunks:
        valid = int(chunk.aux["weight"].sum())
        np.testing.assert_array_equal(chunk.first.obs[1:valid], chunk.second.action[:valid - 1])
        assert chunk.first.done[0] and not chunk.first.done[1:].any()
    assert (chunks[-1].second.action[int(chunks[-1].aux["weight"].sum()):] == 0).all()


def test_pool_samples_under_jit_and_rejects_empty_supervision():
    dataset = AgentTrove(rows=[ROW], tokenizer=Tokenizer(), length=128, pool_size=4)
    state = dataset.init()
    batch = jax.jit(lambda key: dataset.sample(state, key, (2, 128)))(jax.random.key(0))
    assert batch.first.obs.shape == (2, 128)
    assert batch.aux["weight"].sum() > 0
    with pytest.raises(ValueError, match="expects"):
        dataset.sample(state, jax.random.key(0), (2, 4))
    with pytest.raises(ValueError, match="no trainable"):
        AgentTrove(rows=[{"messages": [{"role": "user", "content": "hi"}]}],
                   tokenizer=Tokenizer(), pool_size=1).init()


def test_role_content_schema_and_unknown_roles():
    row = {"messages": [{"role": "assistant", "content": "ok"}]}
    assert tokenize(row, Tokenizer())[1].any()
    with pytest.raises(ValueError, match="Unsupported"):
        tokenize({"messages": [{"role": "bad", "content": "x"}]}, Tokenizer())


@pytest.mark.parametrize("empty", [
    {"messages": None, "conversations": None},
    {"conversations": []}, {"messages": []},
])
def test_empty_trajectories_are_skipped_without_losing_valid_rows(empty):
    assert list(windows(empty, Tokenizer(), 8, 0)) == []
    dataset = AgentTrove(rows=[empty, ROW, empty], tokenizer=Tokenizer(), length=128, pool_size=4)
    assert dataset.init().transitions.aux["weight"].sum() > 0
    with pytest.raises(ValueError, match="no trainable"):
        AgentTrove(rows=[empty], tokenizer=Tokenizer(), pool_size=1).init()


def test_nullable_union_fields_fall_back_to_sharegpt_values():
    row = {"messages": None, "conversations": [
        {"role": None, "from": "gpt", "content": None, "value": "answer"},
    ]}
    expected = {"messages": [{"role": "assistant", "content": "answer"}]}
    for actual, wanted in zip(tokenize(row, Tokenizer()), tokenize(expected, Tokenizer())):
        np.testing.assert_array_equal(actual, wanted)
    with pytest.raises(ValueError, match="Unsupported AgentTrove columns"):
        tokenize({"unexpected_schema": []}, Tokenizer())


def test_pool_refresh_uses_epoch_key_and_preserves_sharding():
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    class Rows(list):
        def set_epoch(self, seed):
            self.epoch = seed

    rows = Rows([ROW])
    dataset = AgentTrove(rows=rows, tokenizer=Tokenizer(), length=128, pool_size=4)
    state = dataset.init()
    mesh = Mesh(np.asarray(jax.devices()), ("data",))
    sharding = jax.tree.map(lambda x: NamedSharding(mesh, P("data") if np.ndim(x) else P()), state)
    key = jax.random.key(7)
    refreshed = dataset.update(state, key, sharding)
    expected = int(np.asarray(jax.random.bits(key, (), dtype=np.uint32)))
    assert rows.epoch == expected
    assert refreshed.transitions.first.obs.sharding.spec == P("data")
    np.testing.assert_array_equal(refreshed.transitions.first.obs, state.transitions.first.obs)


def test_long_trajectories_are_excluded_whole_and_never_truncated():
    tokens, _ = tokenize(ROW, Tokenizer())
    assert list(windows(ROW, Tokenizer(), len(tokens) - 1, 0)) == []
    assert len(list(windows(ROW, Tokenizer(), len(tokens), 0))) == 1
    with pytest.raises(ValueError, match="no trainable complete trajectories"):
        AgentTrove(rows=[ROW], tokenizer=Tokenizer(), length=8, pool_size=1).init()


@pytest.mark.parametrize("row", [
    {"messages": [{"role": "assistant", "content": "orphan"}]},
    {"messages": [{"role": "user", "content": " "}, {"role": "assistant", "content": "orphan"}]},
    {"messages": [{"role": "system", "content": "rules"}, {"role": "assistant", "content": "orphan"}]},
    {"messages": [{"role": "user", "content": "task"}, {"role": "assistant", "content": "act"}, {"role": "tool", "content": "unfinished"}]},
])
def test_unconditioned_or_environment_ended_records_are_excluded_for_both_modes(row):
    for mode in (True, False):
        assert list(windows(row, Tokenizer(), 256, 0, assistant_only=mode)) == []


def test_clm_blm_keep_identical_inputs_targets_and_full_task_tool_context():
    row = {"messages": [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "original task"},
        {"role": "assistant", "content": '{"commands":[{"keystrokes":"ls"}]}'},
        {"role": "tool", "content": "tool observation"},
        {"role": "assistant", "content": '{"commands":[],"task_complete":true}'},
    ]}
    blm, = windows(row, Tokenizer(), 512, 0, assistant_only=True)
    clm, = windows(row, Tokenizer(), 512, 0, assistant_only=False)
    for left, right in zip(jax.tree.leaves(blm.first), jax.tree.leaves(clm.first)):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(blm.second.action, clm.second.action)
    tokens, weights = tokenize(row, Tokenizer())
    np.testing.assert_array_equal(blm.first.obs[:len(tokens) - 1], tokens[:-1])
    np.testing.assert_array_equal(blm.aux["weight"][:len(tokens) - 1], weights[1:])
    assert blm.aux["weight"].sum() < clm.aux["weight"].sum()
    assert not blm.aux["weight"][len(tokens) - 1:].any()
    assert not clm.aux["weight"][len(tokens) - 1:].any()


def test_pool_selection_is_identical_between_objectives():
    rows = [ROW, {"messages": [{"role": "user", "content": "no action"}]},
            {"messages": [{"role": "user", "content": "other task"}, {"role": "assistant", "content": "other action"}]}]
    states = [AgentTrove(rows=rows, tokenizer=Tokenizer(), length=128, pool_size=4,
                        assistant_only=mode).init() for mode in (True, False)]
    np.testing.assert_array_equal(states[0].transitions.first.obs, states[1].transitions.first.obs)
    np.testing.assert_array_equal(states[0].transitions.second.action, states[1].transitions.second.action)
