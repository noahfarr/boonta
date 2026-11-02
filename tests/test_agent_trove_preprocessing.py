import json
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from boonta.datasets.agent_trove import AgentTrove, CachedRows, windows
from boonta.networks import Qwen3
from hangar.scripts.preprocess_agent_trove import Accounting, record


@pytest.fixture
def cache(tmp_path):
    tokenizer = Tokenizer(WordLevel({'[UNK]': 0, '<|endoftext|>': 1, '<|im_start|>': 2, '<|im_end|>': 3}, unk_token='[UNK]'))
    tokenizer.add_special_tokens(['<|endoftext|>', '<|im_start|>', '<|im_end|>'])
    path = tmp_path / 'tokenizer.json'
    tokenizer.save(str(path))
    row = {'messages': [{'role': 'user', 'content': 'task'}, {'role': 'assistant', 'content': 'action'}]}
    raw = tmp_path / 'rows.jsonl'
    raw.write_text(json.dumps(row) + '\n')
    output = tmp_path / 'cache'
    subprocess.run([sys.executable, '-m', 'hangar.scripts.preprocess_agent_trove', '--output', str(output), '--tokenizer', str(path), '--revision', 'fixture', '--expected-rows', '1', '--rows-jsonl', str(raw)], check=True)
    return output, tokenizer, row


def test_cache_replays_exact_shifted_sequences_and_paired_jit_sampling(cache):
    path, tokenizer, row = cache
    split = json.loads((path / 'records.jsonl').read_text())['split']
    states = []
    for objective in (True, False):
        dataset = AgentTrove(cache_dir=path, split=split, tokenizer=tokenizer, length=32, pool_size=2, assistant_only=objective)
        state = dataset.init()
        reference, = windows(row, tokenizer, 32, dataset.pad_id, objective)
        np.testing.assert_array_equal(state.transitions.first.obs[0], reference.first.obs)
        np.testing.assert_array_equal(state.transitions.second.action[0], reference.second.action)
        np.testing.assert_array_equal(state.transitions.aux['weight'][0], reference.aux['weight'])
        batch = jax.jit(lambda key: dataset.sample(state, key, (2, 32)))(jax.random.key(7))
        states.append(batch)
    np.testing.assert_array_equal(states[0].first.obs, states[1].first.obs)
    np.testing.assert_array_equal(states[0].second.action, states[1].second.action)
    assert states[0].aux['weight'].sum() < states[1].aux['weight'].sum()
    wrong = Tokenizer(WordLevel({'[UNK]': 0, '<|endoftext|>': 1, '<|im_start|>': 2, '<|im_end|>': 3, 'extra': 4}, unk_token='[UNK]'))
    with pytest.raises(ValueError, match='Tokenizer'):
        AgentTrove(cache_dir=path, split=split, tokenizer=wrong)
    manifest = json.loads((path / 'manifest.json').read_text())
    manifest['complete'] = False
    (path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='complete'):
        CachedRows(path, split)


def test_accounting_conserves_tokens_and_padding(cache):
    _, tokenizer, row = cache
    tokens, mask, item = record(row, tokenizer)
    n = len(tokens)
    a = Accounting((n-1, n, n+10))
    a.add(item)
    report = a.summary()['all']
    assert report['length_histogram'] == {str(n): 1}
    for context, values in report['contexts'].items():
        assert values['retained_serialized_tokens'] + values['discarded_serialized_tokens'] == n
        assert values['padding_positions'] + values['clm_targets'] == values['allocated_positions']
        assert values['eligible_trajectories'] == int(int(context) >= n)
    kept = report['contexts'][str(n)]
    assert kept['blm_targets'] == int(mask[1:].sum())
    assert kept['clm_targets'] == n-1
    assert kept['padding_positions'] == 1
    assert record(row, tokenizer)[2]['split'] == item['split']


def test_qwen3_causal_and_batch_isolation():
    model = Qwen3(features=8, num_layers=1, num_heads=2, num_groups=1, head_dim=4,
                  context_length=8, hidden_dim=16, max_wavelength=1_000_000.)
    x = jnp.arange(64, dtype=jnp.float32).reshape(2, 4, 8) / 64
    done = jnp.zeros((2, 4), bool).at[:, 0].set(True)
    params = model.init(jax.random.key(0), None, x, done)
    apply = jax.jit(lambda x: model.apply(params, None, x, done)[1])
    original = apply(x)
    changed = apply(x.at[1].set(-10).at[0, -1].set(20))
    np.testing.assert_allclose(original[0, :-1], changed[0, :-1], atol=1e-6)


def test_uniform_pool_index_visits_all_metadata_and_is_deterministic(cache):
    path, _, _ = cache
    item = json.loads((path / 'records.jsonl').read_text())
    (path / 'records.jsonl').write_text(''.join(json.dumps({**item, 'index': i}) + '\n' for i in range(100)))
    manifest = json.loads((path / 'manifest.json').read_text())
    manifest['rows'] = 100
    (path / 'manifest.json').write_text(json.dumps(manifest))
    rows = CachedRows(path, item['split'])
    selection = lambda seed: [r['index'] for r in rows.sample_rows(32, 3, seed)]
    assert selection(7) == selection(7)
    assert len(set(selection(7))) == 3
    assert max(selection(7)) > 5  # not the prefix of a two-row shuffle buffer
    assert selection(8) != selection(7)
    assert list(rows.sample_rows(1, 3, 7)) == []


def test_generic_error_is_not_inferred_task_failure():
    from hangar.scripts.audit_agent_trove import outcome
    assert outcome({'result': 'error'}) == 'unknown'
    assert outcome({'result': 'DaytonaError'}) == 'infrastructure_error'


def test_resume_trims_uncommitted_tail_and_matches_original(cache):
    path, _, _ = cache
    before = {name: (path / name).read_bytes() for name in ('tokens.bin', 'mask.bin', 'records.jsonl', 'statistics.json')}
    manifest = json.loads((path / 'manifest.json').read_text())
    manifest['complete'] = False
    (path / 'manifest.json').write_text(json.dumps(manifest))
    for name in ('tokens.bin', 'mask.bin', 'records.jsonl'):
        with (path / name).open('ab') as f:
            f.write(b'uncommitted')
    subprocess.run([sys.executable, '-m', 'hangar.scripts.preprocess_agent_trove', '--resume', '--output', str(path), '--tokenizer', str(path.parent / 'tokenizer.json'), '--revision', 'fixture', '--expected-rows', '1', '--rows-jsonl', str(path.parent / 'rows.jsonl'), '--executor', 'processes', '--workers', '2'], check=True)
    for name, original in before.items():
        assert (path / name).read_bytes() == original


def test_cached_pool_refresh_and_jit_preserve_named_sharding(cache):
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    path, tokenizer, _ = cache
    split = json.loads((path / 'records.jsonl').read_text())['split']
    batch_size = max(2, jax.device_count())
    dataset = AgentTrove(cache_dir=path, split=split, tokenizer=tokenizer, length=32,
                         pool_size=batch_size, num_devices=jax.device_count())
    state = dataset.init()
    mesh = Mesh(np.asarray(jax.devices()), ('data',))
    data = NamedSharding(mesh, P('data'))
    replicated = NamedSharding(mesh, P())
    shardings = jax.tree.map(lambda x: data if np.ndim(x) else replicated, state)
    refreshed = dataset.update(state, jax.random.key(3), shardings)
    sample = jax.jit(lambda state, key: dataset.sample(state, key, (batch_size, 32)),
                     in_shardings=(shardings, replicated), out_shardings=data)
    result = sample(refreshed, jax.random.key(4))
    assert result.first.obs.sharding.spec == P('data')
    assert result.aux['weight'].sharding.spec == P('data')
    np.testing.assert_array_equal(result.first.obs[0], result.first.obs[1])


def test_accounting_matches_existing_readiness_and_preserves_outcomes():
    from types import SimpleNamespace
    from hangar.scripts.check_agent_trove_readiness import measure
    class Characters:
        def encode(self, text, add_special_tokens=False):
            return SimpleNamespace(ids=list(map(ord, text)), offsets=[(i, i+1) for i in range(len(text))])
    tokenizer = Characters()
    row = {'messages': [{'role': 'user', 'content': 'task'},
                        {'role': 'assistant', 'content': '{"commands":[{"keystrokes":"ls"}]}'},
                        {'role': 'tool', 'content': 'observed output'},
                        {'role': 'assistant', 'content': 'done'}], 'original_source': 'fixture'}
    a = Accounting((64, 512))
    reference = measure(row, tokenizer, (64, 512))
    for label, metadata in [('success', {'success': True}), ('failure', {'success': False}),
                            ('timeout', {'result': 'AgentTimeoutError'}),
                            ('infrastructure_error', {'result': 'DaytonaError'}), ('unknown', {})]:
        _, _, item = record({**row, **metadata}, tokenizer)
        assert item['outcome'] == label
        assert item['blm_targets'] == reference['all_blm_targets']
        assert item['command_tokens'] == reference['all_action_tokens']
        a.add(item)
    summary = a.summary()
    for c in (64, 512):
        current = summary['all']['contexts'][str(c)]
        old = reference['budgets'][str(c)]
        assert current['eligible_trajectories'] == 5 * old['retained']
        assert current['blm_targets'] == 5 * old['blm_targets']
        assert current['clm_targets'] == 5 * old['clm_targets']
        assert current['command_tokens'] == 5 * old['action_tokens']
    assert all(summary['outcome/' + label]['totals']['rows'] == 1 for label in ('success', 'failure', 'timeout', 'infrastructure_error', 'unknown'))


def test_process_and_thread_preprocessing_are_byte_identical(cache):
    path, _, row = cache
    raw = path.parent / 'multiple.jsonl'
    raw.write_text(''.join(json.dumps({**row, 'original_source': str(i)}) + '\n' for i in range(3)))
    directories = [path.parent / name for name in ('threads', 'processes')]
    for executor, output in zip(('threads', 'processes'), directories):
        subprocess.run([sys.executable, '-m', 'hangar.scripts.preprocess_agent_trove', '--output', str(output), '--tokenizer', str(path.parent / 'tokenizer.json'), '--revision', 'fixture', '--expected-rows', '3', '--rows-jsonl', str(raw), '--executor', executor, '--workers', '2'], check=True)
    for name in ('tokens.bin', 'mask.bin', 'records.jsonl', 'statistics.json', 'report.md'):
        assert (directories[0] / name).read_bytes() == (directories[1] / name).read_bytes()


def test_tokenizer_cannot_silently_truncate_or_pad_messages(cache):
    _, tokenizer, _ = cache
    tokenizer.enable_truncation(max_length=2)
    with pytest.raises(ValueError, match='truncation and padding'):
        AgentTrove(rows=[], tokenizer=tokenizer)
    tokenizer.no_truncation()
    tokenizer.enable_padding(length=128)
    with pytest.raises(ValueError, match='truncation and padding'):
        AgentTrove(rows=[], tokenizer=tokenizer)


def test_completed_cache_readiness_checks_statistics_and_sharded_pair(cache):
    from hangar.scripts.check_agent_trove_readiness import check_cache
    path, tokenizer, _ = cache
    result = check_cache(path, tokenizer, (4096,))
    assert result['passed']
    assert result['rows_verified'] == 1
    assert result['contexts']['4096']['sampled'] == 2 * jax.device_count()
