import argparse
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from itertools import islice
from collections import Counter, defaultdict, deque
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from boonta.datasets.agent_trove import complete_trajectory, encode_messages, serialize, validate_tokenizer
from hangar.scripts.audit_agent_trove import commands, outcome

CONTEXTS = (4096, 8192, 16384, 32768)


def record(row, tokenizer):
    encodings = list(encode_messages(row, tokenizer))
    messages = [message for message, _, _ in encodings]
    tokens, mask = serialize(encodings)
    command_tokens = 0
    for message, _, encoding in encodings:
        if message['role'] == 'assistant':
            for start, end, values in commands(message['content']):
                if values:
                    command_tokens += sum(a < end and b > start for a, b in encoding.offsets)
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()
    identity = hashlib.sha256(canonical).hexdigest()
    split = 'validation' if int(identity[:16], 16) % 100 == 0 else 'train'
    metadata = {k: row[k] for k in ('success', 'is_success', 'reward', 'result', 'task_id', 'task', 'run_id') if row.get(k) is not None}
    return tokens, mask, dict(
        length=len(tokens), blm_targets=int(mask[1:].sum()), command_tokens=command_tokens,
        eligible=complete_trajectory(row, tokens, mask, max(len(tokens), 1)),
        source=row.get('original_source') or 'unknown', outcome=outcome(row),
        metadata=metadata, identity=identity, split=split,
    )


_WORKER_TOKENIZER = None


def initialize_worker(path):
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = Tokenizer.from_file(path)


def measure_worker(row):
    return record(row, _WORKER_TOKENIZER)


def restore(output, manifest, accounting):
    previous = json.loads((output / 'manifest.json').read_text())
    if previous.get('complete'):
        raise ValueError('Cannot resume an already complete cache')
    for field in ('format_version', 'revision', 'tokenizer_sha256', 'input_kind', 'input_sha256', 'expected_rows', 'cache_max_length', 'cache_written'):
        if previous.get(field) != manifest.get(field):
            raise ValueError(f'Resume mismatch: {field}')
    count = offset = serialized = 0
    with (output / 'records.jsonl').open('r+b') as source:
        valid_end = 0
        for line in source:
            if not line.endswith(b'\n'):
                break
            item = json.loads(line)
            if item['index'] != count or item['offset'] != offset:
                raise ValueError('Cache record index/offset mismatch')
            accounting.add(item)
            count += 1
            serialized += item['length']
            if item['cached']:
                offset += item['length']
            valid_end = source.tell()
        source.truncate(valid_end)
    if manifest['cache_written']:
        for name, size in (('tokens.bin', 4*offset), ('mask.bin', offset)):
            path = output / name
            if path.stat().st_size < size:
                raise ValueError(f'Truncated committed cache file: {name}')
            with path.open('r+b') as target:
                target.truncate(size)
    return count, offset, serialized


class Accounting:
    def __init__(self, contexts=CONTEXTS):
        self.contexts = contexts
        self.groups = defaultdict(lambda: dict(totals=Counter(), lengths=Counter(), metadata_presence=Counter(), raw_results=Counter(), retained_lengths={str(c): Counter() for c in contexts}, contexts={str(c): Counter() for c in contexts}))

    def add(self, item):
        for group in ('all', 'source/' + item['source'], 'outcome/' + item['outcome'], 'split/' + item['split']):
            g = self.groups[group]
            n = item['length']
            g['lengths'][n] += 1
            g['metadata_presence'].update(item['metadata'].keys())
            g['raw_results'][json.dumps(item['metadata'].get('result'), sort_keys=True, default=str)] += 1
            g['totals'].update(rows=1, serialized_tokens=n, blm_targets=item['blm_targets'], command_tokens=item['command_tokens'])
            for c in self.contexts:
                t = g['contexts'][str(c)]
                keep = item['eligible'] and n <= c
                if keep:
                    g['retained_lengths'][str(c)][n] += 1
                t.update(eligible_trajectories=int(keep), retained_serialized_tokens=n if keep else 0,
                         discarded_serialized_tokens=0 if keep else n,
                         blm_targets=item['blm_targets'] if keep else 0, clm_targets=max(n-1, 0) if keep else 0,
                         command_tokens=item['command_tokens'] if keep else 0,
                         allocated_positions=c if keep else 0, padding_positions=c-n+1 if keep else 0,
                         power_of_two_bucket_positions=max(128, 1 << max(n-2, 0).bit_length()) if keep else 0)

    def summary(self, parameters=200_000_000):
        result = {}
        for name, g in sorted(self.groups.items()):
            contexts = {}
            for c, counter in g['contexts'].items():
                t = dict(counter)
                ratio = lambda a, b: a / b if b else None
                t.update(retained_row_fraction=ratio(t['eligible_trajectories'], g['totals']['rows']),
                         retained_length_histogram={str(k): v for k, v in sorted(g['retained_lengths'][c].items())},
                         clm_density_allocated=ratio(t['clm_targets'], t['allocated_positions']),
                         token_retention=ratio(t['retained_serialized_tokens'], g['totals']['serialized_tokens']),
                         command_retention=ratio(t['command_tokens'], g['totals']['command_tokens']),
                         blm_density_valid=ratio(t['blm_targets'], t['clm_targets']),
                         blm_density_allocated=ratio(t['blm_targets'], t['allocated_positions']),
                         padding_fraction=ratio(t['padding_positions'], t['allocated_positions']),
                         token_cache_bytes=5*t['retained_serialized_tokens'],
                         dense_training_flops_estimate=6*parameters*t['allocated_positions'])
                contexts[c] = t
            result[name] = dict(totals=dict(g['totals']), metadata_presence=dict(g['metadata_presence']), raw_results=dict(g['raw_results']), length_histogram={str(k): v for k, v in sorted(g['lengths'].items())}, contexts=contexts)
        return result


def markdown(report):
    manifest, measurements = report['manifest'], report['measurements']
    lines = ['# AgentTrove accounting', '',
             f"Scope: {'complete input' if manifest['complete'] else 'PARTIAL input; not a full-corpus budget'}. Rows: {manifest['rows']:,}/{manifest['expected_rows']:,}.",
             f"Revision: `{manifest['revision']}`. Input: `{manifest['input_kind']}`.", '',
             '| Context | Eligible | Retained tokens | Discarded tokens | BLM targets | CLM targets | Commands | Token retention | Padding |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for c, t in measurements.get('all', {}).get('contexts', {}).items():
        percent = lambda value: f'{100*value:.2f}%' if value is not None else 'undefined'
        lines.append(f"| {c} | {t['eligible_trajectories']:,} | {t['retained_serialized_tokens']:,} | {t['discarded_serialized_tokens']:,} | {t['blm_targets']:,} | {t['clm_targets']:,} | {t['command_tokens']:,} | {percent(t['token_retention'])} | {percent(t['padding_fraction'])} |")
    lines += ['', '## Retained source composition', '', '| Source | Input rows | ' + ' | '.join(map(str, CONTEXTS)) + ' |', '| --- | ---: | ' + ' | '.join(['---:'] * len(CONTEXTS)) + ' |']
    for name, g in measurements.items():
        if name.startswith('source/'):
            lines.append('| ' + name[7:].replace('|', r'\|') + ' | ' + str(g['totals']['rows']) + ' | ' + ' | '.join(str(g['contexts'][str(c)]['eligible_trajectories']) for c in CONTEXTS) + ' |')
    lines += ['', 'Outcomes (all rows): ' + ', '.join(f"{name[8:]}={g['totals']['rows']}" for name, g in measurements.items() if name.startswith('outcome/')), '',
              'JSON contains exact length histograms, outcome/source/split budgets, metadata availability, command retention, supervision density, and padding estimates.',
              'Compute estimates use 6 × 200M × allocated positions per pass; they exclude quadratic attention and are not measured hardware throughput.',
              'Power-of-two bucket positions are a potential batching estimate, not implemented packing or measured training speed.', '']
    return '\n'.join(lines)


def file_digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--revision', required=True, help='Immutable dataset commit SHA')
    p.add_argument('--expected-rows', type=int, required=True)
    p.add_argument('--rows-jsonl', type=Path, help='Local untruncated raw rows; otherwise HF streaming')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--executor', choices=['threads', 'processes'], default='threads')
    p.add_argument('--resume', action='store_true', help='Continue an incomplete cache from committed row records')
    p.add_argument('--cache-max-length', type=int, choices=CONTEXTS, default=32768, help='Cache only eligible whole trajectories up to this length; statistics still cover every row')
    p.add_argument('--accounting-only', action='store_true', help='Measure all rows without writing token/mask binaries; not a training cache')
    p.add_argument('--limit', type=int, help='Explicitly partial smoke run')
    args = p.parse_args()
    if args.workers < 1 or args.cache_max_length < 1 or args.expected_rows < 1 or (args.limit is not None and args.limit < 1):
        p.error('expected-rows and limit must be positive')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    validate_tokenizer(tokenizer)
    if args.rows_jsonl:
        rows = (json.loads(line) for line in args.rows_jsonl.open())
    else:
        from datasets import load_dataset
        rows = load_dataset('open-thoughts/AgentTrove', revision=args.revision, split='train', streaming=True)
    accounting = Accounting()
    manifest = dict(format_version=1, dataset='open-thoughts/AgentTrove', revision=args.revision,
                    tokenizer_sha256=hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
                    tokenizer_runtime_sha256=hashlib.sha256(tokenizer.to_str().encode()).hexdigest(),
                    cache_written=not args.accounting_only, cache_max_length=args.cache_max_length,
                    expected_rows=args.expected_rows, complete=False, rows=0,
                    input_kind='local_jsonl' if args.rows_jsonl else 'huggingface_stream',
                    input_path=str(args.rows_jsonl) if args.rows_jsonl else None,
                    input_sha256=file_digest(args.rows_jsonl) if args.rows_jsonl else None,
                    split_policy='sha256(normalized conversation UTF-8 JSON) first 64 bits modulo 100 == 0 validation',
                    deduplication=False, window_policy='complete_trajectory',
                    compute_estimate='6*N*allocated positions per epoch, N=200M; excludes attention quadratic cost',
                    command_definition='heuristic final commands-list payload; not all tool-call formats')
    count = offset = serialized = 0
    if args.resume:
        count, offset, serialized = restore(args.output, manifest, accounting)
        print(f'Resuming after {count} committed rows', flush=True)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    with ExitStack() as stack:
        rf = stack.enter_context((args.output / 'records.jsonl').open('a'))
        if not args.accounting_only:
            tf = stack.enter_context((args.output / 'tokens.bin').open('ab'))
            mf = stack.enter_context((args.output / 'mask.bin').open('ab'))
        if args.executor == 'processes':
            executor = stack.enter_context(ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'), initializer=initialize_worker, initargs=(str(args.tokenizer),)))
            measure = measure_worker
        else:
            executor = stack.enter_context(ThreadPoolExecutor(max_workers=args.workers))
            measure = lambda row: record(row, tokenizer)
        source = iter(islice(rows, count, args.limit))
        def measured():
            pending = deque(executor.submit(measure, row) for row in islice(source, 32))
            while pending:
                yield pending.popleft().result()
                try:
                    row = next(source)
                except StopIteration:
                    continue
                pending.append(executor.submit(measure, row))
        for tokens, mask, item in measured():
            cached = not args.accounting_only and item['eligible'] and len(tokens) <= args.cache_max_length
            if cached:
                tokens.astype('<i4').tofile(tf)
                mask.astype('u1').tofile(mf)
            rf.write(json.dumps(dict(index=count, offset=offset, cached=cached, **item), ensure_ascii=False, default=str) + '\n')
            serialized += len(tokens)
            if cached:
                offset += len(tokens)
            count += 1
            accounting.add(item)
            if count % 1000 == 0:
                print(f'{count} rows, {serialized} tokens, {offset*5} cache bytes', flush=True)
    manifest.update(rows=count, serialized_tokens=serialized, cached_tokens=offset, complete=args.limit is None and count == args.expected_rows)
    report = dict(manifest=manifest, measurements=accounting.summary())
    (args.output / 'statistics.json').write_text(json.dumps(report, indent=2))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (args.output / 'report.md').write_text(markdown(report))
    if not manifest['complete']:
        print('PARTIAL: not a certified full-corpus budget', flush=True)
        if args.limit is None:
            raise ValueError('Corpus row count mismatch')


if __name__ == '__main__':
    main()
