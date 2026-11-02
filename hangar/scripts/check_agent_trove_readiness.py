"""Offline data-readiness checks against a cached AgentTrove audit sample."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from tokenizers import Tokenizer

from boonta.datasets.agent_trove import AgentTrove, complete_trajectory, normalize_messages, tokenize, windows
from hangar.scripts.audit_agent_trove import commands, outcome


def measure(row, tokenizer, contexts):
    messages = normalize_messages(row)
    tokens, mask = tokenize(row, tokenizer)
    non_system = [m for m in messages if m["role"] != "system"]
    has_task = bool(non_system and non_system[0]["role"] == "user" and non_system[0]["content"].strip())
    # Recognized final command payload tokens with their exact positions in the
    # same serialization as the adapter. Invalid action syntax is not guessed.
    position = 0
    action_indices = []
    for message in messages:
        role, content = message["role"], message["content"]
        position += len(tokenizer.encode(f"<|im_start|>{role}\n", add_special_tokens=False).ids)
        body = tokenizer.encode(content + "<|im_end|>\n", add_special_tokens=False)
        if role == "assistant":
            for start, end, values in commands(content):
                if values:
                    action_indices.extend(position + i for i, (a, b) in enumerate(body.offsets) if a < end and b > start)
        position += len(body.ids)
    assert position == len(tokens)
    budgets = {}
    for context in contexts:
        retained = complete_trajectory(row, tokens, mask, context)
        if retained:
            blm, = windows(row, tokenizer, context, 0, assistant_only=True)
            clm, = windows(row, tokenizer, context, 0, assistant_only=False)
            for left, right in zip(jax.tree.leaves(blm.first), jax.tree.leaves(clm.first), strict=True):
                np.testing.assert_array_equal(left, right)
            np.testing.assert_array_equal(blm.second.action, clm.second.action)
            np.testing.assert_array_equal(blm.first.obs[:len(tokens) - 1], tokens[:-1])
            np.testing.assert_array_equal(blm.second.action[:len(tokens) - 1], tokens[1:])
            assert blm.first.done[0] and not blm.first.done[1:].any()
            assert not blm.aux["weight"][len(tokens) - 1:].any()
            assert not clm.aux["weight"][len(tokens) - 1:].any()
            assert int(blm.aux["weight"].sum()) == int(mask[1:].sum())
        budgets[str(context)] = {
            "retained": int(retained),
            "blm_targets": int(mask[1:].sum()) if retained else 0,
            "clm_targets": len(tokens) - 1 if retained else 0,
            "action_tokens": len(action_indices) if retained else 0,
            "conditioned_action_tokens": len(action_indices) if retained and has_task else 0,
            "observation_body_tokens": sum(len(tokenizer.encode(m["content"] + "<|im_end|>\n", add_special_tokens=False).ids)
                                           for m in messages if m["role"] in {"user", "tool"}) if retained else 0,
            "allocated_positions": context if retained else 0,
            # Legacy chunks only have the original task prefix in the first
            # window. Later resets cannot satisfy the chosen history contract.
            "legacy_conditioned_action_tokens": sum(0 < t <= context for t in action_indices) if has_task else 0,
        }
    return {"serialized_tokens": len(tokens), "all_blm_targets": int(mask[1:].sum()),
            "all_action_tokens": len(action_indices), "budgets": budgets,
            "source": row.get("original_source") or "unknown", "outcome": outcome(row),
            "has_task_prefix": has_task}


def summarize(records, contexts):
    nonempty = [r["serialized_tokens"] for r in records if r["serialized_tokens"]]
    total_actions = sum(r["all_action_tokens"] for r in records)
    total_blm = sum(r["all_blm_targets"] for r in records)
    budgets = {}
    for context in contexts:
        values = [r["budgets"][str(context)] for r in records]
        total = Counter()
        for value in values:
            total.update(value)
        total["q_action_context"] = total["conditioned_action_tokens"] / total["action_tokens"] if total["action_tokens"] else None
        total["legacy_q_action_context"] = total["legacy_conditioned_action_tokens"] / total_actions if total_actions else None
        total["action_budget_retention"] = total["action_tokens"] / total_actions if total_actions else None
        total["blm_budget_retention"] = total["blm_targets"] / total_blm if total_blm else None
        total["blm_supervision_density"] = total["blm_targets"] / total["allocated_positions"] if total["allocated_positions"] else None
        budgets[str(context)] = dict(total)
    return {"rows": len(records), "nonempty_rows": len(nonempty),
            "all_blm_targets": total_blm, "all_action_tokens": total_actions,
            "complete_record_length_quantiles": {str(q): float(np.quantile(nonempty, q)) for q in [0, .1, .25, .5, .75, .9, .95, .99, 1]} if nonempty else {},
            "contexts": budgets}


def check_cache(path, tokenizer, contexts):
    """Verify an entire completed artifact plus small sharded sampling checks."""
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    report = json.loads((path / "statistics.json").read_text())
    manifest = report["manifest"]
    assert manifest["complete"]
    if any(c > manifest["cache_max_length"] for c in contexts):
        raise ValueError("Requested context exceeds the preprocessed cache limit")
    validator = AgentTrove(cache_dir=path, tokenizer=tokenizer, length=min(contexts),
                           pool_size=2*jax.device_count(), num_devices=jax.device_count())
    totals = report["measurements"]["all"]
    rows = serialized = cached = 0
    cohorts = Counter()
    measured = {str(c): Counter() for c in contexts}
    with (path / "records.jsonl").open() as records:
        for line in records:
            item = json.loads(line)
            assert item["index"] == rows and item["offset"] == cached
            expected_split = "validation" if int(item["identity"][:16], 16) % 100 == 0 else "train"
            assert item["split"] == expected_split
            rows += 1
            serialized += item["length"]
            cached += item["length"] if item["cached"] else 0
            for c in contexts:
                if item["eligible"] and item["length"] <= c:
                    cohorts[(item["split"], c)] += 1
                    measured[str(c)].update(
                        eligible_trajectories=1, retained_serialized_tokens=item["length"],
                        blm_targets=item["blm_targets"], clm_targets=item["length"]-1,
                        command_tokens=item["command_tokens"], allocated_positions=c,
                        padding_positions=c-item["length"]+1,
                    )
    assert rows == manifest["rows"] == manifest["expected_rows"]
    assert serialized == manifest["serialized_tokens"] == totals["totals"]["serialized_tokens"]
    assert cached == manifest["cached_tokens"]
    assert sum(totals["length_histogram"].values()) == rows
    for c in contexts:
        expected = totals["contexts"][str(c)]
        for name, value in measured[str(c)].items():
            assert expected[name] == value, (c, name)
        assert expected["retained_serialized_tokens"] + expected["discarded_serialized_tokens"] == serialized
        assert expected["clm_targets"] + expected["padding_positions"] == expected["allocated_positions"]
        for prefix in ("source/", "outcome/", "split/"):
            groups = [g for name, g in report["measurements"].items() if name.startswith(prefix)]
            for name in ("eligible_trajectories", "retained_serialized_tokens", "blm_targets", "clm_targets", "command_tokens"):
                assert sum(g["contexts"][str(c)][name] for g in groups) == expected[name]
    mask = np.memmap(path / "mask.bin", mode="r", dtype="u1") if cached else np.array([], dtype="u1")
    # Every eligible record begins with an unweighted user/system header, so
    # dropping the initial prediction target does not change this mask sum.
    mask_targets = int(mask.sum(dtype=np.uint64))
    assert mask_targets == totals["contexts"][str(manifest["cache_max_length"])]["blm_targets"]
    del mask
    mesh = Mesh(np.asarray(jax.devices()), ("data",))
    sharding = NamedSharding(mesh, P("data"))
    replicated = NamedSharding(mesh, P())
    batch_size = 2 * jax.device_count()
    def check_split(split):
        shared_rows = validator.rows if split == "train" else None
        checks = {}
        for c in contexts:
            if not cohorts[(split, c)]:
                checks[str(c)] = {"sampled": 0}
                continue
            batches = []
            for mode in (True, False):
                options = {"cache_dir": path} if shared_rows is None else {"rows": shared_rows}
                dataset = AgentTrove(tokenizer=tokenizer, length=c, pool_size=batch_size,
                                     num_devices=jax.device_count(), assistant_only=mode, split=split, **options)
                shared_rows = dataset.rows
                state = dataset.init()
                shardings = jax.tree.map(lambda x: sharding if np.ndim(x) else replicated, state)
                state = dataset.update(state, jax.random.key(7), shardings)
                sample = jax.jit(lambda state, key: dataset.sample(state, key, (batch_size, c)),
                                 in_shardings=(shardings, replicated), out_shardings=sharding)
                batch = sample(state, jax.random.key(8))
                assert batch.first.obs.sharding.spec == P("data")
                assert np.asarray(batch.first.done)[:, 0].all()
                assert not np.asarray(batch.first.done)[:, 1:].any()
                batches.append(batch)
            blm, clm = batches
            for left, right in zip(jax.tree.leaves(blm.first), jax.tree.leaves(clm.first), strict=True):
                np.testing.assert_array_equal(left, right)
            np.testing.assert_array_equal(blm.second.action, clm.second.action)
            valid = np.asarray(clm.aux["weight"], bool)
            assert not np.asarray(blm.aux["weight"])[~valid].any()
            for index in range(batch_size):
                size = int(valid[index].sum())
                np.testing.assert_array_equal(np.asarray(clm.first.obs)[index, 1:size],
                                              np.asarray(clm.second.action)[index, :size-1])
            checks[str(c)] = {"sampled": batch_size, "input_sha256": hashlib.sha256(np.asarray(blm.first.obs).tobytes()).hexdigest()}
        return checks

    checks = check_split("train")
    validation_checks = check_split("validation")
    result = {"rows_verified": rows, "serialized_tokens": serialized,
              "cached_tokens": cached, "cached_blm_targets_verified": mask_targets,
              "device_count": jax.device_count(), "contexts": checks,
              "validation_contexts": validation_checks, "split_assignment_verified": True, "passed": True}
    (path / "cache_readiness.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, help="Verify completed full-corpus preprocessing artifacts")
    parser.add_argument("--sample", type=Path, default=Path("outputs/agent_trove_audit/sample"))
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=None)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    if args.cache:
        print(json.dumps(check_cache(args.cache, tokenizer, args.contexts or [4096, 8192, 16384, 32768]), indent=2))
        return
    args.contexts = args.contexts or [1024, 2048, 4096, 8192, 16384, 32768]
    paths = sorted(args.sample.glob("rows_*.json"))
    if not paths:
        parser.error(f"No cached audit row blocks found in {args.sample}")
    records, seen, duplicate_rows, truncated = [], set(), [], []
    # Full-conversation exact deduplication within the sample, independent of
    # objective or context length. No outcome-based exclusions are applied.
    for path in paths:
        for item in json.loads(path.read_text())["rows"]:
            if item.get("truncated_cells"):
                truncated.append(item["row_idx"])
                continue
            row = item["row"]
            canonical = json.dumps(normalize_messages(row), sort_keys=True)
            if canonical != "[]" and canonical in seen:
                duplicate_rows.append(item["row_idx"])
                continue
            seen.add(canonical)
            record = measure(row, tokenizer, args.contexts)
            records.append({"index": item["row_idx"], **record})
    report = {
        "policy": "complete_trajectory",
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "context_contract": "first non-system turn is a nonempty original user task; preserve every preceding message, including the latest environment interaction and complete action prefix; whole record fits context",
        "completion_definition": "complete recorded interaction, not verified task success; require final non-system turn to be assistant",
        "action_definition": "recognized final commands-list payload tokens; all BLM assistant-body targets also retain their full serialized prefix",
        "checks": "asserted actual CLM/BLM input and target identity, exact original serialization, no internal resets, shifted targets and padding masks for every retained sample/context pair",
        "structural_checks_passed": True,
        "scope": "cached sample only; this is not a full-corpus budget or deduplication certification",
        "duplicates_removed": duplicate_rows, "truncated_rows_excluded": truncated,
        "summary": summarize(records, args.contexts),
        "by_source": {source: summarize([r for r in records if r["source"] == source], args.contexts) for source in sorted({r["source"] for r in records})},
        "by_outcome": {label: summarize([r for r in records if r["outcome"] == label], args.contexts) for label in sorted({r["outcome"] for r in records})},
    }
    (args.sample / "readiness_records.json").write_text(json.dumps(records, indent=2))
    (args.sample / "readiness.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
