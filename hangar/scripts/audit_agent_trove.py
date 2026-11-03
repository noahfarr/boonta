"""Reproducible, cached stratified sample audit of AgentTrove.

The viewer returns bounded row blocks rather than downloading whole Parquet
shards. Results describe the sample, not an exhaustive deduplication scan.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import time
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from tokenizers import Tokenizer

from boonta.datasets.agent_trove import tokenize


ALIASES = {"human": "user", "gpt": "assistant", "observation": "tool"}
OUTPUT = re.compile(
    r"(?:new terminal output|previous (?:command )?output|output from (?:the )?(?:last|previous)|"
    r"current terminal output|terminal state|exit code[: ]|process exited with|"
    r"tool (?:result|response)|observation:)", re.I,
)
ERROR = re.compile(r"(?:traceback \(most recent call last\)|command not found|permission denied|timed out)", re.I)
OUTPUT_START = re.compile(r"^\s*(?:new terminal output:|current terminal state:|current terminal screen:|previous response had (?:warnings|parsing errors):)", re.I)
DECODER = json.JSONDecoder()


def command_fields(content):
    thoughts = [(m.start(),m.end()) for m in re.finditer(r'<think>.*?</think>',content,re.S)]
    return [m for m in re.finditer(r'"commands"\s*:\s*',content)
            if not any(a <= m.start() < b for a,b in thoughts)]

def commands(content):
    spans = []
    for match in command_fields(content)[-1:]:
        try:
            value, end = DECODER.raw_decode(content, match.end())
            if isinstance(value,list):
                spans.append((match.end(),end,value))
        except ValueError:
            pass
    return spans[-1:]


def fetch(endpoint, params, path):
    if path.exists():
        return json.loads(path.read_text())
    url = "https://datasets-server.huggingface.co/" + endpoint + "?" + urlencode(params)
    for attempt in range(3):
        try:
            with urlopen(url, timeout=45) as response:
                payload = json.load(response)
            path.write_text(json.dumps(payload, ensure_ascii=False))
            return payload
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def conversation(row):
    messages = row.get("messages") or row.get("conversations") or []
    return [
        {"role": ALIASES.get(m.get("role") or m.get("from"), m.get("role") or m.get("from")),
         "content": m.get("content") if m.get("content") is not None else m.get("value")}
        for m in messages
    ]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def outcome(row):
    # Completion claimed by the assistant is deliberately not an outcome label.
    evidence = []
    for field in ("reward", "success", "is_success", "result"):
        value = row.get(field)
        if value is None:
            continue
        if isinstance(value, dict):
            value = value.get("success", value.get("reward"))
        label = str(value).strip().lower()
        if label in {"1", "1.0", "true", "success", "passed", "pass"}:
            evidence.append("success")
        elif label in {"0", "0.0", "false", "failure", "failed", "fail"}:
            evidence.append("failure")
        elif "timeout" in label:
            evidence.append("timeout")
        elif label.startswith("daytona") and "error" in label:
            evidence.append("infrastructure_error")
    return evidence[0] if len(set(evidence)) == 1 else "conflict" if evidence else "unknown"


def audit_row(index, row, tokenizer, length):
    """Measure raw roles/masks and the historical fixed-token-chunk baseline."""
    messages = conversation(row)
    roles = Counter(m["role"] for m in messages)
    flags = Counter()
    role_tokens = Counter()
    markers = Counter()
    command_tokens = calls = nonempty_calls = malformed_calls = unobserved_calls = 0
    assistant_output_candidates = []
    claimed_complete = False
    for i, message in enumerate(messages):
        role, content = message["role"], message["content"]
        if not isinstance(content, str):
            flags["non_string_content"] += 1
            continue
        encoding = tokenizer.encode(content + "<|im_end|>\n", add_special_tokens=False)
        role_tokens[role or "null"] += len(encoding.ids)
        if OUTPUT.search(content):
            markers[role or "null"] += 1
            if role == "assistant":
                assistant_output_candidates.append(i)
        if ERROR.search(content) and role in {"user", "tool"}:
            flags["environment_error_text"] += 1
        if role != "assistant":
            continue
        if re.search(r'"task_complete"\s*:\s*true', content, re.I):
            claimed_complete = True
        spans = commands(content)
        if command_fields(content) and not spans:
            malformed_calls += 1
        for start, end, values in spans:
            calls += 1
            if values:
                nonempty_calls += 1
                command_tokens += sum(a < end and b > start for a, b in encoding.offsets)
                following = messages[i + 1:i + 2]
                if not following or following[0]["role"] not in {"user", "tool"}:
                    unobserved_calls += 1
                if any(not isinstance(v, dict) or not isinstance(v.get("keystrokes", v.get("command")), str)
                       for v in values):
                    flags["malformed_command_entries"] += 1
    if not messages:
        flags["empty_trajectory"] += 1
    flags["assistant_output_marker_candidate"] = len(assistant_output_candidates)
    flags["malformed_command_json_candidate"] = malformed_calls
    flags["unobserved_command_turn"] = unobserved_calls
    if messages and messages[-1]["role"] in {"user", "tool"}:
        flags["ends_with_environment_turn"] = 1
    if messages and messages[-1]["role"] == "assistant" and not claimed_complete:
        flags["no_completion_claim_at_assistant_end"] = 1
    trainable = supervised = serialized = kept_windows = allocated = later_windows = inside_assistant = 0
    error = None
    try:
        tokens, weight = tokenize(row, tokenizer)
        serialized = len(tokens)
        supervised = int(weight[1:].sum())
        for start in range(0, len(tokens) - 1, length):
            if weight[start + 1:min(start + length + 1, len(tokens))].any():
                kept_windows += 1
                later_windows += int(start > 0)
                inside_assistant += int(start > 0 and weight[start] > 0)
        allocated = kept_windows * length
        trainable = int(supervised > 0)
    except (ValueError, TypeError, AttributeError) as exc:
        error = str(exc)
        flags["adapter_rejected"] = 1
    prompt = next((m["content"] for m in messages if m["role"] == "user"), None)
    return {
        "index": index, "source": row.get("original_source") or "unknown",
        "teacher": row.get("original_teacher") or row.get("model") or "unknown",
        "outcome": outcome(row), "result_raw": row.get("result"),
        "metadata_present": [k for k in ("reward", "result", "success", "is_success", "task_id", "task", "trial_name", "episode", "run_id") if row.get(k) is not None],
        "roles": dict(roles), "body_tokens_by_role": dict(role_tokens),
        "output_markers_by_role": dict(markers), "flags": dict(+flags),
        "assistant_output_candidate_turns": assistant_output_candidates,
        "serialized_tokens": serialized, "supervised_tokens": supervised,
        "command_value_tokens": command_tokens, "command_turns": calls,
        "nonempty_command_turns": nonempty_calls, "kept_windows": kept_windows,
        "allocated_positions": allocated, "trainable": trainable,
        "later_windows_without_task_prefix": later_windows,
        "window_starts_in_assistant_body": inside_assistant,
        "terminal_feedback_by_role": dict(Counter(m["role"] for m in messages if isinstance(m["content"], str) and OUTPUT_START.match(m["content"]))),
        "parser_error_feedback_turns": sum(m["role"] == "user" and (m["content"] or "").startswith("Previous response had parsing errors:") for m in messages),
        "has_task_binary": bool(row.get("task_binary")),
        "claims_task_complete": claimed_complete,
        "trajectory_hash": digest(messages) if messages else None,
        "prompt_hash": digest(prompt) if prompt else None,
        "adapter_error": error,
    }


def summarize(records):
    total = Counter(rows=len(records))
    for record in records:
        for field in ("serialized_tokens", "supervised_tokens", "command_value_tokens", "kept_windows", "allocated_positions", "trainable", "command_turns", "nonempty_command_turns", "later_windows_without_task_prefix", "window_starts_in_assistant_body", "parser_error_feedback_turns"):
            total[field] += record[field]
    total["supervised_fraction_of_serialized"] = total["supervised_tokens"] / max(total["serialized_tokens"], 1)
    total["supervised_fraction_of_allocated_positions"] = total["supervised_tokens"] / max(total["allocated_positions"], 1)
    total["command_fraction_of_supervised"] = total["command_value_tokens"] / max(total["supervised_tokens"], 1)
    def counts(field):
        result = Counter()
        for record in records:
            result.update(record[field])
        return dict(result)
    def duplicates(field):
        groups = defaultdict(list)
        for record in records:
            if record[field]:
                groups[record[field]].append(record["index"])
        groups = [indices for indices in groups.values() if len(indices) > 1]
        return {"extra_copies": sum(len(g) - 1 for g in groups), "groups": groups}
    return {"totals": dict(total), "roles": counts("roles"),
            "body_tokens_by_role": counts("body_tokens_by_role"),
            "output_markers_by_role": counts("output_markers_by_role"),
            "terminal_feedback_by_role": counts("terminal_feedback_by_role"),
            "rows_with_parser_error_feedback": sum(r["parser_error_feedback_turns"] > 0 for r in records),
            "rows_with_task_binary": sum(r["has_task_binary"] for r in records),
            "unobserved_command_rows_by_completion_claim": dict(Counter(str(r["claims_task_complete"]) for r in records if r["flags"].get("unobserved_command_turn"))),
            "flags_by_message_or_trajectory": counts("flags"),
            "flagged_trajectories": dict(Counter(flag for r in records for flag in r["flags"])),
            "outcomes": dict(Counter(r["outcome"] for r in records)),
            "raw_result_values": dict(Counter(str(r["result_raw"]) for r in records)),
            "metadata_presence": dict(Counter(k for r in records for k in r["metadata_present"])),
            "exact_trajectory_duplicates": duplicates("trajectory_hash"),
            "repeated_prompts": duplicates("prompt_hash")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/agent_trove_audit"))
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=48)
    parser.add_argument("--rows-per-block", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    params = {"dataset": "open-thoughts/AgentTrove", "config": "default", "split": "train"}
    size = fetch("size", params, args.output / "size.json")
    split = next(s for s in size["size"]["splits"] if s["config"] == "default" and s["split"] == "train")
    count = split["num_rows"]
    generator = random.Random(args.seed)
    offsets = [generator.randint(i * count // args.blocks, max(i * count // args.blocks, (i + 1) * count // args.blocks - args.rows_per_block)) for i in range(args.blocks)]
    payloads, failures = [], []
    def block(offset):
        return fetch("rows", {**params, "offset": offset, "length": args.rows_per_block},
                     args.output / f"rows_{offset}_{args.rows_per_block}.json")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(block, offset): offset for offset in offsets}
        for future in as_completed(futures):
            offset = futures[future]
            try:
                payload = future.result()
                payloads.append(payload)
                print(f"fetched {len(payload['rows'])} rows at {offset}", flush=True)
            except Exception as exc:
                failures.append({"offset": offset, "error": str(exc)})
                print(f"failed block {offset}: {exc}", flush=True)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    records, truncated = [], []
    for payload in payloads:
        for item in payload["rows"]:
            if item.get("truncated_cells"):
                truncated.append(item["row_idx"])
                continue
            records.append(audit_row(item["row_idx"], item["row"], tokenizer, args.length))
    records.sort(key=lambda r: r["index"])
    (args.output / "records.json").write_text(json.dumps(records, indent=2))
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "dataset_rows": count,
        "method": "random contiguous blocks within equally sized row-index strata; sample only",
        "window_policy_audited": "legacy fixed token chunks (historical baseline); run check_agent_trove_readiness for the current complete_trajectory policy",
        "seed": args.seed, "blocks_requested": args.blocks, "rows_per_block": args.rows_per_block,
        "offsets": offsets, "failed_blocks": failures, "excluded_truncated_rows": truncated,
        "dataset_revisions": sorted({p.get("dataset_git_revision", "not_reported") for p in payloads}),
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "context_length": args.length, "summary": summarize(records),
        "by_source": {source: summarize([r for r in records if r["source"] == source]) for source in sorted({r["source"] for r in records})},
        "by_outcome": {label: summarize([r for r in records if r["outcome"] == label]) for label in sorted({r["outcome"] for r in records})},
    }
    # Resample whole fetched blocks, not individual neighboring rows. These
    # intervals describe sampling variability, not unmeasured source bias.
    block_totals = []
    for offset in offsets:
        group = [r for r in records if offset <= r["index"] < offset + args.rows_per_block]
        if group:
            block_totals.append([sum(r[k] for r in group) for k in
                                 ("serialized_tokens", "supervised_tokens", "command_value_tokens", "allocated_positions", "trainable")])
    if block_totals:
        rng = np.random.default_rng(args.seed)
        values = np.asarray(block_totals, float)
        sampled = values[rng.integers(len(values), size=(2000, len(values)))].sum(axis=1)
        report["block_bootstrap_95pct"] = {
            "supervised_fraction_of_serialized": np.quantile(sampled[:, 1] / np.maximum(sampled[:, 0], 1), [.025, .975]).tolist(),
            "command_fraction_of_supervised": np.quantile(sampled[:, 2] / np.maximum(sampled[:, 1], 1), [.025, .975]).tolist(),
        }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"dataset_rows": count, "rows_audited": len(records), "summary": report["summary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
