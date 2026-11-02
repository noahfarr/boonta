from types import SimpleNamespace

from hangar.scripts.audit_agent_trove import audit_row, commands, outcome, summarize
from hangar.scripts.check_agent_trove_readiness import measure, summarize as readiness_summary


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=list(range(len(text))), offsets=[(i, i + 1) for i in range(len(text))])


def test_command_parser_ignores_thought_drafts_and_rejects_final_broken_action():
    draft = '<think>{"commands":[{"keystrokes":"draft"}]}</think>'
    assert commands(draft) == []
    content = draft + '\n{"commands":[{"keystrokes":"real"}]}'
    assert commands(content)[0][2][0]["keystrokes"] == "real"
    assert commands(content + '\n{"commands": [broken') == []


def test_outcome_does_not_treat_completion_claim_as_verified_success():
    assert outcome({"task_complete": True}) == "unknown"
    assert outcome({"result": "AgentTimeoutError"}) == "timeout"
    assert outcome({"result": "DaytonaNotFoundError"}) == "infrastructure_error"
    assert outcome({"reward": 0}) == "failure"
    assert outcome({"result": "success", "reward": 0}) == "conflict"


def test_audit_counts_mask_and_distinguishes_feedback_from_its_mention():
    row = {"conversations": [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": '{"commands":[{"keystrokes":"ls"}]}'},
        {"role": "user", "content": "New Terminal Output:\nfile.txt"},
        {"role": "assistant", "content": 'The previous output is fine. {"commands":[],"task_complete":true}'},
    ]}
    record = audit_row(1, row, Tokenizer(), 32)
    assert record["terminal_feedback_by_role"] == {"user": 1}
    assert record["output_markers_by_role"]["assistant"] == 1
    assert record["supervised_tokens"] < record["serialized_tokens"]
    assert record["outcome"] == "unknown"
    assert not record["flags"].get("unobserved_command_turn")
    assert record["later_windows_without_task_prefix"] > 0
    duplicate = {**record, "index": 2}
    report = summarize([record, duplicate])
    assert report["exact_trajectory_duplicates"]["extra_copies"] == 1
    assert report["totals"]["supervised_tokens"] == record["supervised_tokens"] * 2


def test_readiness_distinguishes_context_quality_from_retained_action_budget():
    row = {"messages": [
        {"role": "user", "content": "original task"},
        {"role": "assistant", "content": '{"commands":[{"keystrokes":"ls"}]}'},
        {"role": "tool", "content": "observation " * 10},
        {"role": "assistant", "content": '{"commands":[{"keystrokes":"cat file"}]}'},
    ]}
    record = measure(row, Tokenizer(), [64, 512])
    report = readiness_summary([record], [64, 512])
    assert record["all_action_tokens"] > 0
    assert report["contexts"]["64"]["retained"] == 0
    assert report["contexts"]["64"]["q_action_context"] is None
    assert report["contexts"]["512"]["retained"] == 1
    assert report["contexts"]["512"]["q_action_context"] == 1.0
    assert report["contexts"]["512"]["action_budget_retention"] == 1.0
    assert report["contexts"]["512"]["observation_body_tokens"] > 0
