"""Every hidden-content format in tests/embedded_cases.py must be flagged; plain answers must pass.
Run: python3 tests/test_embedded.py"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "sft_pipeline", "validate"))
import dataset_check as D   # noqa: E402
import embedded_cases as E  # noqa: E402


def flags(msgs):
    r = {"messages": msgs}
    return D.row_issues(r, json.dumps(r).encode(), "x.jsonl", False) - D.INFO_ONLY


def as_assistant(t):
    return [{"role": "user", "content": "Time in Rome?"}, {"role": "assistant", "content": t}]


def as_user(t):
    return [{"role": "user", "content": "Time in Rome?"}, {"role": "assistant", "content": "Let me check."},
            {"role": "user", "content": t}, {"role": "assistant", "content": "It is 10:00."}]


CASES = ([("assistant", k, v) for g in (E.R, E.TC, E.RM, E.AG) for k, v in g.items()] +
         [("user", k, v) for g in (E.TR, E.AGU) for k, v in g.items()])
PLAIN = ["It is 10:00 in Rome.", "Thoughts on this: it is 10:00.", "Use the format {\"name\": value} in JSON.",
         "Observation is important in science.", "The output of the program is 4."]


def test_embedded_flagged():
    for where, k, v in CASES:
        assert flags(as_assistant(v) if where == "assistant" else as_user(v)), k


def test_plain_pass():
    for t in PLAIN:
        assert not flags(as_assistant(t)), t


if __name__ == "__main__":
    bad = [k for w, k, v in CASES if not flags(as_assistant(v) if w == "assistant" else as_user(v))]
    badp = [t for t in PLAIN if flags(as_assistant(t))]
    print(f"{len(CASES)} hidden formats, missed: {bad}; {len(PLAIN)} plain answers, flagged by mistake: {badp}")
    sys.exit(1 if bad or badp else 0)
