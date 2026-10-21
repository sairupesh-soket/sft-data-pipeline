"""Every documented rule example must behave as documented:
  FLAGGED row -> the validation gives the rule id;   CORRECT FORM row -> the validation gives no FLAG rule.
Run: python3 tests/test_rule_examples.py"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "docs"))
sys.path.insert(0, os.path.join(HERE, "..", "sft_pipeline", "validate"))
import validation_rules as V   # noqa: E402
import dataset_check as D      # noqa: E402

_render = None


def issues(row, path, render):
    global _render
    if render and _render is None:
        D.M5.init(); _render = True
    return D.row_issues(row, json.dumps(row).encode(), path or "data/x.jsonl", render)


def has(iss, rid):
    return any(x == rid or x.startswith(rid + "_") or x.startswith(rid + ":") for x in iss)


def check():
    bad = []
    for g, r in V.R:
        f = issues(r["flagged"], r["path"], r["render"])
        c = issues(r["correct"], None, r["render"])
        if not has(f, r["id"]):
            bad.append((r["id"], "flagged row does not get the rule", sorted(f)))
        flags_c = c - D.INFO_ONLY
        if r["label"] == "FLAG" and flags_c:
            bad.append((r["id"], "correct row is flagged", sorted(flags_c)))
        if r["label"] == "INFO" and (f - D.INFO_ONLY):
            bad.append((r["id"], "INFO example has a FLAG rule", sorted(f - D.INFO_ONLY)))
    return bad


def test_rule_examples():
    assert not check()


if __name__ == "__main__":
    b = check()
    for x in b:
        print("FAIL", x)
    print(f"{len(V.R)} rules checked, failures: {len(b)}")
    sys.exit(1 if b else 0)
