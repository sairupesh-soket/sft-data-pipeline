"""Run:  python3 tests/test_pipeline.py      (or: python3 -m pytest tests)"""
import copy
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cases import REPAIR_CASES, ADAPTER_CASES          # noqa: E402
from sft_pipeline.clean import STEPS                   # noqa: E402
from sft_pipeline import openai_format as OF           # noqa: E402
from sft_pipeline.adapters import schema_infer as SI   # noqa: E402


def run_repair(step, row):
    status, out = STEPS[step](copy.deepcopy(row), Counter())
    return status if status.startswith("drop") else out


def run_adapter(fn, inp):
    if fn == "convert_message":
        return OF.convert_message(copy.deepcopy(inp))
    if fn == "parse_chatml":
        return OF.parse_chatml(inp)
    if fn == "generic":
        return SI.to_messages(inp, SI.plan(list(inp)))
    raise ValueError(fn)


def test_repairs():
    for step, title, inp, exp in REPAIR_CASES:
        got = run_repair(step, inp)
        assert got == exp, f"{step}: {title}\n got: {got}\n exp: {exp}"


def test_adapters():
    for title, fn, inp, exp in ADAPTER_CASES:
        got = run_adapter(fn, inp)
        assert got == exp, f"{fn}: {title}\n got: {got}\n exp: {exp}"


if __name__ == "__main__":
    bad = 0
    for t in (test_repairs, test_adapters):
        try:
            t(); print("ok  ", t.__name__)
        except AssertionError as e:
            bad += 1; print("FAIL", t.__name__, e)
    print(f"{len(REPAIR_CASES)} repair cases, {len(ADAPTER_CASES)} adapter cases")
    sys.exit(1 if bad else 0)
