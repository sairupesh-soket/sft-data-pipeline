"""Repair pass: run the repair steps, in order, over a folder of OpenAI-format JSONL.

    python3 -m sft_pipeline.clean SRC OUT [--steps a,b,c] [--skip x,y] [--log DIR] [--nproc N] [--inplace]
                                          [--odia-files PATTERN] [--gate]

SRC is never changed (unless --inplace). OUT gets the same folder layout. Removed rows are written, with the reason,
to LOG/dropped/<rel>; counts to LOG/pass_counts.json.

Default step order (each step: see its module):
  format            repairs/format.py         plain OpenAI keys; arguments as JSON string; reasoning_content kept
  shapes            repairs/shapes.py         flat tool defs, Kimi tool-declare, null answers, <answer> in <think>
  hermes            repairs/tools.py          Hermes <tool_call> text -> OpenAI tool_calls (all or nothing per row)
  tool_ids          repairs/tools.py          tool replies linked to their call id; text-call replies -> user
  think_content     repairs/think_content.py  split think/answer turns, problem-creation think, answer inside think
  move_think        repairs/reasoning.py      <think>R</think>A -> reasoning_content R, content A
  move_other_tags   repairs/reasoning.py      <thinking>, <reasoning>, <|begin_of_thought|> ... -> reasoning_content
  harness_fallback  repairs/roles.py          agent-harness fallback text at the end -> trim
  final             repairs/final.py          dangling tool call after answer, trailing literal \\n, duplicate reasoning
  role_rules        repairs/roles.py          remove empty assistant turns, end on assistant, no assistant after system
Optional (not in the default list):
  text_drops        repairs/text_drops.py     self-talk / cut-off final answers
--gate: after the steps, a row that fails any validation rule (FLAG rules of validate/) is not written; it is put
        in LOG/dropped with the rule ids. Without --gate the validation only flags rows (run validate separately).
"""
import argparse
import json
import os
import sys
from collections import Counter

from . import runner
from .repairs import format as r_format, shapes, tools, think_content, reasoning, roles, final, text_drops

STEPS = {
    "format": r_format.step,
    "shapes": shapes.step,
    "hermes": tools.hermes_step,
    "tool_ids": tools.link_tool_ids_step,
    "think_content": think_content.step,
    "move_think": reasoning.move_think_step,
    "move_other_tags": reasoning.move_other_tags_step,
    "harness_fallback": roles.harness_fallback_step,
    "final": final.step,
    "role_rules": roles.role_rules_step,
    "text_drops": text_drops.step,
}
DEFAULT = ["format", "shapes", "hermes", "tool_ids", "think_content", "move_think", "move_other_tags",
           "harness_fallback", "final", "role_rules"]


def gate_step():
    """Validation rules as a final step (only with --gate)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "validate"))
    import dataset_check as DC
    import orjson

    def step(row, c):
        iss = DC.row_issues(row, orjson.dumps(row), "", False) - DC.INFO_ONLY
        if iss:
            for k in iss:
                c["gate:" + k] += 1
            return "drop:" + ",".join(sorted(iss)), None
        return "keep", row
    return step


def build_steps(names, odia_pattern=None, gate=False):
    out = []
    for n in names:
        if n not in STEPS:
            raise SystemExit(f"unknown step {n}; known: {', '.join(STEPS)}")
        out.append(STEPS[n])
    if gate:
        out.append(gate_step())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--steps", help="comma list (default: %s)" % ",".join(DEFAULT))
    ap.add_argument("--skip", default="", help="comma list of steps to leave out")
    ap.add_argument("--log", help="log folder (default: <out>_logs, outside the data folder)")
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--inplace", action="store_true", help="rewrite SRC (needs OUT == SRC)")
    ap.add_argument("--gate", action="store_true", help="also remove rows that fail a validation FLAG rule")
    a = ap.parse_args(argv)
    names = a.steps.split(",") if a.steps else list(DEFAULT)
    names = [n for n in names if n and n not in set(a.skip.split(","))]
    steps = build_steps(names, gate=a.gate)
    print("steps:", " -> ".join(names + (["gate"] if a.gate else [])), flush=True)
    tot = runner.run_pass(a.src, a.out, steps, log=a.log, nproc=a.nproc, inplace=a.inplace)
    print(json.dumps(dict(sorted(tot.items())), indent=1))


if __name__ == "__main__":
    main()
