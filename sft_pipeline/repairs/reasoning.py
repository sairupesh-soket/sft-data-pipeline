"""Move reasoning out of the answer into the separate field reasoning_content.
move_think_step : ported from scripts_used/step1_think_fix/fix_think_to_reasoning.py (fix_row, unchanged)
    "<think>R</think>A" (one block, at the start)        -> reasoning_content = R, content = A
    "<think><think>R</think>A" (2 open, 1 close)          -> one <think> removed, then as above
    "<think>  </think>A" (empty think)                    -> content = A, no reasoning_content
    dropped: <think> not closed, </think> without <think>, 2+ blocks, block not at the start,
             <think> in a user message, empty final answer after the move
move_other_tags_step : ported from scripts_used/step2_fix2/fix2.py (try_move, unchanged)
    other tag styles (<thinking>, <reasoning>, <|begin_of_thought|>, <scratchpad>, <thought>, ...) at the start
    -> reasoning_content; any other placement -> dropped
"""
import re

OPEN, CLOSE = "<think>", "</think>"
DOUBLE = re.compile(r"^\s*<think>\s*(?=<think>)")

def fix_row(row, system_mode):
    """Return (new_row or None, reason/str status)."""
    msgs = row.get("messages") or []
    for m in msgs:
        t = m.get("content")
        if m.get("role") == "user" and isinstance(t, str) and (OPEN in t or CLOSE in t):
            return None, "drop_user_has_think"
        if m.get("role") == "system" and isinstance(t, str) and (OPEN in t or CLOSE in t) and system_mode == "drop":
            return None, "drop_system_has_think"
    changed = False
    status = "unchanged"
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        t = m.get("content")
        if not isinstance(t, str):
            continue
        o, c = t.count(OPEN), t.count(CLOSE)
        if not (o or c):
            continue
        if o == 2 and c == 1 and DOUBLE.match(t):
            t = DOUBLE.sub("", t, count=1)
            o = 1
            status = "fixed_b1_double_open"
        if o == 1 and c == 1 and t.lstrip().startswith(OPEN):
            body = t.lstrip()[len(OPEN):]
            reasoning, answer = body.split(CLOSE, 1)
            new = {}
            for k, v in m.items():
                if k == "content":
                    new["content"] = answer.strip()
                    if reasoning.strip():
                        new["reasoning_content"] = reasoning.strip()
                elif k != "reasoning_content":
                    new[k] = v
            m.clear(); m.update(new)
            changed = True
            if status == "unchanged":
                status = "moved" if reasoning.strip() else "moved_empty_think"
            continue
        if o > c:
            return None, "drop_think_not_closed"
        if c > o:
            return None, "drop_close_without_open"
        if o > 1:
            return None, "drop_multi_think_blocks"
        return None, "drop_think_not_at_start"
    if changed and msgs and msgs[-1].get("role") == "assistant":
        last = msgs[-1]
        if not (last.get("content") or "").strip() and not last.get("tool_calls"):
            return None, "drop_final_answer_empty_after_move"
    return (row if changed else None), status


def move_think_step(row, c, system_mode="keep"):
    msgs = row.get("messages") or []
    if not any(isinstance(m.get("content"), str) and ("<think>" in m["content"] or "</think>" in m["content"]) for m in msgs):
        return "keep", row
    new, status = fix_row(row, system_mode)
    c["think:" + status] += 1
    if status.startswith("drop_"):
        return "drop:" + status[5:], None
    return ("changed" if new is not None else "keep"), (new if new is not None else row)


PAIRS = [  # name, open, close
    ("thinking", r"<thinking>", r"</thinking>"),
    ("reasoning", r"<reasoning>", r"</reasoning>"),
    ("begin_of_thought", r"<\|begin_of_thought\|>", r"<\|end_of_thought\|>"),
    ("scratchpad", r"<scratchpad>", r"</scratchpad>"),
    ("THOUGHT", r"<THOUGHT>", r"</THOUGHT>"),
    ("startthought", r"<\|startthought\|>", r"<\|endthought\|>"),
    ("pipe_thinking", r"<\|thinking\|>", r"</\|thinking\|>|<\|actual_response\|>"),
    ("analysis", r"<analysis>", r"</analysis>"),
    ("thought", r"<thought>", r"</thought>"),
]
ANY_TAG = re.compile(r"</?thinking>|</?reasoning>|<\|(?:begin|end)_of_thought\|>|</?scratchpad>|</?THOUGHT>"
                     r"|<\|(?:start|end)thought\|>|<\|/?thinking\|>|<reasoning_start>|<\|start_of_reasoning\|>"
                     r"|<\|reasoning\|>|</?analysis>|</?thought>|</\|thinking\|>|</?\|actual_response\|>|◁/?think▷|</?seed:think>", re.I)
FLAGS = {"thinking": re.I, "reasoning": re.I, "thought": re.I}
SOLUTION = re.compile(r"^\s*<\|begin_of_solution\|>\s*|\s*<\|end_of_solution\|>\s*$")
ACTUAL_END = re.compile(r"^\s*<\|actual_response\|>\s*|\s*</\|actual_response\|>\s*$")
ROOT = None


def try_move(c):
    for name, o, cl in PAIRS:
        fl = FLAGS.get(name, 0)
        m = re.match(r"\s*(?:" + o + r")(.*?)(?:" + cl + r")(.*)\Z", c, re.S | fl)
        if not m:
            continue
        r, a = m.group(1).strip(), m.group(2)
        if name == "begin_of_thought":
            a = SOLUTION.sub("", a)
        if name == "pipe_thinking":
            a = ACTUAL_END.sub("", a)
        a = a.strip()
        if not r or not a or ANY_TAG.search(a) or ANY_TAG.search(r):
            return None
        return r, a
    return None


def move_other_tags_step(row, c):
    msgs = row.get("messages") or []
    changed = False
    for i, m in enumerate(msgs):
        cc = m.get("content")
        if m.get("role") != "assistant" or not isinstance(cc, str) or not ANY_TAG.search(cc):
            continue
        res = None if m.get("reasoning_content") else try_move(cc)
        if res is None:
            c["tags:drop_reasoning_tag_unmovable"] += 1
            return "drop:reasoning_tag_unmovable", None
        r, a = res
        new = {}
        for k, v in m.items():
            if k == "content":
                new["content"] = a
                new["reasoning_content"] = r
            elif k != "reasoning_content":
                new[k] = v
        msgs[i] = new
        changed = True
        c["tags:moved_reasoning_tag"] += 1
    return ("changed" if changed else "keep"), row
