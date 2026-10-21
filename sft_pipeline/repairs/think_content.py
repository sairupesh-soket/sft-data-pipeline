"""Repairs for reasoning written as <think>...</think> INSIDE the content (run these BEFORE reasoning.move_think).
Ported from hf_agentic_pipeline/scripts/fix_think_issues.py (logic unchanged).
  1. split turn: assistant message that is only <think>...</think>, followed by another assistant message -> merged
  2. problem-creation reasoning ("The user wants me to create ... problem") on a SOLVE request -> the <think> block is removed
  3a. the only answer is inside <think> but contains a final answer (\\boxed{}, "final answer") -> tags removed
  3. last assistant turn that is only <think>...</think> -> removed; row trimmed to its last real assistant turn
"""
import re

CREATION = re.compile(
    r"^<think>\s*The user wants (me to (create|generate|write|design)|an? (new|original|completely original)"
    r"|a completely original)[^.]{0,200}?\b(problem|question)", re.I)
USER_ASKS_CREATE = re.compile(
    r"\b(create|generate|write|design|compose|make up|come up with)\b.{0,80}\b(problem|question|exercise|puzzle)",
    re.I | re.S)


FINAL_MARK = re.compile(r"\\boxed\{|final answer|the answer is|answer:\s*\(?[A-E]\b", re.I)


def after_think(c):
    return c.split("</think>")[-1].strip() if "</think>" in c else c.strip()


def think_only(m):
    c = m.get("content")
    return (m.get("role") == "assistant" and not m.get("tool_calls") and isinstance(c, str)
            and "</think>" in c and not after_think(c))


def fix_row(msgs, c):
    changed = False
    # 1. merge split turns
    out = []
    for m in msgs:
        if out and m.get("role") == "assistant" and think_only(out[-1]):
            prev = out.pop()
            m = dict(m)
            m["content"] = prev["content"].rstrip() + "\n\n" + (m.get("content") or "")
            c["split_turn_merged"] += 1
            changed = True
        out.append(m)
    msgs = out
    # 2. problem-creation reasoning on a solve request
    last_user = ""
    for i, m in enumerate(msgs):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            last_user = m["content"]
        elif m.get("role") == "assistant" and isinstance(m.get("content"), str) \
                and CREATION.search(m["content"][:600]) and "</think>" in m["content"] \
                and not USER_ASKS_CREATE.search(last_user[:800]):
            msgs[i] = dict(m, content=after_think(m["content"]))
            c["creation_think_removed"] += 1
            changed = True
    # 3a. the ONLY answer sits inside <think> but does contain a final answer
    # (shaafsalman / hoanganhpham put "</think>" after their final answer):
    # unwrap it into a normal visible answer instead of dropping the row.
    asst = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
    if len(asst) == 1 and asst[0] == len(msgs) - 1 and think_only(msgs[-1]) \
            and FINAL_MARK.search(msgs[-1]["content"][-3000:]):
        inner = msgs[-1]["content"].replace("<think>", "").replace("</think>", "").strip()
        msgs[-1] = dict(msgs[-1], content=inner)
        c["answer_inside_think_unwrapped"] += 1
        changed = True
    # 3. empty final answer -> trim / drop
    while msgs and msgs[-1].get("role") == "assistant" and think_only(msgs[-1]):
        msgs = msgs[:-1]
        last = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=-1)
        msgs = msgs[: last + 1]
        c["think_only_final_removed"] += 1
        changed = True
    if not any(m.get("role") == "assistant" for m in msgs):
        return None, True
    return msgs, changed


def step(row, c):
    """runner step: think-in-content repairs."""
    msgs = row.get("messages") or []
    if not any(isinstance(m.get("content"), str) and "</think>" in m["content"] for m in msgs):
        return "keep", row
    new, changed = fix_row(msgs, c)
    if new is None:
        return "drop:no_answer_after_think_repair", None
    if changed:
        row = dict(row, messages=new)
        return "changed", row
    return "keep", row
