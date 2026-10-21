"""Small final repairs. Ported from think_scan_deduped/check3/clean_final.py (fix_row; the <think> move there is
covered by reasoning.move_think_step).
  final_toolcall_after_answer : final assistant turn has a real answer AND tool_calls, nothing after it
                                -> the dangling tool_calls are removed (the answer stays)
  trailing_literal_newline    : literal backslash-n characters at the END of user / assistant text -> removed
  dup_reasoning               : reasoning_content identical to the answer -> reasoning_content removed
"""
import re

TRAIL = re.compile(r"(\\n)+\s*$")


def step(row, c):
    ch = False
    msgs = row.get("messages") or []
    last = msgs[-1] if msgs else {}
    if last.get("role") == "assistant" and last.get("tool_calls") and (last.get("content") or "").strip():
        del last["tool_calls"]; c["fixed:final_toolcall_removed"] += 1; ch = True
    for m in msgs:
        cc = m.get("content")
        if isinstance(cc, str) and m.get("role") in ("user", "assistant") and TRAIL.search(cc):
            m["content"] = TRAIL.sub("", cc); c["fixed:literal_newline"] += 1; ch = True
        rc = m.get("reasoning_content")
        cc = m.get("content")
        if m.get("role") == "assistant" and isinstance(rc, str) and rc.strip() and isinstance(cc, str) and rc.strip() == cc.strip():
            del m["reasoning_content"]; c["fixed:dup_reasoning_removed"] += 1; ch = True
    return ("changed" if ch else "keep"), row
