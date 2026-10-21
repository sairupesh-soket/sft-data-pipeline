"""Plain OpenAI format for every message. Adapted from hf_agentic_pipeline/scripts/to_openai_format.py
(fix_msg) for the CURRENT format: reasoning stays in its own field "reasoning_content".
  - tool_calls[] -> {"id", "type": "function", "function": {"name", "arguments": <JSON string>}}
  - content null -> ""
  - empty reasoning_content removed; reasoning_content only on assistant messages
  - message keys other than the allowed ones are removed; top-level keys other than messages / tools removed
"""
import json

ALLOWED = {
    "system": {"role", "content", "name"},
    "user": {"role", "content", "name"},
    "assistant": {"role", "content", "name", "tool_calls", "reasoning_content"},
    "tool": {"role", "content", "tool_call_id"},
}


def fix_msg(m, c):
    m = dict(m)
    r = m.get("reasoning_content")
    if r is not None and not (isinstance(r, str) and r.strip()):
        del m["reasoning_content"]
        c["empty_reasoning_dropped"] += 1
    if m.get("tool_calls"):
        calls = []
        for tc in m["tool_calls"]:
            fn = tc.get("function") or {}
            a = fn.get("arguments")
            if not isinstance(a, str):
                a = json.dumps(a if a is not None else {}, ensure_ascii=False)
                c["args_to_string"] += 1
            new = {"id": tc.get("id"), "type": "function", "function": {"name": fn.get("name"), "arguments": a}}
            if new != tc:
                c["tool_call_normalised"] += 1
            calls.append(new)
        m["tool_calls"] = calls
    if m.get("content") is None:
        m["content"] = ""
        c["null_content_to_empty"] += 1
    allowed = ALLOWED.get(m.get("role"), set(m))
    for k in [k for k in m if k not in allowed]:
        c[f"dropped_key:{m.get('role')}.{k}"] += 1
        del m[k]
    return m


def step(row, c):
    msgs = row.get("messages")
    if not msgs:
        return "drop:empty_messages", None
    new = {"messages": [fix_msg(m, c) for m in msgs]}
    if row.get("tools"):
        new["tools"] = row["tools"]
    for k in row:
        if k not in ("messages", "tools"):
            c[f"dropped_top_key:{k}"] += 1
    return ("keep" if new == row else "changed"), new
