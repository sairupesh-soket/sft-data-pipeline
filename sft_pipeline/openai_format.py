"""OpenAI chat format: message normalization shared by all source adapters.

Ported from hf_agentic_pipeline/scripts/common.py (logic unchanged, paths removed).
  - role names mapped to system / user / assistant / tool (human->user, gpt->assistant, observation->tool ...)
  - tool calls -> [{"id", "type": "function", "function": {"name", "arguments": <JSON string>}}]
  - legacy OpenAI "function_call" read as a tool call
  - tool definitions -> [{"type": "function", "function": {"name", "description", "parameters"}}]
  - parquet struct-union nulls stripped (only for values that came from a parquet STRUCT)
  - reasoning / reasoning_content / think kept in the separate field "reasoning_content"
  - ChatML text parsed into messages; split text+call turns rejoined; tool ids linked
"""
import ast
import json
import re

import numpy as np

ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "chatgpt": "assistant",
    "bot": "assistant",
    "tool": "tool",
    "function_call": "assistant",
    "function-call": "assistant",
    "observation": "tool",
    "function_response": "tool",
    "function": "tool",
    "tool_response": "tool",
    "ipython": "tool",
    "developer": "system",  # OpenAI's newer name for system-level instructions
}
CORE_ROLES = {"system", "user", "assistant", "tool"}


def to_plain(o):
    """Recursively convert numpy/pyarrow objects into plain Python types."""
    if isinstance(o, np.ndarray):
        return [to_plain(x) for x in o.tolist()]
    if isinstance(o, dict):
        return {k: to_plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_plain(x) for x in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def strip_nulls(o):
    """Drop None-valued dict keys, recursively (list elements are kept)."""
    if isinstance(o, dict):
        return {k: strip_nulls(v) for k, v in o.items() if v is not None}
    if isinstance(o, list):
        return [strip_nulls(x) for x in o]
    return o


def loads_loose(s):
    """json.loads, falling back to Python-literal syntax (single quotes)."""
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        v = ast.literal_eval(s)
        # Python-only literals (Ellipsis `...`, sets, bytes) parse fine here
        # but can't be written as JSON -- seen in Toucan tool arguments
        json.dumps(v)
        return v
    except Exception:
        return None


def normalize_args(args, from_struct=False):
    if isinstance(args, str):
        s = args.strip()
        if not s:
            return {}
        parsed = loads_loose(s)
        if isinstance(parsed, dict):
            return parsed  # came from a string: nulls are real, keep them
        return {"value": args}
    if args is None:
        return {}
    if isinstance(args, dict):
        return strip_nulls(args) if from_struct else args
    return {"value": args}


def normalize_tool_calls(tc, from_struct=False):
    tc = to_plain(tc)
    if not tc:
        return None
    if isinstance(tc, str):
        tc = loads_loose(tc)
        if not tc:
            return None
    if isinstance(tc, dict):
        tc = [tc]
    out = []
    for i, call in enumerate(tc):
        if not isinstance(call, dict):
            continue
        fn = call.get("function")
        if isinstance(fn, dict) and (fn.get("name") or fn.get("arguments") is not None):
            name, args = fn.get("name"), fn.get("arguments")
        else:
            name, args = call.get("name"), call.get("arguments", call.get("input"))
        if not name:
            continue
        out.append({
            "id": call.get("id") or f"call_{i}",
            "type": "function",
            # stored as a JSON STRING -- plain OpenAI format (standing user
            # rule); a template that wants a dict converts it at render time
            "function": {"name": name, "arguments": json.dumps(normalize_args(args, from_struct), ensure_ascii=False)},
        })
    return out or None


def normalize_tools(tools, from_struct=False):
    """Any tool-definition list -> OpenAI [{"type":"function","function":{...}}]."""
    tools = to_plain(tools)
    if isinstance(tools, str):
        tools = loads_loose(tools)
    if not tools:
        return None
    if isinstance(tools, dict):
        tools = [tools]
    out = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        if from_struct:
            fn = strip_nulls(fn)
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters", fn.get("input_schema"))
        if isinstance(params, str):
            params = loads_loose(params)
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        d = {"name": name}
        if fn.get("description"):
            d["description"] = fn["description"]
        d["parameters"] = params
        out.append({"type": "function", "function": d})
    return out or None


def convert_message(m, from_struct=False):
    m = to_plain(m)
    if not isinstance(m, dict):
        return None
    role = m.get("role") or m.get("from")
    if not role:
        return None
    role = str(role).strip().lower()
    if role == "tool_call" and not m.get("tool_calls"):
        # a call logged as its own message, content = {"name", "arguments"}
        # (TIGER-Lab SWE-QA-Pro, Toucan). Left as a role, the GLM template has
        # no branch for it and silently drops the whole call.
        d = m.get("content") or m.get("value")
        d = loads_loose(d) if isinstance(d, str) else d
        if isinstance(d, dict) and d.get("name"):
            m = {"role": "assistant", "content": "", "tool_calls": [d]}
    role = ROLE_MAP.get(role, role)

    content = m.get("content")
    if content is None:
        content = m.get("value")
    if isinstance(content, list):  # content-parts list -> plain text
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))

    tool_calls = normalize_tool_calls(m.get("tool_calls"), from_struct)
    if not tool_calls and m.get("function_call"):
        tool_calls = normalize_tool_calls([m["function_call"]], from_struct)
    if tool_calls:
        role = "assistant"

    reasoning = m.get("reasoning_content") or m.get("reasoning") or m.get("think")
    if content is None and not tool_calls and not reasoning:
        return None
    out = {"role": role, "content": content if content is not None else ""}
    if tool_calls:
        out["tool_calls"] = tool_calls
    if m.get("tool_call_id"):
        out["tool_call_id"] = m["tool_call_id"]
    if m.get("name") and role == "tool":
        out["name"] = m["name"]
    if reasoning:
        out["reasoning_content"] = reasoning
    return out


CHATML_RE = re.compile(r"<\|im_start\|>\s*(\w+)\s*\n(.*?)(?=<\|im_start\|>|\Z)", re.DOTALL)


def parse_chatml(text):
    msgs = []
    for role, content in CHATML_RE.findall(text):
        role = role.strip().lower()
        content = content.replace("<|im_end|>", "").strip("\n")
        if role in CORE_ROLES:
            msgs.append({"role": role, "content": content})
        elif msgs:
            msgs[-1]["content"] = (msgs[-1]["content"] + "\n\n" + content).strip()
        else:
            msgs.append({"role": "assistant", "content": content})
    return msgs or None


def merge_text_then_call(msgs):
    """assistant(text, no calls) immediately followed by assistant(calls, no
    text) is one turn split in two by the logging format (Toucan, Exgentic,
    OpenAI-style exports) -- rejoin it. Consecutive tool-call-only assistant
    messages (parallel calls logged one per message) are joined too."""
    out = []
    for m in msgs:
        if (out and m["role"] == "assistant" and out[-1]["role"] == "assistant"
                and m.get("tool_calls") and not (m.get("content") or "").strip()
                and not m.get("reasoning_content")):
            prev = out[-1]
            prev["tool_calls"] = (prev.get("tool_calls") or []) + m["tool_calls"]
            continue
        out.append(m)
    return out


def link_tool_ids(msgs):
    """Give every tool_call a unique id and point tool replies at them, in
    order, when the source didn't carry ids (glaive, Toucan...)."""
    pending = []
    n = 0
    for m in msgs:
        if m["role"] == "assistant" and m.get("tool_calls"):
            seen = set()
            for tc in m["tool_calls"]:
                if not tc.get("id") or re.fullmatch(r"call_\d+", tc["id"]) or tc["id"] in seen:
                    tc["id"] = f"call_{n}"
                seen.add(tc["id"])
                n += 1
            pending = [tc for tc in m["tool_calls"]]
        elif m["role"] == "tool":
            if pending:
                tc = pending.pop(0)
                if not m.get("tool_call_id") or re.fullmatch(r"call_\d+", str(m.get("tool_call_id"))):
                    m["tool_call_id"] = tc["id"]
                m.setdefault("name", tc["function"]["name"])
        elif m["role"] in ("user", "system"):
            pending = []
    return msgs


def finalize(msgs, tools=None, tools_from_struct=False):
    """Common last step for every converter. Returns (row|None, reason)."""
    msgs = [m for m in msgs if m is not None]
    if not msgs:
        return None, "skip_empty"
    msgs = merge_text_then_call(msgs)
    msgs = link_tool_ids(msgs)
    if all(m["role"] == "system" for m in msgs):
        return None, "skip_system_only"
    if not any(m["role"] == "assistant" for m in msgs):
        return None, "skip_no_assistant"
    row = {"messages": msgs}
    tools = normalize_tools(tools, tools_from_struct)
    if tools:
        row["tools"] = tools
    return row, "ok"


def simple_pair(user, assistant, system=None):
    msgs = []
    if system and str(system).strip():
        msgs.append({"role": "system", "content": str(system).strip()})
    if not user or not str(user).strip() or not assistant or not str(assistant).strip():
        return None, "skip_empty_side"
    msgs.append({"role": "user", "content": str(user).strip()})
    msgs.append({"role": "assistant", "content": str(assistant).strip()})
    return {"messages": msgs}, "ok"
