"""Normalize heterogeneous HF trace rows into OpenAI SFT chat format."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "human_message": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "ai": "assistant",
    "system": "system",
    "developer": "system",
    "tool": "tool",
    "function": "tool",
    "observation": "tool",
    "tool_response": "tool",
    "tool_result": "tool",
}


def _as_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float, bool)):
        return str(v)
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def _coerce_text(v: Any) -> Optional[str]:
    """
    Coerce QA field to text for SFT.
    - str → as-is
    - list[str] → joined
    - list[int] (token ids) → None (not decodable here)
    - other → None (avoid dumping raw structures into chat)
    """
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        if not v:
            return None
        if all(isinstance(x, str) for x in v):
            return "\n".join(x for x in v if x.strip())
        # Token-id sequences (e.g. bigscience P3) — skip
        if all(isinstance(x, int) for x in v):
            return None
        # Mixed / nested — not a clean text field
        return None
    if isinstance(v, (int, float, bool)):
        return str(v)
    return None


def _parse_maybe_json(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return v
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return v
    return v


def _normalize_role(role: Any) -> Optional[str]:
    if role is None:
        return None
    r = str(role).strip().lower()
    return ROLE_MAP.get(r, r if r in {"system", "user", "assistant", "tool"} else None)


def _normalize_tool_calls(tool_calls: Any) -> Optional[list[dict[str, Any]]]:
    tool_calls = _parse_maybe_json(tool_calls)
    if tool_calls is None:
        return None
    if isinstance(tool_calls, dict):
        tool_calls = [tool_calls]
    if not isinstance(tool_calls, list) or not tool_calls:
        return None

    out: list[dict[str, Any]] = []
    for i, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            continue
        # Already OpenAI-ish
        fn = tc.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            args = fn.get("arguments", "{}")
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            out.append(
                {
                    "id": str(tc.get("id") or f"call_{i}"),
                    "type": tc.get("type") or "function",
                    "function": {"name": str(fn["name"]), "arguments": args},
                }
            )
            continue
        # Flat {name, arguments} / {name, args}
        name = tc.get("name") or tc.get("tool_name")
        if name:
            args = tc.get("arguments", tc.get("args", tc.get("parameters", {})))
            if not isinstance(args, str):
                args = json.dumps(args if args is not None else {}, ensure_ascii=False)
            out.append(
                {
                    "id": str(tc.get("id") or tc.get("tool_call_id") or f"call_{i}"),
                    "type": "function",
                    "function": {"name": str(name), "arguments": args},
                }
            )
    return out or None


def _normalize_tools(tools: Any) -> Optional[list[dict[str, Any]]]:
    tools = _parse_maybe_json(tools)
    if tools is None:
        return None
    if isinstance(tools, dict):
        # sometimes {"tools": [...]} or name->schema map
        if "tools" in tools and isinstance(tools["tools"], list):
            tools = tools["tools"]
        else:
            tools = list(tools.values()) if tools else None
    if not isinstance(tools, list) or not tools:
        return None

    out: list[dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            out.append(t)
            continue
        name = t.get("name") or t.get("tool_name") or t.get("base_tool_name")
        if not name:
            continue
        params = (
            t.get("parameters")
            or t.get("input_schema")
            or t.get("inputSchema")
            or {"type": "object", "properties": {}}
        )
        fn: dict[str, Any] = {"name": str(name)}
        if t.get("description"):
            fn["description"] = str(t["description"])
        fn["parameters"] = params
        out.append({"type": "function", "function": fn})
    return out or None


def _message_from_dict(m: dict[str, Any]) -> Optional[dict[str, Any]]:
    role = _normalize_role(m.get("role") or m.get("from") or m.get("speaker") or m.get("author"))
    if not role:
        # ShareGPT-ish without role key
        if "human" in m and "value" not in m:
            return None
        return None

    content = m.get("content", m.get("value", m.get("text", m.get("message"))))
    # Multimodal content lists → stringify text parts
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if p.get("type") == "text" and "text" in p:
                    parts.append(str(p["text"]))
                elif "text" in p:
                    parts.append(str(p["text"]))
                else:
                    parts.append(_as_str(p) or "")
        content = "\n".join(parts)
    elif content is not None and not isinstance(content, str):
        content = _as_str(content)

    # Optional reasoning / thinking channel (final format field)
    reasoning = (
        m.get("reasoning_content")
        or m.get("reasoning")
        or m.get("thinking")
        or m.get("thought")
    )
    if reasoning is not None and not isinstance(reasoning, str):
        reasoning = _as_str(reasoning)
    if isinstance(reasoning, str) and not reasoning.strip():
        reasoning = None

    tool_calls = _normalize_tool_calls(m.get("tool_calls") or m.get("function_call"))
    msg: dict[str, Any] = {"role": role}

    if role == "tool":
        tcid = m.get("tool_call_id") or m.get("tool_call_ID") or m.get("id")
        if tcid:
            msg["tool_call_id"] = str(tcid)
        name = m.get("name")
        if name:
            msg["name"] = str(name)
        msg["content"] = content if content is not None else ""
        return msg

    if tool_calls:
        msg["tool_calls"] = tool_calls
        # Match final format: always include content (may be "")
        msg["content"] = content if content is not None else ""
        if reasoning:
            msg["reasoning_content"] = reasoning
        return msg

    if content is None or (isinstance(content, str) and content.strip() == ""):
        # Allow assistant with only reasoning_content (rare)
        if role == "assistant" and reasoning:
            msg["content"] = content if content is not None else ""
            msg["reasoning_content"] = reasoning
            return msg
        return None
    msg["content"] = content
    if role == "assistant" and reasoning:
        msg["reasoning_content"] = reasoning
    return msg


def normalize_message_list(messages: Any) -> list[dict[str, Any]]:
    messages = _parse_maybe_json(messages)
    if isinstance(messages, dict):
        # single message
        messages = [messages]
    if not isinstance(messages, list):
        return []

    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, str):
            # bare string — skip ambiguous
            continue
        if not isinstance(m, dict):
            continue
        # ShareGPT {from, value}
        if "from" in m and "value" in m and "role" not in m:
            nm = _message_from_dict(
                {
                    "role": m["from"],
                    "content": m["value"],
                    **{
                        k: v
                        for k, v in m.items()
                        if k in ("tool_calls", "tool_call_id", "name", "reasoning_content", "reasoning", "thinking")
                    },
                }
            )
        else:
            nm = _message_from_dict(m)
        if nm:
            out.append(nm)
    return out


def _lower_keys(row: dict[str, Any]) -> dict[str, Any]:
    return {str(k).lower(): v for k, v in row.items()}


def _get(row_l: dict[str, Any], *names: str) -> Any:
    for n in names:
        if n in row_l:
            return row_l[n]
    return None


def _qa_to_messages(user_text: str, assistant_text: str, system_text: Optional[str] = None) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    if system_text and str(system_text).strip():
        msgs.append({"role": "system", "content": str(system_text).strip()})
    msgs.append({"role": "user", "content": str(user_text).strip()})
    msgs.append({"role": "assistant", "content": str(assistant_text).strip()})
    return msgs


def extract_messages_and_tools(row: dict[str, Any]) -> tuple[Optional[list[dict[str, Any]]], Optional[list[dict[str, Any]]], str]:
    """
    Returns (messages, tools, reason_tag).
    messages is None if row cannot be converted.
    """
    if not isinstance(row, dict) or not row:
        return None, None, "empty_row"

    # LFS / garbage
    keys = list(row.keys())
    if keys and str(keys[0]).startswith("version https://git-lfs"):
        return None, None, "lfs_pointer"

    row_l = _lower_keys(row)
    tools = _normalize_tools(_get(row_l, "tools", "functions"))

    # --- Primary chat fields ---
    for field in ("messages", "conversations", "conversation", "chat", "dialogue", "dialog"):
        if field in row_l and row_l[field] is not None:
            msgs = normalize_message_list(row_l[field])
            if msgs:
                # prepend system if separate
                sys_txt = _get(row_l, "system", "system_prompt", "agent_system_prompt")
                if sys_txt and isinstance(sys_txt, str) and sys_txt.strip():
                    if not any(m.get("role") == "system" for m in msgs):
                        msgs = [{"role": "system", "content": sys_txt.strip()}] + msgs
                # stitch assistant_message / response if missing assistant
                roles = {m.get("role") for m in msgs}
                if "assistant" not in roles and "tool" not in roles:
                    am = _get(row_l, "assistant_message", "response", "answer", "output", "completion")
                    if isinstance(am, str) and am.strip():
                        # maybe tool_calls at top level
                        tcs = _normalize_tool_calls(_get(row_l, "tool_calls", "function_call"))
                        asst: dict[str, Any] = {"role": "assistant"}
                        if tcs:
                            asst["tool_calls"] = tcs
                            if am.strip():
                                asst["content"] = am.strip()
                        else:
                            asst["content"] = am.strip()
                        msgs.append(asst)
                    else:
                        # try raw_response
                        raw = _parse_maybe_json(_get(row_l, "raw_response"))
                        if isinstance(raw, dict):
                            try:
                                choice = (raw.get("choices") or [{}])[0]
                                msg = (choice.get("message") or {})
                                nm = _message_from_dict(msg) if isinstance(msg, dict) else None
                                if nm:
                                    msgs.append(nm)
                            except Exception:
                                pass
                        tcs = _normalize_tool_calls(_get(row_l, "tool_calls", "function_call"))
                        if tcs and not any(m.get("tool_calls") for m in msgs):
                            msgs.append({"role": "assistant", "tool_calls": tcs})
                return msgs, tools, f"from_{field}"

    # --- prompt as message list (+ optional completion) ---
    prompt = _get(row_l, "prompt")
    if prompt is not None:
        p_parsed = _parse_maybe_json(prompt)
        if isinstance(p_parsed, list) and p_parsed and isinstance(p_parsed[0], dict):
            msgs = normalize_message_list(p_parsed)
            comp = _get(row_l, "completion", "response")
            c_parsed = _parse_maybe_json(comp)
            if isinstance(c_parsed, list):
                msgs = msgs + normalize_message_list(c_parsed)
            elif isinstance(c_parsed, str) and c_parsed.strip():
                # Avoid duplicating if prompt already ends with same assistant
                if not (msgs and msgs[-1].get("role") == "assistant" and msgs[-1].get("content") == c_parsed.strip()):
                    # If prompt already contains full trajectory ending in assistant, skip empty response
                    msgs.append({"role": "assistant", "content": c_parsed.strip()})
            if msgs:
                return msgs, tools, "from_prompt_list"

        if isinstance(p_parsed, str) and p_parsed.strip():
            comp = _get(row_l, "completion", "response", "answer", "output")
            if isinstance(comp, str) and comp.strip():
                sys_txt = _get(row_l, "system", "system_prompt")
                return _qa_to_messages(p_parsed, comp, sys_txt if isinstance(sys_txt, str) else None), tools, "from_prompt_str"

    # --- turns / trajectory ---
    for field in ("turns", "trajectory", "trace"):
        if field in row_l and row_l[field] is not None:
            val = _parse_maybe_json(row_l[field])
            if isinstance(val, list) and val:
                msgs = normalize_message_list(val)
                if msgs:
                    return msgs, tools, f"from_{field}"
            if isinstance(val, dict) and "messages" in val:
                msgs = normalize_message_list(val["messages"])
                if msgs:
                    return msgs, tools, f"from_{field}_nested"

    # --- JSON blob with messages ---
    for field in ("conversation_json", "raw_response", "payload", "data"):
        if field not in row_l:
            continue
        val = _parse_maybe_json(row_l[field])
        if isinstance(val, dict):
            if "messages" in val:
                msgs = normalize_message_list(val["messages"])
                if msgs:
                    t2 = _normalize_tools(val.get("tools")) or tools
                    return msgs, t2, f"from_{field}"
            # claude-code style: request.messages
            req = val.get("request") if isinstance(val.get("request"), dict) else None
            if req and "messages" in req:
                msgs = normalize_message_list(req["messages"])
                # maybe append assistant from response
                resp = val.get("response")
                if isinstance(resp, dict):
                    content = resp.get("content") or resp.get("completion")
                    if isinstance(content, str) and content.strip():
                        msgs.append({"role": "assistant", "content": content.strip()})
                    elif isinstance(content, list):
                        # anthropic content blocks
                        texts = []
                        for b in content:
                            if isinstance(b, dict) and b.get("type") == "text":
                                texts.append(str(b.get("text", "")))
                        if texts:
                            msgs.append({"role": "assistant", "content": "\n".join(texts)})
                if msgs:
                    return msgs, tools, f"from_{field}_request"

    # --- Classic QA / instruction pairs (incl. odd column names) ---
    qa_pairs = [
        ("question", "answer"),
        ("question", "answer_content"),
        ("question", "reference_answer"),
        ("instruction", "output"),
        ("instruction", "response"),
        ("input", "output"),
        ("query", "response"),
        ("query", "answer"),
        ("problem", "solution"),
        ("problem", "expected_answer"),
        ("problem", "generated_solution"),
        ("human", "assistant"),
        ("prompt", "completion"),
        ("prompt", "response"),
        ("original_text", "translation"),
        ("source", "target"),
        ("src", "tgt"),
        # Flan / Aya / xP3 style
        ("inputs", "targets"),
        ("input", "target"),
        # OpenOrca / similar
        ("question", "response"),
        ("question", "generated_solution"),
        ("question", "expected_answer"),
    ]
    for u_key, a_key in qa_pairs:
        u = _coerce_text(_get(row_l, u_key))
        a = _coerce_text(_get(row_l, a_key))
        if isinstance(u, str) and isinstance(a, str) and u.strip() and a.strip():
            sys_txt = _get(row_l, "system", "system_prompt")
            sys_txt = _coerce_text(sys_txt) if sys_txt is not None else None
            # Prefer reasoning+answer if both exist for assistant
            reasoning = _coerce_text(_get(row_l, "reasoning"))
            asst = a.strip()
            if isinstance(reasoning, str) and reasoning.strip() and a_key != "reasoning":
                # only prepend if answer doesn't already contain it
                if reasoning.strip() not in asst:
                    asst = reasoning.strip() + "\n\n" + asst
            return (
                _qa_to_messages(u, asst, sys_txt if isinstance(sys_txt, str) else None),
                tools,
                f"qa_{u_key}_{a_key}",
            )

    # question + reasoning (no explicit answer) — weak but usable if SQL/answer elsewhere
    q = _get(row_l, "question")
    reasoning = _get(row_l, "reasoning")
    extra_ans = _get(row_l, "sql", "solution", "generated_solution")
    if isinstance(q, str) and q.strip() and isinstance(reasoning, str) and reasoning.strip():
        asst = reasoning.strip()
        if isinstance(extra_ans, str) and extra_ans.strip():
            asst = asst + "\n\n" + extra_ans.strip()
        sys_txt = _get(row_l, "system", "system_prompt")
        return (
            _qa_to_messages(q, asst, sys_txt if isinstance(sys_txt, str) else None),
            tools,
            "qa_question_reasoning",
        )

    return None, None, "unconvertible:" + ",".join(str(k) for k in list(row.keys())[:8])


def validate_sft_messages(messages: list[dict[str, Any]]) -> tuple[bool, str]:
    if not messages:
        return False, "no_messages"
    if not any(m.get("role") == "assistant" for m in messages):
        return False, "no_assistant"
    for m in messages:
        role = m.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            return False, f"bad_role:{role}"
        if role == "assistant":
            has_content = isinstance(m.get("content"), str) and m["content"].strip() != ""
            has_tools = bool(m.get("tool_calls"))
            has_reasoning = isinstance(m.get("reasoning_content"), str) and m["reasoning_content"].strip() != ""
            if not has_content and not has_tools and not has_reasoning:
                return False, "empty_assistant"
        if role == "tool":
            if not m.get("tool_call_id"):
                return False, "tool_missing_tool_call_id"
            if m.get("content") is None:
                return False, "tool_missing_content"
        if role in {"system", "user"}:
            if not isinstance(m.get("content"), str) or not m["content"].strip():
                return False, f"empty_{role}"
    # Prefer ending on assistant or tool (OpenAI SFT train targets)
    if messages[-1].get("role") not in {"assistant", "tool"}:
        return False, "does_not_end_assistant_or_tool"
    return True, "ok"


def trim_to_trainable(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop trailing user/system turns so the example ends on assistant or tool."""
    if not messages:
        return messages
    end = len(messages)
    while end > 0 and messages[end - 1].get("role") in {"user", "system"}:
        end -= 1
    return messages[:end]


def build_sft_record(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """
    Final OpenAI-style SFT record:
      {messages, tools?, parallel_tool_calls?}
    Message fields kept: role, content, reasoning_content?, tool_calls?, tool_call_id?, name?
    """
    clean_msgs: list[dict[str, Any]] = []
    for m in messages:
        role = m["role"]
        cm: dict[str, Any] = {"role": role}

        if role == "assistant":
            # Always include content ("" allowed when tool_calls / reasoning only)
            cm["content"] = m["content"] if isinstance(m.get("content"), str) else ""
            if isinstance(m.get("reasoning_content"), str) and m["reasoning_content"].strip():
                cm["reasoning_content"] = m["reasoning_content"]
            if m.get("tool_calls"):
                cm["tool_calls"] = m["tool_calls"]
        elif role == "tool":
            cm["tool_call_id"] = m["tool_call_id"]
            cm["content"] = m.get("content", "")
            if m.get("name"):
                cm["name"] = m["name"]
        else:
            cm["content"] = m.get("content", "")

        clean_msgs.append(cm)

    rec: dict[str, Any] = {"messages": clean_msgs}
    if tools:
        rec["tools"] = tools
        parallel = any(len(m.get("tool_calls") or []) > 1 for m in clean_msgs)
        rec["parallel_tool_calls"] = parallel
    return rec
