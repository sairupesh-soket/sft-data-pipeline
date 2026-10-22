#!/usr/bin/env python3
"""
Adapt Langfuse / Cursor prod traces to native message JSONL.

Trace exports do NOT include the live Cursor system prompt or toolDefinitions.
For mid-training / trace SFT, default is NO synthetic system message (unlike
SWE-agent convert_trajectories.py which injects a full tool schema).

Tools use Cursor span names: read_file, edit_file, shell, grep_search.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterator


# OpenAI function-calling tool schemas for Cursor-style spans (SFT / SLIME native path).
CURSOR_OPENAI_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "extension": {"type": "string", "description": "File extension if known"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Apply edits to a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "old_string": {"type": "string", "description": "Text to replace"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Search for a pattern in the codebase or a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex or search pattern"},
                    "path": {"type": "string", "description": "File or directory to search"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command in the project environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                },
                "required": ["path"],
            },
        },
    },
]

# Cursor-style system prompt used for SFT (also seed for prompt mutations).
_PROMPT_PATH = Path(__file__).resolve().parent / "cursor_system_prompt.md"
CURSOR_SFT_SYSTEM = _PROMPT_PATH.read_text(encoding="utf-8").strip()


def trunc(x: Any, n: int | None = None) -> str:
    """Serialize payload to text. No length cap by default (full expansion)."""
    if x is None:
        return ""
    if isinstance(x, (dict, list)):
        s = json.dumps(x, ensure_ascii=False)
    else:
        s = str(x)
    if n is None or len(s) <= n:
        return s
    return s[:n] + "\n... [truncated]"


def payload_nonempty(x: Any) -> bool:
    """True if a Langfuse input/output field actually carries content."""
    if x is None:
        return False
    if isinstance(x, str):
        return bool(x.strip())
    if isinstance(x, (dict, list)):
        return bool(x)
    return True


def observation_has_io(obs: dict) -> bool:
    return payload_nonempty(obs.get("input")) or payload_nonempty(obs.get("output"))


def unwrap_json_string(s: Any) -> str:
    if not isinstance(s, str):
        return trunc(s)
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] == '"':
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    return s


def obs_title(obs: dict) -> str:
    return obs.get("title") or obs.get("name") or ""


def obs_time(obs: dict) -> str:
    md = obs.get("metadata") or {}
    if isinstance(md, str):
        try:
            md = json.loads(md)
        except json.JSONDecodeError:
            md = {}
    return (
        obs.get("startTime")
        or obs.get("start_time")
        or md.get("start_time")
        or ""
    )


def _as_tool_args(inp: Any) -> dict:
    if inp is None:
        return {}
    if isinstance(inp, dict):
        return inp
    if isinstance(inp, str):
        s = inp.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return {"raw": inp}
    return {"value": inp}


def map_span_to_cursor_tool(obs: dict) -> tuple[str, dict, str]:
    """Map SPAN to OpenAI tool name + args.

    Prefers session-native ``tool:<name>`` titles (llm-logging / agent apps).
    Falls back to Cursor IDE span names when present.
    """
    title = obs_title(obs)
    inp = obs.get("input")
    out = obs.get("output")

    # Native agent tools: "tool:allocate_and_disburse"
    if title.startswith("tool:"):
        name = title.split(":", 1)[1].strip() or "unknown_tool"
        return name, _as_tool_args(inp), trunc(out)

    if isinstance(inp, str):
        inp_dict: dict = {"raw": inp}
    elif isinstance(inp, dict):
        inp_dict = inp
    else:
        inp_dict = {}

    if title.startswith("shell-exec"):
        cmd = inp_dict.get("command")
        if not cmd and ":" in title:
            cmd = title.split(":", 1)[1].strip()
        return "shell", {"command": cmd or ""}, trunc(out)

    if title.startswith("read-file"):
        path = inp_dict.get("file_path")
        if not path and ":" in title:
            path = title.split(":", 1)[1].strip()
        args = {"path": path or ""}
        if inp_dict.get("extension"):
            args["extension"] = inp_dict["extension"]
        return "read_file", args, trunc(out)

    if title.startswith("edit-file"):
        path = inp_dict.get("file_path")
        if not path and ":" in title:
            path = title.split(":", 1)[1].strip()
        args = {"path": path or ""}
        if isinstance(out, dict):
            edits = out.get("edits") or []
            if edits:
                args["old_string"] = edits[0].get("old_string") or ""
                args["new_string"] = edits[0].get("new_string") or ""
        return "edit_file", args, trunc(out) if out else "OK"

    if title.startswith("grep-search"):
        pattern = inp_dict.get("pattern") or ""
        if not pattern and ":" in title:
            pattern = title.split(":", 1)[1].strip()
        path = inp_dict.get("file_path") or inp_dict.get("search_path") or "."
        return "grep_search", {"pattern": pattern or "", "path": path or "."}, trunc(out)

    if title.startswith("delete-file"):
        path = inp_dict.get("file_path") or title
        return "delete_file", {"path": str(path)}, trunc(out) or "deleted"

    return "unknown_tool", {"title": title}, trunc(out)


def _unwrap_chat_input_obj(chat_input: Any) -> Any:
    if isinstance(chat_input, str):
        s = chat_input.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return chat_input
    return chat_input


def extract_openai_tools(chat_input: Any) -> list[dict]:
    """Pull OpenAI ``tools[]`` schemas from chat_input when present."""
    obj = _unwrap_chat_input_obj(chat_input)
    if not isinstance(obj, dict):
        return []
    tools = obj.get("tools")
    if not isinstance(tools, list):
        return []
    out: list[dict] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            name = t["function"].get("name")
            if isinstance(name, str) and name.strip():
                out.append(t)
    return out


def collect_session_tools(traces: list[dict]) -> list[dict]:
    """Union of tool schemas declared across traces (by function name)."""
    by_name: dict[str, dict] = {}
    for trace in traces:
        for t in extract_openai_tools(trace.get("chat_input")):
            name = t["function"]["name"]
            by_name.setdefault(name, t)
    return [by_name[k] for k in sorted(by_name)]


_BRAND_PHRASES = [
    (re.compile(r"(?im)^x-anthropic-billing-header:.*\n*", re.M), ""),
    (
        re.compile(
            r"You are Chakra[^\n]*\n(?:If asked[^\n]*\n)*"
            r"(?:Never mention[^\n]*\n)*",
            re.I,
        ),
        "You are a coding assistant.\n",
    ),
    (
        re.compile(
            r"You are an expert coding assistant in Loop,\s*created by Soket AI\.?",
            re.I,
        ),
        "You are an expert coding assistant.",
    ),
    (
        re.compile(
            r"operating inside pi,\s*a coding agent harness",
            re.I,
        ),
        "operating as a coding agent",
    ),
    (re.compile(r"You are minsu kang,?\s*", re.I), "You are "),
    (re.compile(r"You are dylanmsk,?\s*", re.I), "You are "),
    (re.compile(r"\bminsu kang\b", re.I), "the developer"),
    (re.compile(r"\bdylanmsk\b", re.I), "the developer"),
    (re.compile(r"DylanMsK", re.I), "the-developer"),
    (re.compile(r"planning specialist for Chakra", re.I), "planning specialist"),
    (re.compile(r"an open-source fork of Soket AI Labs\.?", re.I), ""),
    (re.compile(r"created by Soket AI\.?", re.I), ""),
    (re.compile(r"made by Soket AI Labs\.?", re.I), ""),
    (re.compile(r"Soket AI Labs", re.I), ""),
    (re.compile(r"Soket AI", re.I), ""),
    (re.compile(r"\bSoket\b", re.I), ""),
    (re.compile(r"\bChakra\b", re.I), ""),
    (re.compile(r"\bAnthropic\b", re.I), ""),
    (re.compile(r"\bClaude\b", re.I), ""),
    (re.compile(r"\bin Loop\b"), ""),
    (
        re.compile(r"Current working directory:\s*/\S+", re.I),
        "Current working directory: the workspace",
    ),
    (re.compile(r"/home/\S+"), "the workspace"),
    (re.compile(r"/tmp/synthetic/\S+"), "the workspace"),
]


def genericize_system_prompt(text: str) -> str:
    """Keep the session's own prompt, but drop harness / product / person names."""
    if not text or not str(text).strip():
        return ""
    out = str(text)
    for pattern, repl in _BRAND_PHRASES:
        out = pattern.sub(repl, out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r" +\n", "\n", out)
    out = re.sub(
        r"(You are a coding assistant\.\s*){2,}",
        "You are a coding assistant.\n",
        out,
        flags=re.I,
    )
    return out.strip()


def split_chat_input(chat_input: Any) -> tuple[str | None, str]:
    """Return (system_prompt, user_text) from dump chat_input.

    llm-logging often stores ``{"system": "...", "prompt": "...", "tools": [...]}``
    or a messages list. Plain strings are user-only (no captured system).
    ``tools`` are ignored here — use ``extract_openai_tools``.
    """
    if chat_input is None or chat_input == "":
        return None, ""
    chat_input = _unwrap_chat_input_obj(chat_input)
    if isinstance(chat_input, str):
        return None, chat_input
    if isinstance(chat_input, dict):
        system = chat_input.get("system")
        if isinstance(system, str) and system.strip():
            system_text = system.strip()
        else:
            system_text = None
        for key in ("prompt", "user", "query", "message", "content"):
            val = chat_input.get(key)
            if isinstance(val, str) and val.strip():
                return system_text, val
            if val is not None and key == "content" and not isinstance(val, (dict, list)):
                return system_text, str(val)
        # Avoid dumping the whole tools schema into the user turn
        leftover = {k: v for k, v in chat_input.items() if k not in {"system", "tools"}}
        if system_text and not leftover:
            return system_text, ""
        if leftover:
            return system_text, trunc(leftover)
        return system_text, ""
    if isinstance(chat_input, list):
        system_text = None
        user_parts: list[str] = []
        for msg in chat_input:
            if not isinstance(msg, dict):
                continue
            role = (msg.get("role") or "").lower()
            content = msg.get("content")
            text = content.strip() if isinstance(content, str) else (trunc(content) if content else "")
            if role == "system" and text and not system_text:
                system_text = text
            elif role in {"user", ""} and text:
                user_parts.append(text)
        return system_text, "\n\n".join(user_parts)
    return None, trunc(chat_input)


def normalize_trace_record(trace: dict, session_meta: dict | None = None) -> dict:
    """Unified trace dict from structured dump or Langfuse export."""
    tags = set(trace.get("tags") or [])
    if trace.get("tag"):
        tags.add(trace["tag"])

    chat_input = trace.get("chat_input")
    if chat_input is None and trace.get("input") is not None:
        chat_input = unwrap_json_string(trace["input"])

    chat_output = trace.get("chat_output")
    if chat_output is None and trace.get("output") is not None:
        chat_output = unwrap_json_string(trace["output"])

    observations = trace.get("observations") or []
    # Langfuse export nests under trace key
    return {
        "trace_id": trace.get("trace_id") or trace.get("id"),
        "session_id": (session_meta or {}).get("session_id") or trace.get("sessionId"),
        "tags": list(tags),
        "chat_input": chat_input,
        "chat_output": chat_output,
        "model": trace.get("model"),
        "observations": observations,
    }


def build_messages(
    trace: dict,
    *,
    include_system: bool = False,
    skip_empty_spans: bool = False,
    permissive: bool = False,
) -> list[dict]:
    obs = list(trace.get("observations") or [])
    if any(obs_time(o) for o in obs):
        obs_sorted = sorted(obs, key=lambda o: obs_time(o))
    else:
        obs_sorted = list(reversed(obs))

    captured_system, user_content = split_chat_input(trace.get("chat_input"))
    if not user_content:
        user_content = "" if permissive else "Complete the coding task."

    messages: list[dict] = []
    if include_system:
        messages.append(
            {
                "role": "system",
                "content": captured_system
                or (
                    "Cursor agent session (trace export). "
                    "No system prompt was captured in telemetry."
                ),
            }
        )
    messages.append({"role": "user", "content": user_content})

    pending_think = None
    call_i = 0
    for o in obs_sorted:
        typ = o.get("type")
        title = obs_title(o)
        title_base = title.split()[0] if title else ""
        if typ == "EVENT":
            continue
        # Thinking may be GENERATION or SPAN, often "agent-thinking 1/3"
        if title_base == "agent-thinking" or title.startswith("agent-thinking"):
            pending_think = o.get("output") or ""
            continue
        if typ == "GENERATION" and (
            title_base == "agent-response" or title.startswith("agent-response")
        ):
            resp = o.get("output") or ""
            if skip_empty_spans and not payload_nonempty(resp):
                continue
            msg: dict[str, Any] = {"role": "assistant", "content": resp}
            if pending_think:
                msg["reasoning_content"] = pending_think
                pending_think = None
            messages.append(msg)
            continue
        if typ == "SPAN":
            # Non-tool agent spans already handled above; skip residual noise
            if title.startswith("agent-"):
                continue
            if skip_empty_spans and not observation_has_io(o):
                continue
            name, args, tool_out = map_span_to_cursor_tool(o)
            if skip_empty_spans and (
                name == "unknown_tool" or not payload_nonempty(tool_out)
            ):
                continue
            call_i += 1
            tc_id = f"call_{call_i}"
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": pending_think or "",
                    "tool_calls": [
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args, ensure_ascii=False),
                            },
                        }
                    ],
                }
            )
            pending_think = None
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": tool_out if tool_out else ("" if permissive else "(empty observation)"),
                }
            )

    co = trace.get("chat_output")
    if isinstance(co, str) and co.strip():
        last = messages[-1] if messages else {}
        if last.get("content") != co:
            if last.get("role") == "assistant" and not last.get("tool_calls"):
                last["content"] = co
            elif last.get("role") == "tool" or (
                last.get("role") == "assistant" and last.get("tool_calls")
            ):
                messages.append({"role": "assistant", "content": co})

    return messages


def _session_from_dump_obj(data: dict) -> tuple[dict, list[dict]] | None:
    """llm-logging dump: {session_id, traces: [...]}."""
    if not isinstance(data, dict) or "traces" not in data:
        return None
    if "session_id" not in data and "id" not in data:
        return None
    if "dataset" in data:
        return None
    session_meta = {
        "session_id": data.get("session_id") or data.get("id"),
        "session_name": data.get("session_name"),
    }
    traces = [
        normalize_trace_record(t, session_meta) for t in (data.get("traces") or [])
    ]
    return session_meta, traces


def iter_traces_from_path(path: Path) -> Iterator[tuple[dict | None, dict]]:
    """Yield (session_meta, trace) from structured JSON or single Langfuse export."""
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                parsed = _session_from_dump_obj(data)
                if parsed:
                    session_meta, traces = parsed
                    for trace in traces:
                        yield session_meta, trace
                    continue
                yield None, normalize_trace_record(data)
        return

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    parsed = _session_from_dump_obj(data)
    if parsed:
        session_meta, traces = parsed
        for trace in traces:
            yield session_meta, trace
        return

    if "trace" in data and isinstance(data["trace"], dict):
        yield None, normalize_trace_record(data["trace"])
        return

    if "dataset" in data and "sessions" in data["dataset"]:
        for session in data["dataset"]["sessions"]:
            session_meta = {
                "session_id": session.get("session_id"),
                "session_name": session.get("session_name"),
            }
            for trace in session.get("traces") or []:
                yield session_meta, normalize_trace_record(trace, session_meta)
        return

    if "observations" in data or data.get("input") is not None:
        yield None, normalize_trace_record(data)
        return

    raise ValueError(f"Unrecognized trace JSON shape: {path}")


def iter_sessions_from_path(path: Path) -> Iterator[tuple[dict, list[dict]]]:
    """Yield (session_meta, traces[]) keeping multi-trace sessions together."""
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                parsed = _session_from_dump_obj(data)
                if parsed:
                    yield parsed
                    continue
                trace = normalize_trace_record(data)
                yield {
                    "session_id": trace.get("session_id") or trace.get("trace_id"),
                    "session_name": None,
                }, [trace]
        return

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    parsed = _session_from_dump_obj(data)
    if parsed:
        yield parsed
        return

    if "dataset" in data and "sessions" in data["dataset"]:
        for session in data["dataset"]["sessions"]:
            session_meta = {
                "session_id": session.get("session_id"),
                "session_name": session.get("session_name"),
            }
            traces = [
                normalize_trace_record(t, session_meta) for t in (session.get("traces") or [])
            ]
            yield session_meta, traces
        return

    # Fallback: treat each yielded trace as its own one-trace session
    for session_meta, trace in iter_traces_from_path(path):
        meta = session_meta or {
            "session_id": trace.get("session_id") or trace.get("trace_id"),
            "session_name": None,
        }
        yield meta, [trace]


def should_keep(trace: dict, *, agent_only: bool) -> bool:
    if not agent_only:
        return True
    tags = set(trace.get("tags") or [])
    return "agent" in tags and "tab" not in tags


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt Langfuse/Cursor traces to message JSONL")
    parser.add_argument("input", help="Trace JSON file or structured_full JSON")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--agent-only",
        action="store_true",
        help="Keep only agent-tagged traces (skip tab autocomplete)",
    )
    parser.add_argument(
        "--include-system",
        action="store_true",
        help="Add minimal Cursor-style system line (default: none)",
    )
    parser.add_argument(
        "--openai-sft",
        action="store_true",
        help="Emit daVinci/SWE native SFT shape: messages + tools[] (OpenAI function calling). Skip convert_trajectories XML step.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max trajectories (0 = all)")
    args = parser.parse_args()

    path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for session_meta, trace in iter_traces_from_path(path):
            if not should_keep(trace, agent_only=args.agent_only):
                continue
            msgs = build_messages(
                trace,
                include_system=args.include_system or args.openai_sft,
            )
            if args.openai_sft and msgs and msgs[0].get("role") == "system":
                msgs[0]["content"] = CURSOR_SFT_SYSTEM
            if len(msgs) <= 1:
                continue
            row: dict[str, Any] = {
                "messages": msgs,
            }
            if args.openai_sft:
                row["tools"] = CURSOR_OPENAI_TOOLS
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break

    print(f"Wrote {n} trajectories to {out_path}")


if __name__ == "__main__":
    main()
