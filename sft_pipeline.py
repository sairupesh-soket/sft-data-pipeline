#!/usr/bin/env python3
"""
Cursor / Langfuse traces → OpenAI SFT JSONL.

Reads a traces JSON (or a directory of them), converts each agent trajectory
into OpenAI chat SFT format (system / user / assistant / tool + tools[]),
drops samples with missing input/output on the fly, and validates the
message-role protocol before writing.

Example:
  python env_traj_utils/sft_pipeline.py path/to/structured_full_90d.json \\
      -o traces_openai_sft.jsonl --agent-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapt_langfuse_traces import (
    CURSOR_OPENAI_TOOLS,
    CURSOR_SFT_SYSTEM,
    build_messages,
    collect_session_tools,
    extract_openai_tools,
    genericize_system_prompt,
    iter_sessions_from_path,
    iter_traces_from_path,
    observation_has_io,
    payload_nonempty,
    should_keep,
    split_chat_input,
    unwrap_json_string,
)
from prompt_mutations import load_bank, pick_prompt

PROMPT_BANK_PATH = Path("/projects/data/datasets/traces_team/sft-data/prompt_bank.json")
_PROMPT_BANK: list[str] | None = None


def _mutated_fallback_prompt(session_id: str | None, index: int) -> str:
    global _PROMPT_BANK
    if _PROMPT_BANK is None:
        _PROMPT_BANK = load_bank(PROMPT_BANK_PATH)
    key = session_id or str(index)
    prompt, _ = pick_prompt(_PROMPT_BANK, index=index, seed=hash(key) & 0xFFFFFFFF, mode="hash")
    return prompt

ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.I,
)
THINKING_MARKERS = ('{"type": "thinking"}', '{"type":"thinking"}')
KNOWN_TOOLS = {t["function"]["name"] for t in CURSOR_OPENAI_TOOLS}


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return unwrap_json_string(value).strip() if isinstance(value, str) else str(value).strip()


def drop_reason_before_convert(trace: dict) -> str | None:
    """Reject traces that cannot become valid SFT (missing I/O, junk user, fragments)."""
    user = _as_text(trace.get("chat_input"))
    if not user:
        return "no_user_input"
    if any(user.startswith(m) for m in THINKING_MARKERS):
        return "thinking_fragment"
    if user.startswith("{") and "thinking" in user.lower():
        return "thinking_fragment"
    tid = str(trace.get("trace_id") or "")
    if tid and not UUID_RE.match(tid):
        return "child_span_id"

    obs = list(trace.get("observations") or [])
    spans = [o for o in obs if o.get("type") == "SPAN"]
    if spans and not any(observation_has_io(o) for o in spans):
        return "no_span_input_output"

    has_response = payload_nonempty(trace.get("chat_output")) or any(
        o.get("type") == "GENERATION"
        and (o.get("name") or o.get("title")) == "agent-response"
        and payload_nonempty(o.get("output"))
        for o in obs
    )
    if not has_response and not any(observation_has_io(o) for o in spans):
        return "no_input_output"
    return None


def validate_basic_sft(row: dict) -> list[str]:
    """Role-order validation only; empty content is allowed."""
    errors: list[str] = []
    msgs = row.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return ["need at least system + user messages"]

    if msgs[0].get("role") != "system":
        errors.append("messages[0] must be system")
    if msgs[1].get("role") != "user":
        errors.append("messages[1] must be user")

    pending: list[str] = []
    saw_assistant = False
    for i, msg in enumerate(msgs):
        role = msg.get("role")
        if role not in ALLOWED_ROLES:
            errors.append(f"messages[{i}] invalid role {role!r}")
            continue
        if role == "system" and i != 0:
            errors.append(f"messages[{i}] system must only appear first")
        if role == "user" and i != 1:
            errors.append(f"messages[{i}] user must only appear once at start")
        if role == "assistant":
            saw_assistant = True
            tcs = msg.get("tool_calls") or []
            if tcs:
                if pending:
                    errors.append(f"messages[{i}] tool_calls before prior tool results")
                for tc in tcs:
                    tc_id = tc.get("id")
                    if not tc_id:
                        errors.append(f"messages[{i}] tool_call missing id")
                    elif tc.get("type") != "function":
                        errors.append(f"messages[{i}] tool_call type must be function")
                    else:
                        pending.append(tc_id)
            elif pending:
                errors.append(f"messages[{i}] assistant turn before tool results returned")
        if role == "tool":
            if i == 0 or msgs[i - 1].get("role") != "assistant":
                errors.append(f"messages[{i}] tool must immediately follow assistant")
            if not pending:
                errors.append(f"messages[{i}] tool without pending tool_calls")
            else:
                tid = msg.get("tool_call_id")
                if tid not in pending:
                    errors.append(f"messages[{i}] tool_call_id mismatch")
                else:
                    pending.remove(tid)

    if len(msgs) > 2 and not saw_assistant:
        errors.append("conversation after user must include assistant turns")

    return errors


def validate_openai_sft(row: dict) -> list[str]:
    """Return error strings if the row is not valid OpenAI tool-calling SFT."""
    errors: list[str] = []
    msgs = row.get("messages")
    tools = row.get("tools")

    if not isinstance(msgs, list) or len(msgs) < 3:
        return ["messages must be a list of at least system, user, assistant"]
    if not isinstance(tools, list) or not tools:
        errors.append("missing top-level tools[]")
    else:
        for i, tool in enumerate(tools):
            fn = (tool or {}).get("function") or {}
            if tool.get("type") != "function" or not fn.get("name") or not fn.get("parameters"):
                errors.append(f"tools[{i}] is not a complete OpenAI function schema")

    if msgs[0].get("role") != "system" or not _as_text(msgs[0].get("content")):
        errors.append("messages[0] must be a non-empty system turn")
    if msgs[1].get("role") != "user" or not _as_text(msgs[1].get("content")):
        errors.append("messages[1] must be a non-empty user turn")

    pending: list[str] = []
    for i, msg in enumerate(msgs):
        role = msg.get("role")
        if role not in ALLOWED_ROLES:
            errors.append(f"messages[{i}] invalid role {role!r}")
            continue
        if role == "system" and i != 0:
            errors.append(f"messages[{i}] extra system (only first turn may be system)")
        if role == "user" and i != 1:
            errors.append(
                f"messages[{i}] extra user — tool results must use role=tool, not user"
            )
        if role == "assistant":
            tcs = msg.get("tool_calls") or []
            if tcs:
                if pending:
                    errors.append(f"messages[{i}] new tool_calls before previous tools resolved")
                for tc in tcs:
                    if tc.get("type") != "function":
                        errors.append(f"messages[{i}] tool_call type must be function")
                    tc_id = tc.get("id")
                    if not tc_id:
                        errors.append(f"messages[{i}] tool_call missing id")
                    fn = tc.get("function") or {}
                    name = fn.get("name")
                    if not name:
                        errors.append(f"messages[{i}] tool_call missing function.name")
                    elif name not in KNOWN_TOOLS:
                        errors.append(f"messages[{i}] unknown tool {name!r}")
                    args = fn.get("arguments")
                    if not isinstance(args, str):
                        errors.append(f"messages[{i}] function.arguments must be a JSON string")
                    else:
                        try:
                            json.loads(args)
                        except json.JSONDecodeError:
                            errors.append(f"messages[{i}] function.arguments is not valid JSON")
                    if tc_id:
                        pending.append(tc_id)
            else:
                if pending:
                    errors.append(f"messages[{i}] assistant text while tool results still pending")
                if i == len(msgs) - 1 and not _as_text(msg.get("content")):
                    errors.append("final assistant content is empty")
        if role == "tool":
            if not pending:
                errors.append(f"messages[{i}] role=tool with no pending assistant tool_calls")
                continue
            tid = msg.get("tool_call_id")
            if tid not in pending:
                errors.append(f"messages[{i}] tool_call_id {tid!r} does not match pending calls")
            else:
                pending.remove(tid)
            content = msg.get("content")
            if not _as_text(content) or content == "(empty observation)":
                errors.append(f"messages[{i}] empty tool observation")

    if pending:
        errors.append(f"unresolved tool_call_id(s): {pending}")

    last = msgs[-1]
    if last.get("role") != "assistant" or last.get("tool_calls"):
        errors.append("conversation must end with an assistant text turn (no trailing tool_calls)")

    return errors


def iter_input_paths(src: Path) -> Iterator[Path]:
    if src.is_file():
        yield src
        return
    if src.is_dir():
        session_files = sorted(src.glob("session-*.json"))
        if session_files:
            yield from session_files
            return
        files = [
            p
            for p in sorted(src.glob("*.json")) + sorted(src.glob("*.jsonl"))
            if p.name != "manifest.json"
        ]
        if not files:
            raise FileNotFoundError(f"No .json / .jsonl files in {src}")
        yield from files
        return
    raise FileNotFoundError(src)


def convert_trace(trace: dict, *, permissive: bool = False) -> dict[str, Any]:
    msgs = build_messages(
        trace,
        include_system=True,
        skip_empty_spans=not permissive,
        permissive=permissive,
    )
    if msgs and msgs[0].get("role") == "system":
        captured, _ = split_chat_input(trace.get("chat_input"))
        if captured:
            msgs[0]["content"] = genericize_system_prompt(captured)
        else:
            msgs[0]["content"] = CURSOR_SFT_SYSTEM
    tools = extract_openai_tools(trace.get("chat_input")) or CURSOR_OPENAI_TOOLS
    return {
        "messages": msgs,
        "tools": tools,
        "parallel_tool_calls": False,
    }


def collapse_consecutive_empty_users(messages: list[dict]) -> list[dict]:
    """Drop runs of 2+ empty USER turns; keep a solitary empty USER."""
    out: list[dict] = []
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]
        is_empty_user = (
            msg.get("role") == "user" and not str(msg.get("content") or "").strip()
        )
        if not is_empty_user:
            out.append(msg)
            i += 1
            continue
        j = i + 1
        while j < n:
            nxt = messages[j]
            if nxt.get("role") == "user" and not str(nxt.get("content") or "").strip():
                j += 1
            else:
                break
        streak = j - i
        if streak == 1:
            out.append(msg)
        # streak >= 2: drop the whole empty-USER run
        i = j
    return out


def convert_session(
    traces: list[dict],
    *,
    session_meta: dict | None = None,
    permissive: bool = False,
) -> dict[str, Any]:
    """One SFT row per session: single system prompt + all traces in order."""
    session_system = _pick_session_system_prompt(traces)
    if not session_system:
        # Only fall back to mutated bank when the dump has no system at all
        sid = (session_meta or {}).get("session_id")
        session_system = _mutated_fallback_prompt(sid, 0)
    messages: list[dict] = [{"role": "system", "content": session_system}]
    trace_ids: list[str] = []
    for trace in traces:
        # Skip dedicated system-prompt traces in the dialogue body
        name = (trace.get("trace_name") or "").lower()
        if "system prompt" in name:
            continue
        captured, user = split_chat_input(trace.get("chat_input"))
        # Skip input-only system bags with no user turn
        if not user.strip() and captured and not (trace.get("observations") or []):
            continue
        trace_ids.append(str(trace.get("trace_id") or ""))
        part = build_messages(
            trace,
            include_system=False,
            skip_empty_spans=not permissive,
            permissive=permissive,
        )
        messages.extend(part)

    messages = collapse_consecutive_empty_users(messages)

    tools = collect_session_tools(traces) or CURSOR_OPENAI_TOOLS

    row: dict[str, Any] = {
        "messages": messages,
        "tools": tools,
        "parallel_tool_calls": False,
        "n_traces": len(trace_ids),
        "trace_ids": [t for t in trace_ids if t],
    }
    if session_meta:
        if session_meta.get("session_id"):
            row["session_id"] = session_meta["session_id"]
        if session_meta.get("session_name"):
            row["session_name"] = session_meta["session_name"]
    return row


def _pick_session_system_prompt(traces: list[dict]) -> str | None:
    """Prefer the session's own first system prompt; never invent one here."""
    scored: list[tuple[int, str, str]] = []
    for trace in traces:
        name = (trace.get("trace_name") or "").lower()
        captured, _ = split_chat_input(trace.get("chat_input"))
        if not captured or not str(captured).strip():
            continue
        # Prefer explicit "system prompt" traces, then earliest timestamp
        prefer = 0 if "system prompt" in name else 1
        ts = str(trace.get("timestamp") or "")
        scored.append((prefer, ts, str(captured).strip()))
    if not scored:
        return None
    scored.sort(key=lambda x: (x[0], x[1]))
    return genericize_system_prompt(scored[0][2]) or scored[0][2]


def validate_session_sft(row: dict) -> list[str]:
    """Minimal checks for multi-trace session rows (multiple user turns allowed)."""
    errors: list[str] = []
    msgs = row.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return ["need at least system + one turn"]
    if msgs[0].get("role") != "system":
        errors.append("messages[0] must be system")
    if not any(m.get("role") == "user" for m in msgs[1:]):
        # Allow empty-user traces in permissive exports; still require a user slot
        if not any(m.get("role") in {"user", "assistant", "tool"} for m in msgs[1:]):
            errors.append("session has no user/assistant/tool turns")
    return errors


def run_pipeline(
    src: Path,
    out_path: Path,
    *,
    agent_only: bool,
    permissive: bool,
    limit: int,
    dropped_path: Path | None,
    report_path: Path | None,
    by_session: bool = False,
) -> dict[str, Any]:
    stats: Counter[str] = Counter()
    written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    drop_f = dropped_path.open("w", encoding="utf-8") if dropped_path else None
    try:
        with out_path.open("w", encoding="utf-8") as fout:
            for path in iter_input_paths(src):
                if by_session:
                    for session_meta, traces in iter_sessions_from_path(path):
                        stats["sessions_seen"] += 1
                        kept_traces: list[dict] = []
                        for trace in traces:
                            stats["traces_seen"] += 1
                            if not should_keep(trace, agent_only=agent_only):
                                stats["skip_not_agent"] += 1
                                continue
                            if not permissive:
                                reason = drop_reason_before_convert(trace)
                                if reason:
                                    stats[f"drop_{reason}"] += 1
                                    continue
                            kept_traces.append(trace)
                        if not kept_traces:
                            stats["drop_empty_session"] += 1
                            if drop_f:
                                drop_f.write(
                                    json.dumps(
                                        {
                                            "session_id": (session_meta or {}).get("session_id"),
                                            "reason": "empty_session",
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                            continue

                        row = convert_session(
                            kept_traces,
                            session_meta=session_meta,
                            permissive=permissive,
                        )
                        errors = (
                            validate_session_sft(row)
                            if permissive
                            else validate_session_sft(row)  # multi-user sessions
                        )
                        if errors:
                            stats["drop_invalid_sft"] += 1
                            if drop_f:
                                drop_f.write(
                                    json.dumps(
                                        {
                                            "session_id": row.get("session_id"),
                                            "reason": "invalid_sft",
                                            "errors": errors,
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                            continue

                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                        written += 1
                        stats["kept"] += 1
                        stats["traces_in_kept_sessions"] += len(kept_traces)
                        if limit and written >= limit:
                            break
                else:
                    for _session, trace in iter_traces_from_path(path):
                        stats["seen"] += 1
                        if not should_keep(trace, agent_only=agent_only):
                            stats["skip_not_agent"] += 1
                            continue
                        if not permissive:
                            reason = drop_reason_before_convert(trace)
                            if reason:
                                stats[f"drop_{reason}"] += 1
                                if drop_f:
                                    drop_f.write(
                                        json.dumps(
                                            {
                                                "trace_id": trace.get("trace_id"),
                                                "model": trace.get("model"),
                                                "reason": reason,
                                            },
                                            ensure_ascii=False,
                                        )
                                        + "\n"
                                    )
                                continue

                        row = convert_trace(trace, permissive=permissive)
                        if permissive:
                            errors = validate_basic_sft(row)
                        else:
                            errors = validate_openai_sft(row)
                        if errors:
                            stats["drop_invalid_sft"] += 1
                            if drop_f:
                                drop_f.write(
                                    json.dumps(
                                        {
                                            "trace_id": trace.get("trace_id"),
                                            "model": row.get("model"),
                                            "reason": "invalid_sft",
                                            "errors": errors,
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                            continue

                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                        written += 1
                        stats["kept"] += 1
                        if limit and written >= limit:
                            break
                if limit and written >= limit:
                    break
    finally:
        if drop_f:
            drop_f.close()

    report = {
        "input": str(src),
        "output": str(out_path),
        "kept": written,
        "by_session": by_session,
        "counts": dict(stats),
    }
    if report_path:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Langfuse/Cursor traces to validated OpenAI SFT JSONL"
    )
    parser.add_argument("input", help="Traces JSON file, or a directory of JSON files")
    parser.add_argument("-o", "--output", required=True, help="Output OpenAI SFT JSONL")
    parser.add_argument(
        "--agent-only",
        action="store_true",
        help="Keep only agent-tagged traces (skip tab autocomplete)",
    )
    parser.add_argument(
        "--permissive",
        action="store_true",
        help="Convert all traces; allow empty fields; apply basic system/user/assistant/tool role validation only",
    )
    parser.add_argument(
        "--by-session",
        action="store_true",
        help="One output row per session (all traces concatenated for continuity)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max kept samples (0 = all)")
    parser.add_argument(
        "--dropped",
        default="",
        help="Optional JSONL of dropped sample ids + reasons",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Optional JSON summary of keep/drop counts",
    )
    args = parser.parse_args()

    src = Path(args.input)
    report = run_pipeline(
        src,
        Path(args.output),
        agent_only=args.agent_only,
        permissive=args.permissive,
        limit=args.limit,
        dropped_path=Path(args.dropped) if args.dropped else None,
        report_path=Path(args.report) if args.report else None,
        by_session=args.by_session,
    )
    print(json.dumps(report, indent=2))
    if report["kept"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
