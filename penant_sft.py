#!/usr/bin/env python3
"""
Penant-data (gold-loan / multi-stage agent) → OpenAI SFT JSONL.

Unlike the Cursor IDE adapter, penant GENERATION ``agent-response`` outputs
already contain OpenAI chat messages (``content`` / ``tool_calls``). We use
those as the source of truth and attach tool SPAN results as ``role: tool``.

Correct order:
  user → assistant(tool_calls) → tool(+tool…) → assistant(text|tool_calls) → …

Traces are sorted chronologically. System prompt comes from the session's own
``* system prompt`` traces (not the mutated coding bank).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from adapt_langfuse_traces import (
    collect_session_tools,
    extract_openai_tools,
    split_chat_input,
)


STEP_RE = re.compile(r"^(?P<base>agent-(?:thinking|response))(?:\s+(?P<i>\d+)/(?P<n>\d+))?$")


def _obs_start(o: dict) -> str:
    md = o.get("metadata") or {}
    return str(md.get("start_time") or o.get("startTime") or o.get("start_time") or "")


def _parse_jsonish(val: Any) -> Any:
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return val
    return val


def _tool_content(out: Any) -> str:
    if out is None:
        return ""
    if isinstance(out, str):
        return out
    return json.dumps(out, ensure_ascii=False)


def _assistant_from_generation(out: Any) -> dict | None:
    """Parse agent-response GENERATION output into an OpenAI assistant message."""
    out = _parse_jsonish(out)
    if not isinstance(out, dict):
        if isinstance(out, str) and out.strip():
            return {"role": "assistant", "content": out}
        return None
    # Already OpenAI-shaped
    if out.get("role") == "assistant" or "tool_calls" in out or "content" in out:
        msg: dict[str, Any] = {"role": "assistant"}
        content = out.get("content")
        msg["content"] = "" if content is None else content
        if out.get("tool_calls"):
            # Normalize tool_calls to OpenAI chat form
            tcs = []
            for tc in out["tool_calls"]:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, dict):
                    args = json.dumps(args, ensure_ascii=False)
                elif args is None:
                    args = "{}"
                tcs.append(
                    {
                        "id": tc.get("id") or f"call_{len(tcs)+1}",
                        "type": tc.get("type") or "function",
                        "function": {
                            "name": fn.get("name") or "unknown_tool",
                            "arguments": args if isinstance(args, str) else str(args),
                        },
                    }
                )
            if tcs:
                msg["tool_calls"] = tcs
                if msg["content"] is None:
                    msg["content"] = ""
        return msg
    return None


def build_trace_messages(trace: dict) -> list[dict]:
    """Build OpenAI messages for one penant agent trace (no system)."""
    _, user = split_chat_input(trace.get("chat_input"))
    messages: list[dict] = [{"role": "user", "content": user or ""}]

    obs = sorted(list(trace.get("observations") or []), key=_obs_start)
    pending_think: str | None = None
    all_tool_spans = sorted(
        [
            o
            for o in obs
            if o.get("type") == "SPAN"
            and str(o.get("title") or o.get("name") or "").startswith("tool:")
        ],
        key=_obs_start,
    )
    tool_span_i = 0

    for o in obs:
        typ = o.get("type")
        title = (o.get("title") or o.get("name") or "").strip()
        if typ == "EVENT":
            continue
        if title.startswith("agent-thinking"):
            think = o.get("output")
            pending_think = think if isinstance(think, str) else _tool_content(think)
            continue
        if typ == "GENERATION" and title.startswith("agent-response"):
            msg = _assistant_from_generation(o.get("output"))
            if not msg:
                continue
            if pending_think:
                msg["reasoning_content"] = pending_think
                pending_think = None

            last = messages[-1] if messages else None
            # Collapse consecutive assistant text turns (e.g. response 4/5 then 5/5)
            if (
                not msg.get("tool_calls")
                and last
                and last.get("role") == "assistant"
                and not last.get("tool_calls")
            ):
                if msg.get("reasoning_content"):
                    prev_r = last.get("reasoning_content") or ""
                    last["reasoning_content"] = (
                        (prev_r + "\n" + msg["reasoning_content"]).strip()
                        if prev_r
                        else msg["reasoning_content"]
                    )
                last["content"] = msg.get("content") or last.get("content") or ""
                continue

            # OpenAI: merge preceding assistant text into the tool-call turn
            if (
                msg.get("tool_calls")
                and last
                and last.get("role") == "assistant"
                and not last.get("tool_calls")
            ):
                prev = messages.pop()
                text = (prev.get("content") or "").strip()
                if text and not (msg.get("content") or "").strip():
                    msg["content"] = text
                elif text:
                    msg["content"] = text + "\n" + (msg.get("content") or "")
                prev_r = prev.get("reasoning_content") or ""
                if prev_r:
                    cur_r = msg.get("reasoning_content") or ""
                    msg["reasoning_content"] = (
                        (prev_r + "\n" + cur_r).strip() if cur_r else prev_r
                    )

            messages.append(msg)
            for tc in msg.get("tool_calls") or []:
                want = (tc.get("function") or {}).get("name")
                span = None
                while tool_span_i < len(all_tool_spans):
                    cand = all_tool_spans[tool_span_i]
                    tool_span_i += 1
                    ctitle = cand.get("title") or cand.get("name") or ""
                    cname = (
                        ctitle.split(":", 1)[1].strip()
                        if ctitle.startswith("tool:")
                        else ""
                    )
                    if cname == want:
                        span = cand
                        break
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id"),
                        "content": _tool_content(span.get("output") if span else None),
                    }
                )
            continue

    # Prefer chat_output as final assistant text when present
    co = _parse_jsonish(trace.get("chat_output"))
    final_text = None
    if isinstance(co, dict):
        final_text = co.get("text") or co.get("content")
    elif isinstance(co, str) and co.strip():
        final_text = co
    if final_text:
        last = messages[-1] if messages else {}
        if last.get("role") == "assistant" and not last.get("tool_calls"):
            last["content"] = final_text
        elif last.get("role") == "tool":
            messages.append({"role": "assistant", "content": final_text})

    return messages


def _pick_system(traces: list[dict]) -> str:
    scored: list[tuple[int, str, str]] = []
    for t in traces:
        name = (t.get("trace_name") or "").lower()
        captured, _ = split_chat_input(t.get("chat_input"))
        if not captured:
            continue
        prefer = 0 if "system prompt" in name else 1
        scored.append((prefer, str(t.get("timestamp") or ""), captured.strip()))
    if not scored:
        return "You are a helpful assistant."
    scored.sort(key=lambda x: (x[0], x[1]))
    return scored[0][2]


def convert_penant_session(session: dict) -> dict:
    traces = list(session.get("traces") or [])
    system = _pick_system(traces)

    agent_traces = [
        t
        for t in traces
        if "system prompt" not in (t.get("trace_name") or "").lower()
    ]
    agent_traces.sort(key=lambda t: str(t.get("timestamp") or ""))

    messages: list[dict] = [{"role": "system", "content": system}]
    trace_ids: list[str] = []
    saw_parallel = False
    for t in agent_traces:
        part = build_trace_messages(t)
        for m in part:
            if m.get("role") == "assistant" and len(m.get("tool_calls") or []) > 1:
                saw_parallel = True
        messages.extend(part)
        if t.get("trace_id"):
            trace_ids.append(str(t["trace_id"]))

    tools = collect_session_tools(traces)
    # also union tools from any chat_input
    if not tools:
        for t in traces:
            for tool in extract_openai_tools(t.get("chat_input")):
                tools.append(tool)

    return {
        "messages": messages,
        "tools": tools,
        "parallel_tool_calls": saw_parallel,
        "n_traces": len(agent_traces),
        "trace_ids": trace_ids,
        "session_id": session.get("session_id"),
        "session_name": session.get("session_name"),
        "source": "penant-data",
    }


def validate_openai_tool_order(messages: list[dict]) -> list[str]:
    errs: list[str] = []
    pending: list[str] = []
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                if pending:
                    errs.append(f"messages[{i}] new tool_calls while {len(pending)} pending")
                pending = [tc.get("id") for tc in tcs]
            else:
                if pending:
                    errs.append(f"messages[{i}] text assistant while tool results pending")
                pending = []
        elif role == "tool":
            if not pending:
                errs.append(f"messages[{i}] tool with no pending tool_calls")
            else:
                tcid = m.get("tool_call_id")
                if tcid in pending:
                    pending.remove(tcid)
                else:
                    # allow order-flexible within parallel batch
                    pending.pop(0)
        elif role == "user":
            if pending:
                errs.append(f"messages[{i}] user while tool results pending")
            pending = []
        # asst->asst with no tool_calls between text assistants is OK for multi-turn,
        # but asst(text)->asst(text) back-to-back after user is unusual — flag soft
    if pending:
        errs.append(f"ended with {len(pending)} unresolved tool_calls")
    # hard fail: any consecutive assistant messages (should be merged or have tools between)
    for i in range(1, len(messages)):
        a, b = messages[i - 1], messages[i]
        if a.get("role") == "assistant" and b.get("role") == "assistant":
            errs.append(
                f"messages[{i-1}->{i}] consecutive assistant "
                f"(tc={bool(a.get('tool_calls'))}->{bool(b.get('tool_calls'))})"
            )
    return errs


def _iter_session_paths(src: Path) -> list[Path]:
    if src.is_file():
        return [src]
    if src.is_dir():
        files = sorted(src.glob("session-*.json"))
        if not files:
            files = sorted(src.glob("*.json"))
        return files
    raise FileNotFoundError(src)


def convert_one(path: Path) -> tuple[dict | None, dict]:
    session = json.loads(path.read_text(encoding="utf-8"))
    row = convert_penant_session(session)
    errs = validate_openai_tool_order(row["messages"])
    meta = {
        "input": str(path),
        "session_id": row.get("session_id"),
        "n_messages": len(row["messages"]),
        "n_tools": len(row.get("tools") or []),
        "parallel_tool_calls": row.get("parallel_tool_calls"),
        "n_traces": row.get("n_traces"),
        "validation_errors": errs,
        "ok": not errs,
    }
    if errs:
        return None, meta
    return row, meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "input",
        type=Path,
        help="Enriched penant session JSON, or a directory of session-*.json",
    )
    p.add_argument("-o", "--output", type=Path, required=True, help="Output JSONL")
    p.add_argument("--pretty", type=Path, default=None, help="Pretty JSON (single-file mode only)")
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--dropped", type=Path, default=None, help="JSONL of failed conversions")
    p.add_argument(
        "--keep-invalid",
        action="store_true",
        help="Write rows even if validation fails (still logged)",
    )
    args = p.parse_args()

    paths = _iter_session_paths(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0
    details: list[dict] = []
    drop_f = args.dropped.open("w", encoding="utf-8") if args.dropped else None

    with args.output.open("w", encoding="utf-8") as fout:
        for i, path in enumerate(paths, 1):
            try:
                row, meta = convert_one(path)
            except Exception as e:
                meta = {
                    "input": str(path),
                    "ok": False,
                    "validation_errors": [f"exception: {e}"],
                }
                row = None
            details.append(meta)
            if row is None or (meta.get("validation_errors") and not args.keep_invalid):
                dropped += 1
                if drop_f:
                    drop_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            else:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
                if args.pretty and len(paths) == 1:
                    args.pretty.write_text(
                        json.dumps(row, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
            if i % 50 == 0 or i == len(paths):
                print(f"  [{i}/{len(paths)}] kept={kept} dropped={dropped}", flush=True)

    if drop_f:
        drop_f.close()

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "sessions_seen": len(paths),
        "kept": kept,
        "dropped": dropped,
        "dropped_path": str(args.dropped) if args.dropped else None,
        "format": "openai_sft_penant (assistant.tool_calls -> role:tool; session system+tools)",
    }
    # include a few failure samples
    fails = [d for d in details if not d.get("ok")][:20]
    report["failure_samples"] = fails
    print(json.dumps({k: v for k, v in report.items() if k != "failure_samples"}, indent=2))
    if fails:
        print(f"failures: {len([d for d in details if not d.get('ok')])} (showing up to 20)")
        for f in fails[:5]:
            print(" ", f.get("session_id") or f.get("input"), f.get("validation_errors", [])[:3])
    if args.report:
        report["details"] = details
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
