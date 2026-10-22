#!/usr/bin/env python3
"""
HF ``harvardMadsys/freeinference_agentic_trace`` sessions → OpenAI SFT JSONL.

The release is structure-only: **no message text is published** (no system,
user, assistant or tool-result text). What *is* published per request:
trigger (user / tool), tool calls (name + anonymized arguments such as
``python <path> --status``), outcome, token counts and tool definition names.

So the output keeps the real trajectory shape — turn order, tool names,
argument keys/values-with-placeholders, parallel calls — and fills every
missing text field with a typed placeholder carrying its token count, e.g.
``<user_message tokens≈70>``. Rows are useful for format / tool-protocol
checks and as skeletons for re-hydration, not as-is for content SFT.

Mapping (one session → one row, same as ``sft_pipeline.py --by-session``):
  step_trigger=user  → role:user  (placeholder)
  each request       → role:assistant (+ tool_calls when the response had them)
  each tool call     → role:tool  (placeholder with outcome / result_tokens)

Example:
  python hf_freeinference_sft.py day.jsonl.zst -o _sample_final/hf_traces/openai_sft.jsonl \\
      --limit 2 --pretty _sample_final/hf_traces/openai_sft.pretty.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Iterator

from penant_sft import validate_openai_tool_order
from prompt_mutations import load_bank, pick_prompt
from sft_pipeline import validate_session_sft

SYSTEM_PLACEHOLDER = "<system_prompt tokens≈{n}>"


def iter_sessions(path: Path) -> Iterator[dict]:
    """Yield sessions from a ``.jsonl`` or ``.jsonl.zst`` trace file."""
    if path.suffix == ".zst":
        proc = subprocess.run(["zstd", "-dc", str(path)], check=True, capture_output=True)
        lines = proc.stdout.decode("utf-8").splitlines()
    else:
        lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if line.strip():
            yield json.loads(line)


def _args_json(arguments: Any) -> str:
    """HF args are ``[[key, value], ...]`` pairs → OpenAI JSON-string arguments."""
    if isinstance(arguments, list):
        out: dict[str, Any] = {}
        for pair in arguments:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                out[str(pair[0])] = pair[1]
        return json.dumps(out, ensure_ascii=False)
    if isinstance(arguments, dict):
        return json.dumps(arguments, ensure_ascii=False)
    return "{}"


def _tool_result(tc: dict) -> str:
    parts = [f"outcome={tc.get('outcome')}", f"tokens≈{tc.get('result_tokens')}"]
    sub = tc.get("subagent")
    if sub:
        parts.append(f"subagent_requests={len(sub.get('requests') or [])}")
    return f"<tool_result {' '.join(parts)}>"


def build_tools(session: dict) -> list[dict]:
    """OpenAI tools[] from definition names + argument keys seen in calls."""
    names: list[str] = []
    for r in session.get("requests") or []:
        for n in r.get("tool_definition_names") or []:
            if n != "other" and n not in names:
                names.append(n)
    props: dict[str, dict[str, dict]] = {n: {} for n in names}
    for r in session.get("requests") or []:
        for tc in r.get("tool_calls") or []:
            name = tc.get("name")
            if name not in props:
                names.append(name)
                props[name] = {}
            for key, val in json.loads(_args_json(tc.get("arguments"))).items():
                t = "string" if isinstance(val, str) else "number" if isinstance(val, (int, float)) else "object"
                if val == "<num>":
                    t = "number"
                props[name].setdefault(key, {"type": t})
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": "",
                "parameters": {"type": "object", "properties": props[n]},
            },
        }
        for n in names
    ]


def convert_session(session: dict, *, bank: list[str] | None = None) -> dict:
    requests = sorted(session.get("requests") or [], key=lambda r: r.get("step", 0))
    sys_tokens = ((requests[0].get("role_tokens") or {}).get("system") if requests else 0) or 0
    if bank:
        system, _ = pick_prompt(bank, index=0, seed=hash(session.get("session_id")) & 0xFFFFFFFF, mode="hash")
    else:
        system = SYSTEM_PLACEHOLDER.format(n=sys_tokens)

    messages: list[dict] = [{"role": "system", "content": system}]
    prev_user_tokens: int | None = None
    saw_parallel = False

    for r in requests:
        user_tokens = (r.get("role_tokens") or {}).get("user") or 0
        if r.get("step_trigger") == "user":
            delta = user_tokens - prev_user_tokens if prev_user_tokens is not None else None
            n = f"≈{delta}" if delta is not None and delta > 0 else "=?"
            messages.append({"role": "user", "content": f"<user_message tokens{n}>"})
        prev_user_tokens = user_tokens

        completion = (r.get("provider_tokens") or {}).get("completion")
        tcs = sorted(r.get("tool_calls") or [], key=lambda t: t.get("call_index", 0))
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": f"<assistant_message completion_tokens≈{completion}>",
        }
        if tcs:
            saw_parallel = saw_parallel or len(tcs) > 1
            msg["tool_calls"] = [
                {
                    "id": f"call_s{r.get('step')}_{tc.get('call_index', i)}",
                    "type": "function",
                    "function": {"name": tc.get("name"), "arguments": _args_json(tc.get("arguments"))},
                }
                for i, tc in enumerate(tcs)
            ]

        last = messages[-1]
        # Same as penant_sft: never emit back-to-back assistant text turns
        if last.get("role") == "assistant" and not last.get("tool_calls"):
            messages.pop()
        messages.append(msg)
        for tc, call in zip(tcs, msg.get("tool_calls") or []):
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": _tool_result(tc)})

    # SFT rows must end on an assistant text turn
    if messages[-1].get("role") == "tool":
        messages.append({"role": "assistant", "content": "<assistant_message completion_tokens=?>"})

    return {
        "messages": messages,
        "tools": build_tools(session),
        "parallel_tool_calls": saw_parallel,
        "session_id": session.get("session_id"),
        "source": "hf:harvardMadsys/freeinference_agentic_trace",
        "harness": session.get("harness"),
        "model": requests[0].get("model") if requests else None,
        "day": session.get("day"),
        "n_requests": len(requests),
        "text_redacted": True,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="HF traces file (.jsonl or .jsonl.zst)")
    p.add_argument("-o", "--output", type=Path, required=True, help="Output OpenAI SFT JSONL")
    p.add_argument("--session-ids", nargs="*", default=None, help="Only convert these session ids")
    p.add_argument("--limit", type=int, default=0, help="Max kept sessions (0 = all)")
    p.add_argument("--pretty", type=Path, default=None, help="Also write kept rows as pretty JSON list")
    p.add_argument("--report", type=Path, default=None)
    p.add_argument(
        "--system-bank",
        type=Path,
        default=None,
        help="Use a mutated prompt from this bank instead of the <system_prompt> placeholder",
    )
    args = p.parse_args()

    bank = load_bank(args.system_bank) if args.system_bank else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    details: list[dict] = []
    for session in iter_sessions(args.input):
        if args.session_ids and session.get("session_id") not in args.session_ids:
            continue
        row = convert_session(session, bank=bank)
        errs = validate_session_sft(row) + validate_openai_tool_order(row["messages"])
        details.append(
            {
                "session_id": row["session_id"],
                "harness": row["harness"],
                "n_requests": row["n_requests"],
                "n_messages": len(row["messages"]),
                "n_tools": len(row["tools"]),
                "validation_errors": errs,
            }
        )
        if not errs:
            rows.append(row)
        if args.limit and len(rows) >= args.limit:
            break

    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.pretty:
        args.pretty.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = {"input": str(args.input), "output": str(args.output), "kept": len(rows), "details": details}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
