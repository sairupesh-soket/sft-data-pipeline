#!/usr/bin/env python3
"""
Convert pipeline_v2 / github-issues raw traces → OpenAI SFT JSONL.

Raw shape (per line):
  {traceId, task, repo, observations: [{thought, tool_call, observation}, ...]}

OpenAI SFT shape:
  system / user / assistant(tool_calls + reasoning_content) / tool / ...

``thought`` (and any think/reasoning text) goes in ``reasoning_content``,
NOT in assistant ``content``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# Minimal OpenAI tool schemas for coding-agent tools used in this pipeline
TOOL_SCHEMAS: dict[str, dict] = {
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    "write_file": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a new file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    "edit_file": {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Apply a patch/edit to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "patch": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    "search_code": {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search code in the workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    "run_bash": {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    "run_tests": {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run tests",
            "parameters": {
                "type": "object",
                "properties": {"tests": {"type": "string"}},
            },
        },
    },
    "print_tree": {
        "type": "function",
        "function": {
            "name": "print_tree",
            "description": "Print a directory tree",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "depth": {"type": "integer"},
                },
            },
        },
    },
}

SYSTEM_MESSAGE = (
    "You are an AI coding agent with access to tools for reading, writing, "
    "editing, and searching code, and for running tests and shell commands."
)

THINK_TAG_RE = re.compile(
    r"<\s*(think|thinking|reasoning)\s*>(.*?)<\s*/\s*\1\s*>",
    re.I | re.S,
)


def _infer_name_from_args(args: Any) -> str:
    if not isinstance(args, dict):
        return "unknown"
    keys = set(args.keys())
    if "patch" in keys:
        return "edit_file"
    if "content" in keys:
        return "write_file"
    if "query" in keys:
        return "search_code"
    if "tests" in keys:
        return "run_tests"
    if "command" in keys:
        return "run_bash"
    if "depth" in keys:
        return "print_tree"
    if "path" in keys:
        return "read_file"
    return "unknown"


def _parse_tool_call(tc: str) -> tuple[str, dict]:
    tc = (tc or "").strip()
    if not tc:
        return "unknown", {}
    try:
        args = json.loads(tc)
        if isinstance(args, dict):
            return _infer_name_from_args(args), args
    except Exception:
        pass
    paren = tc.find("(")
    if paren != -1 and tc.endswith(")"):
        name = tc[:paren].strip()
        raw_args = tc[paren + 1 : -1]
        try:
            args = json.loads(raw_args) if raw_args else {}
        except Exception:
            args = {"raw": raw_args} if raw_args else {}
        if not name:
            name = _infer_name_from_args(args)
        return name, args if isinstance(args, dict) else {"raw": args}
    return "unknown", {"raw": tc}


def _split_reasoning(text: str) -> tuple[str, str]:
    """Return (reasoning, remaining_content)."""
    if not text:
        return "", ""
    parts = []
    rest = text
    for m in THINK_TAG_RE.finditer(text):
        parts.append(m.group(2).strip())
    if parts:
        rest = THINK_TAG_RE.sub("", text).strip()
        return "\n\n".join(parts), rest
    # bare thought strings from the pipeline go entirely to reasoning
    return text.strip(), ""


def convert_trace(trace: dict) -> dict | None:
    observations = trace.get("observations") or []
    if not observations:
        return None

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": trace.get("task") or ""},
    ]
    tool_names: set[str] = set()

    for i, step in enumerate(observations):
        thought = (step.get("thought") or "").strip()
        reasoning, leftover = _split_reasoning(thought) if thought else ("", "")
        name, args = _parse_tool_call(step.get("tool_call") or "")
        tool_names.add(name)
        tc_id = f"call_{i}"
        asst: dict[str, Any] = {
            "role": "assistant",
            "content": leftover if leftover else "",
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
        if reasoning:
            asst["reasoning_content"] = reasoning
        messages.append(asst)

        obs_text = step.get("observation") or ""
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc_id,
                "content": obs_text
                if obs_text
                else "(no observation recorded for this trace)",
            }
        )

    tools = [TOOL_SCHEMAS[n] for n in sorted(tool_names) if n in TOOL_SCHEMAS]
    # fallback stub schemas for unknowns
    for n in sorted(tool_names):
        if n not in TOOL_SCHEMAS and n != "unknown":
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": n,
                        "description": n,
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            )

    return {
        "messages": messages,
        "tools": tools,
        "parallel_tool_calls": False,
        "session_id": None,
        "trace_id": trace.get("traceId") or "",
        "repo": trace.get("repo"),
        "verified": bool(trace.get("verified")),
        "source": "github-issues-pipeline",
        "_source_family": (trace.get("taxonomy") or {}).get("family", "patch_reconstruction"),
    }


def validate(messages: list[dict]) -> list[str]:
    errs: list[str] = []
    if not messages or messages[0].get("role") != "system":
        errs.append("missing system")
    pending: list[str] = []
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "assistant":
            # reasoning must not live in content as <think> tags
            c = m.get("content") or ""
            if isinstance(c, str) and THINK_TAG_RE.search(c):
                errs.append(f"messages[{i}] think tags still in content")
            tcs = m.get("tool_calls") or []
            if tcs:
                if pending:
                    errs.append(f"messages[{i}] tool_calls while pending")
                pending = [tc.get("id") for tc in tcs]
            else:
                pending = []
        elif role == "tool":
            if not pending:
                errs.append(f"messages[{i}] orphan tool")
            else:
                tcid = m.get("tool_call_id")
                if tcid in pending:
                    pending.remove(tcid)
                else:
                    pending.pop(0)
        elif role == "user":
            pending = []
    if pending:
        errs.append(f"unresolved tool_calls {pending}")
    for i in range(1, len(messages)):
        if messages[i - 1].get("role") == "assistant" and messages[i].get("role") == "assistant":
            errs.append(f"messages[{i-1}->{i}] consecutive assistant")
    return errs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw_jsonl", type=Path, help="pipeline raw traces JSONL")
    p.add_argument(
        "--session-meta",
        type=Path,
        default=None,
        help="Langfuse session index JSON; only keep traces named sft:<id> in it",
    )
    p.add_argument("--limit", type=int, default=3, help="Max traces to convert")
    p.add_argument("--prefer-thought", action="store_true", default=True)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--pretty-dir", type=Path, default=None)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args()

    allow: set[str] | None = None
    if args.session_meta and args.session_meta.exists():
        meta = json.loads(args.session_meta.read_text(encoding="utf-8"))
        allow = set()
        for t in meta.get("traces") or []:
            n = t.get("name") or ""
            if n.startswith("sft:"):
                allow.add(n[4:])

    candidates: list[dict] = []
    with args.raw_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            tid = row.get("traceId") or ""
            if allow is not None and tid not in allow:
                continue
            obs = row.get("observations") or []
            if not obs:
                continue
            th = sum(1 for o in obs if (o.get("thought") or "").strip())
            tc = sum(1 for o in obs if (o.get("tool_call") or "").strip())
            candidates.append((th, tc, len(obs), int(bool(row.get("verified"))), row))

    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
    picked = [c[-1] for c in candidates[: args.limit]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.pretty_dir:
        args.pretty_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    details = []
    with args.output.open("w", encoding="utf-8") as fout:
        for row in picked:
            rec = convert_trace(row)
            if not rec:
                continue
            errs = validate(rec["messages"])
            details.append(
                {
                    "trace_id": rec.get("trace_id"),
                    "repo": rec.get("repo"),
                    "n_messages": len(rec["messages"]),
                    "n_tools": len(rec.get("tools") or []),
                    "has_reasoning": any(
                        m.get("reasoning_content") for m in rec["messages"]
                    ),
                    "validation_errors": errs,
                    "ok": not errs,
                }
            )
            if errs:
                continue
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            if args.pretty_dir:
                safe = re.sub(r"[^\w.-]+", "_", rec["trace_id"])[:80]
                (args.pretty_dir / f"{safe}.pretty.json").write_text(
                    json.dumps(rec, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

    report = {
        "raw": str(args.raw_jsonl),
        "session_meta": str(args.session_meta) if args.session_meta else None,
        "candidates": len(candidates),
        "picked": len(picked),
        "kept": kept,
        "output": str(args.output),
        "details": details,
    }
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
