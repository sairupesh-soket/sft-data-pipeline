#!/usr/bin/env python3
"""
Flatten OpenAI SFT JSONL into linearized mid-training text.

Primary mid-train output is Markdown (.md) with USER / THINK / TOOL / ASSISTANT
sections. Also supports .jsonl ({"text": "..."}) for tokenization and plain .txt.

Skips the system prompt by default.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _as_text(val) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    return str(val).strip()


def linearize_messages(messages: list[dict], *, include_system: bool = False) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system" and not include_system:
            continue
        if role == "system":
            parts.append("SYSTEM:\n" + _as_text(msg.get("content")))
        elif role == "user":
            parts.append("USER:\n" + _as_text(msg.get("content")))
        elif role == "assistant":
            think = _as_text(msg.get("reasoning_content"))
            if think:
                parts.append("THINK:\n" + think)
            tcs = msg.get("tool_calls") or []
            if tcs:
                for tc in tcs:
                    fn = (tc.get("function") or {})
                    name = fn.get("name") or "tool"
                    args_raw = fn.get("arguments") or "{}"
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    except json.JSONDecodeError:
                        args = {"raw": args_raw}
                    if isinstance(args, dict):
                        body = "\n".join(f"  {k}: {v}" for k, v in args.items())
                    else:
                        body = f"  {args}"
                    parts.append(f"TOOL {name}:\n{body}")
            content = _as_text(msg.get("content"))
            if content:
                parts.append("ASSISTANT:\n" + content)
        elif role == "tool":
            parts.append("TOOL observation:\n" + _as_text(msg.get("content")))
    return "\n\n".join(p for p in parts if p.strip())


def collapse_consecutive_empty_user_blocks(text: str) -> str:
    """In linearized MD, drop runs of 2+ empty USER blocks; keep a single empty USER."""
    # Split on blank-line separated blocks, but keep header lines intact.
    blocks = re.split(r"\n\n+", text.strip())
    out: list[str] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        stripped = b.strip()
        is_empty_user = stripped == "USER:" or (
            stripped.startswith("USER:") and not stripped[5:].strip()
        )
        if not is_empty_user:
            out.append(b)
            i += 1
            continue
        j = i + 1
        while j < len(blocks):
            s2 = blocks[j].strip()
            if s2 == "USER:" or (s2.startswith("USER:") and not s2[5:].strip()):
                j += 1
            else:
                break
        streak = j - i
        if streak == 1:
            out.append(b)
        i = j
    return "\n\n".join(out).strip() + ("\n" if out else "")


def format_md_sample(n: int, text: str, *, session_id: str = "", n_traces: int = 0) -> str:
    """One linearized mid-train document (trajectory text only; no sample/session headers)."""
    del n, session_id, n_traces
    return text.strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Linearize OpenAI SFT JSONL into mid-training Markdown (or jsonl/txt)"
    )
    parser.add_argument("input", help="OpenAI SFT JSONL from sft_pipeline.py")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output path: .md (mid-train default), .jsonl, or .txt",
    )
    parser.add_argument(
        "--include-system",
        action="store_true",
        help="Keep the system prompt in the linearized text (default: drop it)",
    )
    parser.add_argument(
        "--per-sample",
        action="store_true",
        help="Write one sample_NNNNN.md per session into the output directory",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="First sample_NNNNN.md index when using --per-sample (default: 1)",
    )
    args = parser.parse_args()

    src = Path(args.input)
    dest = Path(args.output)
    per_sample = args.per_sample or dest.is_dir()
    if per_sample:
        dest.mkdir(parents=True, exist_ok=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
    suffix = dest.suffix.lower()
    as_jsonl = suffix == ".jsonl" and not per_sample
    as_md = per_sample or suffix == ".md" or suffix == ""

    n = 0
    fout = None if per_sample else dest.open("w", encoding="utf-8")
    try:
        if fout is not None and as_md:
            fout.write(
                "# Mid-training linearized trajectories\n\n"
                "Converted from OpenAI SFT (`messages`) into plain USER / THINK / "
                "TOOL / ASSISTANT text for continued pretraining / mid-training.\n\n"
            )
        with src.open(encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = linearize_messages(
                    row.get("messages") or [], include_system=args.include_system
                )
                text = collapse_consecutive_empty_user_blocks(text)
                if not text.strip():
                    continue
                n += 1
                if per_sample:
                    idx = args.start_index + n - 1
                    sample_path = dest / f"sample_{idx:05d}.md"
                    sample_path.write_text(format_md_sample(n, text), encoding="utf-8")
                elif as_jsonl:
                    fout.write(
                        json.dumps(
                            {"text": text},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                elif as_md:
                    if n > 1:
                        fout.write("\n---\n\n")
                    fout.write(format_md_sample(n, text))
                else:
                    fout.write(f"===== SAMPLE {n} =====\n")
                    fout.write(text)
                    fout.write("\n\n")
    finally:
        if fout is not None:
            fout.close()

    print(f"Wrote {n} linearized trajectories to {dest}")


if __name__ == "__main__":
    main()
