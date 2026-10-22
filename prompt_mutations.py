#!/usr/bin/env python3
"""
Baseten-style system-prompt mutations for OpenAI SFT JSONL.

1) Generate a small bank of meaning-preserving paraphrases via a self-hosted
   OpenAI-compatible chat API (e.g. Qwen).
2) Assign one variant per SFT row (trajectory / gold outputs stay fixed).

Env (optional defaults):
  OPENAI_BASE_URL  e.g. http://127.0.0.1:8000/v1
  OPENAI_API_KEY   any non-empty key your gateway expects
  OPENAI_MODEL     e.g. qwen3-30b

Examples:
  # Generate 12 variants with your Qwen endpoint
  python env_traj_utils/prompt_mutations.py generate \\
      --base-url http://HOST:PORT/v1 --model qwen3-30b \\
      --count 12 -o prompt_bank.json

  # Apply bank onto existing SFT JSONL
  python env_traj_utils/prompt_mutations.py apply \\
      structured_5000_openai_sft.jsonl \\
      --bank prompt_bank.json \\
      -o structured_5000_openai_sft_mutated.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapt_langfuse_traces import CURSOR_SFT_SYSTEM

MUTATION_SYSTEM = (
    "You rewrite general agentic coding-assistant system prompts. "
    "Preserve intent, constraints, section tags, and tool semantics exactly. "
    "Change only phrasing within sections (wording / sentence structure). "
    "Keep the same XML-like section tags if present "
    "(communication, tool_calling, making_code_changes, search_and_reading, debugging, calling_external_apis). "
    "Do not mention any specific IDE or product brand (especially not Cursor). "
    "Do not add new tools, remove tools, or invent schemas. "
    "Do not add markdown fences or commentary. "
    "Return ONLY the full rewritten system prompt."
)

MUTATION_USER_TMPL = """Rewrite this agentic coding assistant system prompt. Keep the same meaning and structure.

Hard constraints for the rewrite:
- Keep it as a general (product-agnostic) agentic coding assistant prompt
- Do NOT mention Cursor or any other IDE/product brand name
- Preserve section tags if present
- Keep the same tool names if listed: read_file, edit_file, grep_search, shell, delete_file
- Still require: brief explanation before tool calls; never name tools to the USER
- Prefer search/read before edit; keep edits focused and runnable
- Do not invent new capabilities

Seed prompt:
{seed}
"""

# Offline fallbacks use the general seed — see HANDWRITTEN_VARIANTS below.


def _chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    timeout_s: float,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 4096,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {err}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to reach {url}: {e}") from e

    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"Empty choices from API: {payload!r}")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):
        # Some gateways return content parts
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text") or "")
            elif isinstance(p, str):
                parts.append(p)
        content = "".join(parts)
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"No text content in response: {payload!r}")
    return content.strip()


def _clean_prompt(text: str) -> str:
    text = text.strip()
    # Drop optional reasoning wrappers some chat models emit
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.I).strip()
    text = re.sub(r"<reasoning>[\s\S]*?</reasoning>", "", text, flags=re.I).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:\w+)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = text.strip().strip('"').strip("'").strip()
    # Preserve newlines (multi-section prompts); tidy whitespace only
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _is_plausible_variant(text: str, seed: str) -> bool:
    if len(text) < max(80, len(seed) // 4) or len(text) > max(4000, len(seed) * 3):
        return False
    lower = text.lower()
    if not any(k in lower for k in ("tool", "agent", "coding", "assistant")):
        return False
    # If seed has section tags, require at least one still present
    seed_tags = re.findall(r"</?([a-z_]+)>", seed.lower())
    if seed_tags:
        if not any(tag in lower for tag in ("communication", "tool_calling", "making_code_changes")):
            return False
    banned = (
        "<function=",
        "str_replace_editor",
        "you must output json",
        "as an ai language model",
        "cursor",  # no IDE/product brand in mutated prompts
    )
    if any(b in lower for b in banned):
        return False
    return True


HANDWRITTEN_VARIANTS = [CURSOR_SFT_SYSTEM]


def generate_bank(
    *,
    seed: str,
    count: int,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout_s: float,
    include_seed: bool,
    max_attempts: int,
) -> dict[str, Any]:
    prompts: list[str] = []
    seen: set[str] = set()
    if include_seed:
        prompts.append(seed)
        seen.add(seed.lower())

    attempts = 0
    while len(prompts) < count and attempts < max_attempts:
        attempts += 1
        raw = _chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=[
                {"role": "system", "content": MUTATION_SYSTEM},
                {"role": "user", "content": MUTATION_USER_TMPL.format(seed=seed)},
            ],
            temperature=temperature,
            timeout_s=timeout_s,
        )
        cleaned = _clean_prompt(raw)
        key = cleaned.lower()
        if key in seen:
            continue
        if not _is_plausible_variant(cleaned, seed):
            continue
        seen.add(key)
        prompts.append(cleaned)

    if len(prompts) < 2:
        raise RuntimeError(
            f"Only got {len(prompts)} usable prompt(s) after {attempts} attempts. "
            "Check model endpoint / temperature."
        )

    return {
        "seed": seed,
        "generator_model": model,
        "base_url": base_url.rstrip("/"),
        "count": len(prompts),
        "prompts": prompts,
    }


def load_bank(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        prompts = [str(x).strip() for x in data if str(x).strip()]
    elif isinstance(data, dict):
        prompts = [str(x).strip() for x in (data.get("prompts") or []) if str(x).strip()]
    else:
        raise ValueError(f"Unrecognized bank format: {path}")
    if not prompts:
        raise ValueError(f"Empty prompt bank: {path}")
    return prompts


def pick_prompt(prompts: list[str], *, index: int, seed: int, mode: str) -> tuple[str, int]:
    if mode == "round_robin":
        i = index % len(prompts)
        return prompts[i], i
    # stable hash so re-runs are deterministic for the same row order + seed
    h = hashlib.sha256(f"{seed}:{index}".encode()).hexdigest()
    i = int(h[:8], 16) % len(prompts)
    return prompts[i], i


def apply_bank(
    src: Path,
    dest: Path,
    prompts: list[str],
    *,
    seed: int,
    mode: str,
    keep_prompt_id: bool,
) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    counts: dict[int, int] = {}
    with src.open(encoding="utf-8") as fin, dest.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row: dict[str, Any] = json.loads(line)
            msgs = row.get("messages")
            pid = -1
            if isinstance(msgs, list) and msgs and msgs[0].get("role") == "system":
                prompt, pid = pick_prompt(prompts, index=n, seed=seed, mode=mode)
                msgs[0]["content"] = prompt
                if keep_prompt_id:
                    row["prompt_id"] = pid
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[pid] = counts.get(pid, 0) + 1
            n += 1
    return {"written": n, "bank_size": len(prompts), "prompt_id_counts": counts}


def cmd_generate(args: argparse.Namespace) -> None:
    if args.seed_file:
        seed = Path(args.seed_file).read_text(encoding="utf-8").strip()
    elif args.seed:
        seed = args.seed
    else:
        seed = CURSOR_SFT_SYSTEM
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or ""
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    model = args.model or os.environ.get("OPENAI_MODEL") or ""
    if not base_url or not model:
        raise SystemExit(
            "Need --base-url and --model (or OPENAI_BASE_URL / OPENAI_MODEL env vars)."
        )

    bank = generate_bank(
        seed=seed,
        count=args.count,
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=args.temperature,
        timeout_s=args.timeout,
        include_seed=not args.no_seed,
        max_attempts=args.max_attempts,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bank, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {bank['count']} prompts to {out}")


def cmd_apply(args: argparse.Namespace) -> None:
    if args.bank:
        prompts = load_bank(Path(args.bank))
    elif args.handwritten:
        prompts = list(HANDWRITTEN_VARIANTS)
    else:
        raise SystemExit("Provide --bank PATH or --handwritten")

    report = apply_bank(
        Path(args.input),
        Path(args.output),
        prompts,
        seed=args.seed,
        mode=args.mode,
        keep_prompt_id=args.keep_prompt_id,
    )
    print(json.dumps(report, indent=2))


def cmd_write_handwritten(args: argparse.Namespace) -> None:
    out = Path(args.output)
    bank = {
        "seed": CURSOR_SFT_SYSTEM,
        "generator_model": "handwritten",
        "count": len(HANDWRITTEN_VARIANTS),
        "prompts": HANDWRITTEN_VARIANTS,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bank, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(HANDWRITTEN_VARIANTS)} handwritten prompts to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="System prompt mutations for SFT JSONL")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Call self-hosted Qwen to build a prompt bank")
    g.add_argument("-o", "--output", required=True, help="Output prompt_bank.json")
    g.add_argument("--base-url", help="OpenAI-compatible base URL (.../v1)")
    g.add_argument("--api-key", help="API key (default OPENAI_API_KEY or EMPTY)")
    g.add_argument("--model", help="Model name served by your gateway")
    g.add_argument("--count", type=int, default=12, help="Target number of prompts in bank")
    g.add_argument("--temperature", type=float, default=0.9)
    g.add_argument("--timeout", type=float, default=120.0)
    g.add_argument("--max-attempts", type=int, default=40)
    g.add_argument("--seed", help="Override seed system prompt text")
    g.add_argument(
        "--seed-file",
        help="Seed prompt file (default: cursor_system_prompt.md via CURSOR_SFT_SYSTEM)",
    )
    g.add_argument("--no-seed", action="store_true", help="Do not keep original seed in bank")
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser("apply", help="Assign bank prompts onto SFT JSONL rows")
    a.add_argument("input", help="Input OpenAI SFT JSONL")
    a.add_argument("-o", "--output", required=True, help="Output mutated JSONL")
    a.add_argument("--bank", help="prompt_bank.json from generate / handwritten")
    a.add_argument(
        "--handwritten",
        action="store_true",
        help="Use built-in handwritten variants (no LLM call)",
    )
    a.add_argument("--mode", choices=("hash", "round_robin"), default="hash")
    a.add_argument("--seed", type=int, default=42, help="RNG / hash seed for assignment")
    a.add_argument(
        "--keep-prompt-id",
        action="store_true",
        help="Write prompt_id index onto each row",
    )
    a.set_defaults(func=cmd_apply)

    h = sub.add_parser("write-handwritten", help="Dump built-in variants to JSON")
    h.add_argument("-o", "--output", required=True)
    h.set_defaults(func=cmd_write_handwritten)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
