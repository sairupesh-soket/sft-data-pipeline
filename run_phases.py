#!/usr/bin/env python3
"""One-shot OpenAI SFT pipeline (Phases 1–3). Single file for login / Slurm / K8s.

Phase 1 — chat (~129M): messages / conversations → OpenAI messages
Phase 2 — pairs (~476M): alpaca / Q-A (user+assistant cols) → OpenAI messages
Phase 3 — aliases (~15M): Q/A, long_answer, … → user/assistant via alias map

Training JSONL has no _meta (OpenAI-ready). Optional --with-review keeps provenance.

K8s / env defaults:
  SFT_SRC   parquet root   (default: /projects/.../SFT/OUTPUT)
  SFT_OUT   output dir
  SFT_DEDUP path with sft_normalize.py

Examples:
  python pipeline/run_phases.py --phase 1 2 3 --limit 20 --seed 42
  python pipeline/run_phases.py --phase 2 --limit 50 --src /data/OUTPUT --out /data/out
  # one pod per shard (file-level split; use with --limit for samples, or later --full)
  python pipeline/run_phases.py --phase 1 --limit 100 --num-shards 8 --shard-id 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from bisect import bisect_right
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Paths (override with flags or env for K8s)
# ---------------------------------------------------------------------------
DEFAULT_SRC = Path(os.environ.get("SFT_SRC", "/projects/data/datasets/translation_data/SFT/OUTPUT"))
DEFAULT_OUT = Path(os.environ.get("SFT_OUT", "/home/bharath/sft-pipeline/review_openai/phases"))
DEFAULT_DEDUP = Path(
    os.environ.get("SFT_DEDUP", "/projects/data/datasets/translation_data/SFT/DEDUP_PIPELINE")
)

sys.path.insert(0, str(DEFAULT_DEDUP))
import sft_normalize as N  # noqa: E402

# ---------------------------------------------------------------------------
# Phase 3 — explicit column aliases → user / assistant roles
# (does not rewrite parquet; only maps names into a fields plan)
# ---------------------------------------------------------------------------
ALIAS_PAIRS: list[tuple[str, str]] = [
    ("q", "a"),
    ("question", "long_answer"),
    ("question", "reference_answer"),
    ("question", "correct_answer"),
    ("question", "final_answer"),
    ("question", "answer_text"),
    ("question", "final_response"),
    ("prompt", "reference"),
    ("instruction", "final_answer"),
    ("user_query", "enhanced_completion"),
    ("query", "positive"),
]
SYSTEM_ALIASES = ("system", "system_prompt", "sys_prompt", "persona", "role_prompt")
THINK_ALIASES = (
    "think", "reasoning", "reasoning_content", "gt_cot",
    "chain_of_thought", "cot", "rationale",
)

PHASE_HELP = {
    1: "chat (~129M) — messages/conversations already chat-shaped",
    2: "pairs (~476M) — alpaca / Q-A with both prompt + reply columns",
    3: "extra aliases (~15M) — Q/A, long_answer, … → user/assistant",
}

# ---------------------------------------------------------------------------
# Convert helpers
# ---------------------------------------------------------------------------
ROLE_MAP = {
    "system": "system", "sys": "system",
    "user": "user", "human": "user", "usr": "user", "prompter": "user",
    "assistant": "assistant", "gpt": "assistant", "chatgpt": "assistant",
    "bot": "assistant", "ai": "assistant", "model": "assistant",
    "answer": "assistant", "response": "assistant",
    "tool_call": "assistant", "tool call": "assistant",
    "function_call": "assistant", "tool_use": "assistant",
    "tool": "tool", "tool_response": "tool", "tool_output": "tool",
    "tool_result": "tool", "function": "tool", "function_response": "tool",
    "observation": "tool", "ipython": "tool",
}
REASONING_ROLES = {
    "reasoning", "think", "thought", "thinking",
    "chain_of_thought", "cot", "rationale",
}
ALLOWED_ROLES = {"system", "user", "assistant", "tool"}


def plan_extra_alias(colnames: list[str]) -> dict | None:
    low = {c.lower().strip(): c for c in colnames}
    user_c = asst_c = None
    pair_key = None
    for u, a in ALIAS_PAIRS:
        if u in low and a in low:
            user_c, asst_c = low[u], low[a]
            pair_key = f"{u}→{a}"
            break
    if not user_c or not asst_c:
        return None
    return {
        "mode": "fields",
        "system": next((low[k] for k in SYSTEM_ALIASES if k in low), None),
        "user": user_c,
        "user_extra": None,
        "think": next((low[k] for k in THINK_ALIASES if k in low), None),
        "assistant": asst_c,
        "alias_pair": pair_key,
        "phase": 3,
    }


def canonicalize_turns(turns):
    out, pending = [], []
    for t in turns:
        role = str(t.get("role", "")).strip().lower()
        content = str(t.get("content", "") or "")
        if not content.strip():
            continue
        if role in REASONING_ROLES:
            pending.append(content.strip())
            continue
        r = ROLE_MAP.get(role)
        if r is None:
            continue
        if r == "assistant" and pending:
            content = "<think>" + "\n\n".join(pending) + "</think>\n\n" + content
            pending = []
        if out and out[-1]["role"] == r:
            out[-1]["content"] += "\n\n" + content
        else:
            out.append({"role": r, "content": content})
    while out and out[-1]["role"] != "assistant":
        out.pop()
    return out


def validate_messages(messages):
    if not messages:
        return "empty"
    if any(m["role"] not in ALLOWED_ROLES for m in messages):
        return "bad_role"
    if any(not str(m.get("content") or "").strip() for m in messages):
        return "empty_turn"
    if not any(m["role"] == "user" for m in messages):
        return "no_user"
    if not any(m["role"] == "assistant" for m in messages):
        return "no_assistant"
    if messages[-1]["role"] != "assistant":
        return "no_final_answer"
    return None


def fold_reasoning_in_chat(raw):
    if not isinstance(raw, list):
        return raw
    out = []
    for t in raw:
        if not isinstance(t, dict):
            out.append(t)
            continue
        t = dict(t)
        rc = t.pop("reasoning_content", None)
        if rc is None:
            rc = t.get("reasoning")
        role = str(t.get("role") or t.get("from") or "").lower()
        cont = t.get("content") or t.get("value") or t.get("text") or t.get("message") or ""
        if rc and str(rc).strip() and role in {"assistant", "gpt", "model", "ai", "bot"}:
            t["content"] = f"<think>{str(rc).strip()}</think>\n\n{cont}"
            t.pop("value", None)
        out.append(t)
    return out


def openai_messages(turns: list[dict]) -> list[dict]:
    msgs = []
    for t in turns:
        role = t.get("role")
        if role not in ALLOWED_ROLES:
            continue
        item = {"role": role, "content": str(t.get("content") or "")}
        if role == "tool":
            tcid = t.get("tool_call_id")
            if tcid:
                item["tool_call_id"] = tcid
        if role == "assistant" and t.get("tool_calls"):
            item["tool_calls"] = t["tool_calls"]
        msgs.append(item)
    return msgs


def row_to_openai(row: dict, plan: dict) -> tuple[dict | None, str | None]:
    if plan.get("mode") == "chat":
        col = plan["col"]
        row = dict(row)
        row[col] = fold_reasoning_in_chat(row.get(col))
    raw = N.to_messages(row, plan)
    if not raw:
        return None, "unmappable_row"
    turns = canonicalize_turns(raw)
    err = validate_messages(turns)
    if err:
        return None, err
    rec: dict = {"messages": openai_messages(turns)}
    tools = row.get("tools")
    has_tool_turn = any(m.get("role") == "tool" or m.get("tool_calls") for m in rec["messages"])
    if has_tool_turn and isinstance(tools, list) and tools:
        rec["tools"] = tools
        rec["parallel_tool_calls"] = False
    return rec, None


def iter_parquet_files(src: Path):
    for dirpath, _, filenames in os.walk(src):
        for fn in filenames:
            if fn.endswith(".parquet"):
                yield Path(dirpath) / fn


def index_chat_or_fields(src: Path, mode: str) -> list[dict]:
    files = []
    for path in iter_parquet_files(src):
        try:
            pf = pq.ParquetFile(path, memory_map=True)
        except Exception:
            continue
        cols = list(pf.schema_arrow.names)
        plan = N.plan(cols)
        if plan.get("mode") is None:
            continue
        if mode != "all" and plan["mode"] != mode:
            continue
        n = pf.metadata.num_rows
        if n <= 0:
            continue
        rel = str(path.relative_to(src))
        files.append(
            {
                "path": path,
                "rel": rel,
                "domain": rel.split(os.sep)[0],
                "n": n,
                "plan": plan,
                "cols": cols,
            }
        )
    files.sort(key=lambda x: x["rel"])
    return files


def index_phase3(src: Path) -> list[dict]:
    files = []
    for path in iter_parquet_files(src):
        try:
            pf = pq.ParquetFile(path, memory_map=True)
        except Exception:
            continue
        cols = list(pf.schema_arrow.names)
        if N.plan(cols).get("mode") is not None:
            continue
        plan = plan_extra_alias(cols)
        if plan is None:
            continue
        n = pf.metadata.num_rows
        if n <= 0:
            continue
        rel = str(path.relative_to(src))
        files.append(
            {
                "path": path,
                "rel": rel,
                "domain": rel.split(os.sep)[0],
                "n": n,
                "plan": plan,
                "cols": cols,
            }
        )
    files.sort(key=lambda x: x["rel"])
    return files


def index_phase(src: Path, phase: int) -> list[dict]:
    if phase == 1:
        return index_chat_or_fields(src, "chat")
    if phase == 2:
        files = index_chat_or_fields(src, "fields")
        return [f for f in files if f["plan"].get("user") and f["plan"].get("assistant")]
    if phase == 3:
        return index_phase3(src)
    raise SystemExit(f"Unknown phase: {phase}")


def apply_shard(files: list[dict], num_shards: int, shard_id: int) -> list[dict]:
    if num_shards <= 1:
        return files
    if shard_id < 0 or shard_id >= num_shards:
        raise SystemExit(f"--shard-id must be in [0, {num_shards})")
    return [f for i, f in enumerate(files) if i % num_shards == shard_id]


def needed_columns(plan: dict, cols: list[str]) -> list[str]:
    keys = []
    for k in ("col", "system", "user", "user_extra", "think", "assistant"):
        v = plan.get(k)
        if v:
            keys.append(v)
    if "tools" in cols:
        keys.append("tools")
    return keys or cols[:8]


def read_row(path: Path, abs_idx: int, columns: list[str]) -> dict | None:
    pf = pq.ParquetFile(path, memory_map=True)
    seen = 0
    want = [c for c in columns if c in pf.schema_arrow.names]
    for rg in range(pf.metadata.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        if abs_idx < seen + n:
            table = pf.read_row_group(rg, columns=want)
            return table.slice(abs_idx - seen, 1).to_pylist()[0]
        seen += n
    return None


def sample_converted(files, *, limit: int, seed: int):
    total = sum(f["n"] for f in files)
    if total <= 0:
        return [], [], {"pool_files": 0, "pool_rows": 0, "attempts": 0}
    cum, acc = [], 0
    for f in files:
        acc += f["n"]
        cum.append(acc)
    rng = random.Random(seed)
    kept, skipped = [], []
    seen_keys = set()
    attempts = 0
    max_attempts = max(limit * 40, 80)
    while len(kept) < limit and attempts < max_attempts:
        attempts += 1
        pick = rng.randrange(total)
        fi = bisect_right(cum, pick)
        meta = files[fi]
        local_idx = pick if fi == 0 else pick - cum[fi - 1]
        key = (meta["rel"], local_idx)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        cols = needed_columns(meta["plan"], meta["cols"])
        try:
            row = read_row(meta["path"], local_idx, cols)
        except Exception as e:
            skipped.append({"file": meta["rel"], "row": local_idx, "reason": type(e).__name__})
            continue
        if row is None:
            skipped.append({"file": meta["rel"], "row": local_idx, "reason": "row_not_found"})
            continue
        rec, reason = row_to_openai(row, meta["plan"])
        if rec is None:
            skipped.append({"file": meta["rel"], "row": local_idx, "reason": reason})
            continue
        rec["_meta"] = {
            "source_file": meta["rel"],
            "domain": meta["domain"],
            "row_index": local_idx,
            "adapter": meta["plan"]["mode"],
            "alias_pair": meta["plan"].get("alias_pair"),
            "plan_user": meta["plan"].get("user"),
            "plan_assistant": meta["plan"].get("assistant"),
            "n_turns": len(rec["messages"]),
            "multi_turn": sum(1 for m in rec["messages"] if m["role"] == "user") > 1,
            "has_tools": "tools" in rec,
            "phase": meta["plan"].get("phase"),
        }
        kept.append(rec)
    return kept, skipped, {"pool_files": len(files), "pool_rows": total, "attempts": attempts}


def stream_converted(
    files,
    *,
    jsonl_path: Path,
    review_path: Path | None,
    phase: int,
    batch_size: int = 512,
    progress_every: int = 50_000,
):
    """Convert every row in the shard; stream train JSONL (no _meta) to disk."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    skip_reasons: Counter = Counter()
    domains: Counter = Counter()
    kept = 0
    failures = 0
    has_system = 0
    has_tools = 0
    rows_seen = 0
    review_f = None
    try:
        with jsonl_path.open("w", encoding="utf-8") as out_f:
            if review_path is not None:
                review_f = review_path.open("w", encoding="utf-8")
            for fi, meta in enumerate(files):
                cols = needed_columns(meta["plan"], meta["cols"])
                try:
                    pf = pq.ParquetFile(meta["path"], memory_map=True)
                    want = [c for c in cols if c in pf.schema_arrow.names]
                    local_idx = 0
                    for batch in pf.iter_batches(columns=want, batch_size=batch_size):
                        for row in batch.to_pylist():
                            rows_seen += 1
                            rec, reason = row_to_openai(row, meta["plan"])
                            if rec is None:
                                failures += 1
                                skip_reasons[reason or "unknown"] += 1
                                local_idx += 1
                                continue
                            train = strip_meta(rec)
                            out_f.write(json.dumps(train, ensure_ascii=False) + "\n")
                            kept += 1
                            domains[meta["domain"]] += 1
                            if any(m["role"] == "system" for m in train["messages"]):
                                has_system += 1
                            if "tools" in train:
                                has_tools += 1
                            if review_f is not None:
                                rec["_meta"] = {
                                    "source_file": meta["rel"],
                                    "domain": meta["domain"],
                                    "row_index": local_idx,
                                    "adapter": meta["plan"]["mode"],
                                    "alias_pair": meta["plan"].get("alias_pair"),
                                    "phase": phase,
                                }
                                review_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            local_idx += 1
                            if kept % progress_every == 0:
                                print(
                                    f"  progress: kept={kept:,} fail={failures:,} "
                                    f"files={fi + 1}/{len(files)} rows_seen={rows_seen:,}",
                                    flush=True,
                                )
                except Exception as e:
                    failures += 1
                    skip_reasons[type(e).__name__] += 1
                    print(f"  skip file {meta['rel']}: {type(e).__name__}: {e}", flush=True)
                    continue
                if (fi + 1) % 25 == 0 or fi + 1 == len(files):
                    print(
                        f"  files done {fi + 1}/{len(files)} kept={kept:,} fail={failures:,}",
                        flush=True,
                    )
                    out_f.flush()
    finally:
        if review_f is not None:
            review_f.close()

    return {
        "kept": kept,
        "convert_failures": failures,
        "rows_seen": rows_seen,
        "pool_files": len(files),
        "pool_rows": sum(f["n"] for f in files),
        "skip_reasons": dict(skip_reasons),
        "domains": dict(domains),
        "has_system_count": has_system,
        "has_tools_count": has_tools,
    }


def strip_meta(rec: dict) -> dict:
    out = {"messages": rec["messages"]}
    if "tools" in rec:
        out["tools"] = rec["tools"]
        out["parallel_tool_calls"] = rec.get("parallel_tool_calls", False)
    return out


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_pretty(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def write_preview_csv(path: Path, train_rows: list[dict], metas: list[dict]):
    import pandas as pd

    recs = []
    for rec, meta in zip(train_rows, metas):
        user = next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
        asst = next(
            (m["content"] for m in reversed(rec["messages"]) if m["role"] == "assistant"),
            "",
        )
        recs.append(
            {
                "domain": meta.get("domain"),
                "source_file": meta.get("source_file"),
                "row_index": meta.get("row_index"),
                "alias_pair": meta.get("alias_pair"),
                "n_turns": len(rec["messages"]),
                "has_system": any(m["role"] == "system" for m in rec["messages"]),
                "roles": " > ".join(m["role"] for m in rec["messages"]),
                "user": " ".join(user.split())[:400],
                "assistant": " ".join(asst.split())[:400],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs).to_csv(path, index=False, quoting=1, escapechar="\\")


def run_phase(
    phase: int,
    *,
    src: Path,
    out_dir: Path,
    limit: int,
    seed: int,
    with_review: bool,
    num_shards: int,
    shard_id: int,
) -> dict:
    print(f"\n=== PHASE {phase}: {PHASE_HELP[phase]} ===", flush=True)
    files = index_phase(src, phase)
    files = apply_shard(files, num_shards, shard_id)
    pool_rows = sum(f["n"] for f in files)
    shard_tag = f"_shard{shard_id}of{num_shards}" if num_shards > 1 else ""
    print(
        f"pool: {len(files)} files, {pool_rows:,} rows"
        + (f" (shard {shard_id}/{num_shards})" if num_shards > 1 else ""),
        flush=True,
    )

    alias_pool = Counter()
    if phase == 3:
        for f in files:
            alias_pool[f["plan"].get("alias_pair") or "?"] += f["n"]
        print("alias pool:", dict(alias_pool.most_common(12)), flush=True)

    tag = f"phase{phase}{shard_tag}"
    full = limit <= 0
    size_tag = "full" if full else str(limit)
    jsonl = out_dir / f"{tag}_openai_sft_{size_tag}.jsonl"
    report_path = out_dir / f"{tag}_openai_sft_{size_tag}_report.json"

    if full:
        print(f"mode: full stream -> {jsonl}", flush=True)
        review_path = (
            out_dir / f"{tag}_openai_sft_{size_tag}_review.jsonl" if with_review else None
        )
        stats = stream_converted(
            files,
            jsonl_path=jsonl,
            review_path=review_path,
            phase=phase,
        )
        report = {
            "phase": phase,
            "phase_desc": PHASE_HELP[phase],
            "src": str(src),
            "seed": seed,
            "mode": "full",
            "requested": 0,
            "kept": stats["kept"],
            "convert_failures": stats["convert_failures"],
            "rows_seen": stats["rows_seen"],
            "pool_files": stats["pool_files"],
            "pool_rows": stats["pool_rows"],
            "num_shards": num_shards,
            "shard_id": shard_id,
            "skip_reasons": stats["skip_reasons"],
            "domains": stats["domains"],
            "has_system_count": stats["has_system_count"],
            "has_tools_count": stats["has_tools_count"],
            "output_jsonl": str(jsonl),
            "meta_in_training_jsonl": False,
        }
    else:
        kept_with_meta, skipped, stats = sample_converted(files, limit=limit, seed=seed)
        for r in kept_with_meta:
            r.setdefault("_meta", {})["phase"] = phase

        train_rows = [strip_meta(r) for r in kept_with_meta]
        metas = [r.get("_meta") or {} for r in kept_with_meta]

        pretty = out_dir / f"{tag}_openai_sft_{size_tag}.pretty.json"
        preview = out_dir / f"{tag}_openai_sft_{size_tag}_preview.csv"

        write_jsonl(jsonl, train_rows)
        write_pretty(pretty, train_rows)
        write_preview_csv(preview, train_rows, metas)
        if with_review:
            write_jsonl(out_dir / f"{tag}_openai_sft_{size_tag}_review.jsonl", kept_with_meta)

        report = {
            "phase": phase,
            "phase_desc": PHASE_HELP[phase],
            "src": str(src),
            "seed": seed,
            "mode": "sample",
            "requested": limit,
            "kept": len(train_rows),
            "convert_failures": len(skipped),
            "pool_files": stats["pool_files"],
            "pool_rows": stats["pool_rows"],
            "attempts": stats["attempts"],
            "num_shards": num_shards,
            "shard_id": shard_id,
            "skip_reasons": dict(Counter(s["reason"] for s in skipped)),
            "domains": dict(Counter(m.get("domain") for m in metas)),
            "has_system_count": sum(
                1 for r in train_rows if any(m["role"] == "system" for m in r["messages"])
            ),
            "has_tools_count": sum(1 for r in train_rows if "tools" in r),
            "output_jsonl": str(jsonl),
            "output_pretty": str(pretty),
            "meta_in_training_jsonl": False,
        }

    if phase == 3:
        report["alias_pool"] = dict(alias_pool)
        if not full:
            report["alias_in_sample"] = dict(
                Counter(m.get("alias_pair") for m in metas if m.get("alias_pair"))
            )
        report["alias_pairs_supported"] = [f"{u}→{a}" for u, a in ALIAS_PAIRS]
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description="OpenAI SFT phased pipeline (1=chat, 2=pairs, 3=aliases)")
    p.add_argument("--phase", type=int, nargs="+", choices=[1, 2, 3], required=True)
    p.add_argument("--src", default=str(DEFAULT_SRC), help="Parquet root (or env SFT_SRC)")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Output dir (or env SFT_OUT)")
    p.add_argument(
        "--dedup",
        default=str(DEFAULT_DEDUP),
        help="Dir containing sft_normalize.py (or env SFT_DEDUP)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Sample size per phase; use 0 for full stream of the shard",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--with-review", action="store_true", help="Also write JSONL with _meta")
    p.add_argument("--num-shards", type=int, default=1, help="Split file pool across K8s pods")
    p.add_argument("--shard-id", type=int, default=0, help="This pod's shard index [0, num-shards)")
    args = p.parse_args()

    # IndexedJob pods: allow env override when flag left at default
    if args.num_shards > 1 and "JOB_COMPLETION_INDEX" in os.environ:
        args.shard_id = int(os.environ["JOB_COMPLETION_INDEX"])

    dedup = Path(args.dedup)
    if str(dedup) not in sys.path:
        sys.path.insert(0, str(dedup))
    # re-bind normalize if --dedup differs from default import
    global N
    import importlib
    import sft_normalize as _N
    importlib.reload(_N)
    N = _N

    src = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for phase in args.phase:
        reports.append(
            run_phase(
                phase,
                src=src,
                out_dir=out_dir,
                limit=args.limit,
                seed=args.seed,
                with_review=args.with_review,
                num_shards=args.num_shards,
                shard_id=args.shard_id,
            )
        )

    size_tag = "full" if args.limit <= 0 else str(args.limit)
    summary = out_dir / f"phases_summary_{size_tag}.json"
    summary.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nsummary -> {summary}", flush=True)
    if args.limit > 0 and any(r["kept"] < args.limit for r in reports):
        sys.exit(1)


if __name__ == "__main__":
    main()
