#!/usr/bin/env python3
"""
Convert HF trace parquets → OpenAI supervised fine-tuning JSONL.

Target format (per OpenAI SFT docs):
  {"messages":[{"role":"system|user|assistant|tool", ...}], "tools":[...], "parallel_tool_calls": bool}

Filters:
  - Drop samples with Claude / other AI-vendor-centric system prompts
  - Scrub model-name mentions from message content (NOT tool ids / tool call names)
  - Strip all non-SFT metadata fields

Usage:
  python3 convert_to_openai_sft.py
  python3 convert_to_openai_sft.py --limit-files 5 --dry-run
  python3 convert_to_openai_sft.py --input-dir /path/OUTPUT --output-dir /path/OPENAI_SFT
  python3 convert_to_openai_sft.py --workers 8 --max-rows-per-jsonl 200000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Optional

import pyarrow.parquet as pq

# Local imports (same package dir)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from filters import sample_has_vendor_system, scrub_messages  # noqa: E402
from normalize import (  # noqa: E402
    build_sft_record,
    extract_messages_and_tools,
    trim_to_trainable,
    validate_sft_messages,
)

DEFAULT_INPUT = "/projects/data/datasets/translation_data/Traces_datasets/OUTPUT"
DEFAULT_OUTPUT = str(Path(__file__).resolve().parents[1] / "openai_sft_out")

SHARD_RE = re.compile(r"^(.*)_(\d+)$")


def dataset_key_from_filename(stem: str) -> str:
    m = SHARD_RE.match(stem)
    if m and len(m.group(2)) <= 6:
        return m.group(1)
    return stem


def list_parquet_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.parquet"))
    # Skip HuggingFace cache tree stubs / junk
    files = [
        f
        for f in files
        if ".cache__huggingface__trees__" not in f.name and not f.name.startswith(".")
    ]
    return files


def convert_row(row: dict[str, Any]) -> tuple[Optional[dict[str, Any]], str]:
    """Convert one row → SFT record or (None, skip_reason)."""
    messages, tools, tag = extract_messages_and_tools(row)
    if not messages:
        return None, f"skip_unconvertible:{tag}"

    if sample_has_vendor_system(messages):
        return None, "skip_vendor_system"

    messages = scrub_messages(messages)
    messages = trim_to_trainable(messages)
    ok, why = validate_sft_messages(messages)
    if not ok:
        return None, f"skip_invalid:{why}"

    rec = build_sft_record(messages, tools)
    return rec, f"ok:{tag}"


def looks_unconvertible_schema(cols: list[str]) -> Optional[str]:
    """Cheap schema-only reject for known non-SFT layouts."""
    if not cols:
        return "empty_schema"
    if str(cols[0]).startswith("version https://git-lfs"):
        return "lfs_pointer"
    lower = {c.lower() for c in cols}
    # CPU branch traces
    if lower >= {"ip", "type", "taken"} and len(lower) <= 5:
        return "cpu_branch_trace"
    # bare event session logs with no chat fields
    chatish = {
        "messages", "conversations", "conversation", "prompt", "completion",
        "question", "answer", "answer_content", "problem", "solution",
        "instruction", "output", "turns", "trajectory", "conversation_json",
        "query", "response", "human", "assistant", "system", "system_prompt",
        "generated_solution", "reasoning", "chat", "dialogue",
        "original_text", "translation", "source", "target",
        "inputs", "targets", "expected_answer",
    }
    if not (lower & chatish):
        # allow if any json blob field
        if not any("json" in c or c in {"trace", "payload", "data", "spans"} for c in lower):
            return "no_chat_like_columns"
    return None


def is_token_id_pair_schema(schema) -> bool:
    """True when inputs/targets (or input/target) are list<int> token sequences (e.g. P3)."""
    try:
        names = {f.name.lower(): f for f in schema}
    except Exception:
        return False
    for uk, ak in (("inputs", "targets"), ("input", "target")):
        if uk in names and ak in names:
            ut, at = str(names[uk].type), str(names[ak].type)
            # list<item: int32/int64> etc.
            if "list" in ut.lower() and "int" in ut.lower() and "list" in at.lower() and "int" in at.lower():
                return True
    return False


def process_one_file(args: tuple) -> dict[str, Any]:
    """
    Worker: convert one parquet → one or more JSONL shards under output_dir/by_dataset/.
    Returns stats dict.
    """
    path_str, output_dir, max_rows_per_jsonl, dry_run = args
    path = Path(path_str)
    stats: dict[str, Any] = {
        "file": path.name,
        "rows_in": 0,
        "rows_out": 0,
        "reasons": Counter(),
        "error": None,
        "out_files": [],
    }
    try:
        pf = pq.ParquetFile(str(path))
        cols = [f.name for f in pf.schema_arrow]
        schema_skip = looks_unconvertible_schema(cols)
        if schema_skip:
            # Still count rows for reporting without converting
            stats["rows_in"] = pf.metadata.num_rows if pf.metadata else 0
            stats["reasons"][f"skip_schema:{schema_skip}"] = stats["rows_in"] or 1
            return stats
        if is_token_id_pair_schema(pf.schema_arrow):
            stats["rows_in"] = pf.metadata.num_rows if pf.metadata else 0
            stats["reasons"]["skip_schema:token_id_inputs_targets"] = stats["rows_in"] or 1
            return stats

        ds_key = dataset_key_from_filename(path.stem)
        out_ds_dir = Path(output_dir) / "by_dataset" / ds_key
        if not dry_run:
            out_ds_dir.mkdir(parents=True, exist_ok=True)

        # Unique shard name from source file stem to avoid collisions
        shard_base = path.stem
        buffer: list[str] = []
        part_idx = 0
        written = 0

        def flush():
            nonlocal buffer, part_idx, written
            if not buffer:
                return
            if dry_run:
                written += len(buffer)
                buffer = []
                return
            part_idx += 1
            out_path = out_ds_dir / f"{shard_base}_sft_{part_idx:04d}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(buffer))
                f.write("\n")
            stats["out_files"].append(str(out_path))
            written += len(buffer)
            buffer = []

        # Stream row groups to limit memory
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg)
            rows = table.to_pylist()
            stats["rows_in"] += len(rows)
            for row in rows:
                rec, reason = convert_row(row)
                # Coarse bucket for summary
                if reason.startswith("ok"):
                    stats["reasons"]["ok"] += 1
                elif reason.startswith("skip_vendor_system"):
                    stats["reasons"]["skip_vendor_system"] += 1
                elif reason.startswith("skip_invalid"):
                    stats["reasons"]["skip_invalid"] += 1
                elif reason.startswith("skip_unconvertible"):
                    stats["reasons"]["skip_unconvertible"] += 1
                else:
                    stats["reasons"][reason[:80]] += 1
                if rec is None:
                    continue
                buffer.append(json.dumps(rec, ensure_ascii=False))
                stats["rows_out"] += 1
                if len(buffer) >= max_rows_per_jsonl:
                    flush()
        flush()
        stats["rows_written"] = written
    except Exception:
        stats["error"] = traceback.format_exc(limit=8)
    return stats


def merge_counters(dst: Counter, src: Counter | dict) -> None:
    for k, v in src.items():
        dst[k] += int(v)


def write_manifest(output_dir: Path, all_stats: list[dict[str, Any]], elapsed: float) -> Path:
    reasons = Counter()
    total_in = total_out = errors = 0
    per_dataset: dict[str, dict[str, int]] = {}

    for st in all_stats:
        total_in += st.get("rows_in", 0)
        total_out += st.get("rows_out", 0)
        if st.get("error"):
            errors += 1
        merge_counters(reasons, st.get("reasons") or {})
        ds = dataset_key_from_filename(Path(st["file"]).stem)
        d = per_dataset.setdefault(ds, {"rows_in": 0, "rows_out": 0, "files": 0, "errors": 0})
        d["rows_in"] += st.get("rows_in", 0)
        d["rows_out"] += st.get("rows_out", 0)
        d["files"] += 1
        if st.get("error"):
            d["errors"] += 1

    report = {
        "elapsed_sec": round(elapsed, 2),
        "files_processed": len(all_stats),
        "files_errored": errors,
        "rows_in": total_in,
        "rows_out": total_out,
        "keep_rate": round(total_out / total_in, 4) if total_in else 0.0,
        "reason_counts": dict(reasons.most_common(80)),
        "per_dataset_sample": {
            k: per_dataset[k]
            for k in sorted(per_dataset.keys())[:50]
        },
        "num_datasets": len(per_dataset),
    }
    out_path = output_dir / "conversion_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    # Also full per-dataset
    with open(output_dir / "per_dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(per_dataset, f, indent=2)
    # Error log
    with open(output_dir / "errors.jsonl", "w", encoding="utf-8") as f:
        for st in all_stats:
            if st.get("error"):
                f.write(json.dumps({"file": st["file"], "error": st["error"]}) + "\n")
    return out_path


def append_combined_jsonl(output_dir: Path, max_lines: int = 0) -> Optional[Path]:
    """Optional: concatenate all by_dataset jsonl into combined/ (capped if max_lines>0)."""
    by_ds = output_dir / "by_dataset"
    if not by_ds.exists():
        return None
    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    out_path = combined_dir / "all_sft.jsonl"
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for p in sorted(by_ds.rglob("*.jsonl")):
            with open(p, "r", encoding="utf-8") as inp:
                for line in inp:
                    if not line.strip():
                        continue
                    out.write(line if line.endswith("\n") else line + "\n")
                    n += 1
                    if max_lines and n >= max_lines:
                        return out_path
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert trace parquets to OpenAI SFT JSONL")
    p.add_argument("--input-dir", default=DEFAULT_INPUT)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4))))
    p.add_argument(
        "--max-pending",
        type=int,
        default=0,
        help="Max in-flight files (default: workers*2). Prevents ProcessPool pipe deadlock.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="How many files to schedule per outer shard/batch (logging + memory).",
    )
    p.add_argument("--max-rows-per-jsonl", type=int, default=200_000)
    p.add_argument("--limit-files", type=int, default=0, help="Process only first N files (0=all)")
    p.add_argument("--only-glob", default="", help="Optional glob filter on filenames, e.g. 'akseljoonas*'")
    p.add_argument("--dry-run", action="store_true", help="Convert in-memory only; do not write JSONL")
    p.add_argument("--combine", action="store_true", help="Also write combined/all_sft.jsonl")
    p.add_argument("--resume", action="store_true", help="Skip files already listed in done_files.txt")
    p.add_argument(
        "--small-first",
        action="store_true",
        default=True,
        help="Process smaller parquet files first (default on)",
    )
    p.add_argument("--no-small-first", action="store_false", dest="small_first")
    return p.parse_args()


def _run_bounded_pool(
    tasks: list[tuple],
    workers: int,
    max_pending: int,
    on_result,
    log: logging.Logger,
) -> None:
    """
    Submit tasks with a bounded in-flight window so ProcessPoolExecutor
    does not deadlock on a full pipe when queueing tens of thousands of jobs.
    """
    if max_pending < 1:
        max_pending = max(2, workers * 2)

    it = iter(tasks)
    in_flight: dict = {}
    submitted = 0
    total = len(tasks)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        def _submit_one() -> bool:
            nonlocal submitted
            try:
                t = next(it)
            except StopIteration:
                return False
            fut = ex.submit(process_one_file, t)
            in_flight[fut] = t[0]
            submitted += 1
            return True

        # Prime the pool
        while len(in_flight) < max_pending and _submit_one():
            pass

        while in_flight:
            # Wait for at least one to finish
            done_set, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for fut in done_set:
                path = in_flight.pop(fut)
                try:
                    st = fut.result()
                except Exception:
                    st = {
                        "file": Path(path).name,
                        "rows_in": 0,
                        "rows_out": 0,
                        "reasons": Counter(),
                        "error": traceback.format_exc(limit=8),
                        "out_files": [],
                    }
                on_result(st, submitted, total)
                # Keep the window full
                while len(in_flight) < max_pending and _submit_one():
                    pass


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "convert.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("openai_sft")

    if not input_dir.is_dir():
        log.error("Input dir not found: %s", input_dir)
        return 1

    files = list_parquet_files(input_dir)
    if args.only_glob:
        files = [f for f in files if fnmatch(f.name, args.only_glob)]

    done_path = output_dir / "done_files.txt"
    done = set()
    if args.resume and done_path.exists():
        done = {line.strip() for line in done_path.read_text().splitlines() if line.strip()}
        files = [f for f in files if f.name not in done]

    if args.small_first:
        # Fast wins first; huge files (multi-GB) last so ETA is visible early
        files = sorted(files, key=lambda p: p.stat().st_size)

    if args.limit_files:
        files = files[: args.limit_files]

    max_pending = args.max_pending or max(2, args.workers * 2)
    log.info("Input: %s (%d parquet files to process)", input_dir, len(files))
    log.info(
        "Output: %s | workers=%d | max_pending=%d | batch_size=%d | dry_run=%s | small_first=%s",
        output_dir, args.workers, max_pending, args.batch_size, args.dry_run, args.small_first,
    )

    if not files:
        log.warning("No files to process.")
        return 0

    t0 = time.time()
    all_stats: list[dict[str, Any]] = []
    rows_in = rows_out = errors = 0
    done_n = 0
    total = len(files)

    # Process in outer shards/batches for clearer progress + smaller all_stats peaks
    batch_size = max(1, args.batch_size)
    num_batches = (total + batch_size - 1) // batch_size

    def handle_result(st: dict[str, Any], submitted: int, batch_total: int) -> None:
        nonlocal done_n, rows_in, rows_out, errors
        all_stats.append(st)
        done_n += 1
        rows_in += st.get("rows_in", 0)
        rows_out += st.get("rows_out", 0)
        if st.get("error"):
            errors += 1
            log.warning("Error in %s: %s", st["file"], st["error"].splitlines()[-1][:200])
        if not args.dry_run and not st.get("error"):
            with open(done_path, "a", encoding="utf-8") as df:
                df.write(st["file"] + "\n")

        if done_n % 10 == 0 or done_n == total:
            elapsed = max(time.time() - t0, 1e-6)
            rate = done_n / elapsed
            remain = (total - done_n) / rate if rate > 0 else 0
            log.info(
                "Progress %d/%d files (%.1f%%) | rows_in=%d rows_out=%d errors=%d | "
                "%.2f files/s | ETA ~%.1f min",
                done_n, total, 100.0 * done_n / total, rows_in, rows_out, errors,
                rate, remain / 60.0,
            )
            # Lightweight live progress file
            try:
                with open(output_dir / "progress.json", "w", encoding="utf-8") as pf:
                    json.dump(
                        {
                            "done_files": done_n,
                            "total_files": total,
                            "rows_in": rows_in,
                            "rows_out": rows_out,
                            "errors": errors,
                            "elapsed_sec": round(elapsed, 1),
                            "files_per_sec": round(rate, 3),
                            "eta_min": round(remain / 60.0, 1),
                        },
                        pf,
                    )
            except Exception:
                pass

    for bi in range(num_batches):
        chunk = files[bi * batch_size : (bi + 1) * batch_size]
        log.info("Shard/batch %d/%d — scheduling %d files", bi + 1, num_batches, len(chunk))
        tasks = [
            (str(f), str(output_dir), args.max_rows_per_jsonl, args.dry_run)
            for f in chunk
        ]
        _run_bounded_pool(tasks, args.workers, max_pending, handle_result, log)

    elapsed = time.time() - t0
    report_path = write_manifest(output_dir, all_stats, elapsed)
    log.info("Wrote report: %s", report_path)
    log.info(
        "DONE in %.1fs (%.1f min) | files=%d rows_in=%d rows_out=%d errors=%d",
        elapsed, elapsed / 60.0, len(all_stats), rows_in, rows_out, errors,
    )

    if args.combine and not args.dry_run:
        cpath = append_combined_jsonl(output_dir)
        log.info("Combined JSONL: %s", cpath)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
