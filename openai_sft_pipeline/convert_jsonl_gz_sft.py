#!/usr/bin/env python3
"""
Convert recursive *.jsonl.gz (native / near-OpenAI chat) → OpenAI SFT JSONL.

Same rules as convert_to_openai_sft.py:
  - messages with system|user|assistant|tool
  - drop vendor-centric system prompts
  - scrub model/company name mentions (not tool ids)
  - strip non-SFT metadata

Usage:
  python3 convert_jsonl_gz_sft.py \
    --input-dir /path/A_native_openai \
    --output-dir /projects/data/datasets/traces_team/A_native_openai_sft \
    --workers 16
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_to_openai_sft import convert_row  # noqa: E402

DEFAULT_INPUT = (
    "/projects/data/datasets/code_data/sai_rupesh/taxonomy/"
    "posttraining_openai_sft/A_native_openai"
)
DEFAULT_OUTPUT = "/projects/data/datasets/traces_team/A_native_openai_sft"


def list_gz_files(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*.jsonl.gz") if p.is_file())


def dataset_key(path: Path, input_dir: Path) -> str:
    rel = path.relative_to(input_dir)
    # nemotron_SFT/cascade_sft_stage1/general.jsonl.gz → nemotron_SFT__cascade_sft_stage1__general
    parts = list(rel.parts[:-1]) + [rel.stem.replace(".jsonl", "")]
    return "__".join(parts)


def process_one_gz(args: tuple) -> dict[str, Any]:
    path_str, input_dir, output_dir, max_rows_per_jsonl, dry_run = args
    path = Path(path_str)
    input_dir_p = Path(input_dir)
    stats: dict[str, Any] = {
        "file": str(path.relative_to(input_dir_p)),
        "rows_in": 0,
        "rows_out": 0,
        "reasons": Counter(),
        "error": None,
        "out_files": [],
    }
    try:
        ds_key = dataset_key(path, input_dir_p)
        out_ds_dir = Path(output_dir) / "by_dataset" / ds_key
        buffer: list[str] = []
        part_idx = 0
        shard_base = path.name.replace(".jsonl.gz", "").replace(".gz", "")

        def flush() -> None:
            nonlocal buffer, part_idx
            if not buffer:
                return
            part_idx += 1
            if dry_run:
                buffer = []
                return
            out_ds_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_ds_dir / f"{shard_base}_sft_{part_idx:04d}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(buffer))
                f.write("\n")
            stats["out_files"].append(str(out_path))
            buffer = []

        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stats["rows_in"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    stats["reasons"]["skip_bad_json"] += 1
                    continue
                rec, reason = convert_row(row)
                if reason.startswith("ok"):
                    stats["reasons"]["ok"] += 1
                else:
                    stats["reasons"][reason.split(":", 1)[0]] += 1
                if rec is None:
                    continue
                buffer.append(json.dumps(rec, ensure_ascii=False))
                stats["rows_out"] += 1
                if len(buffer) >= max_rows_per_jsonl:
                    flush()
        flush()
    except Exception:
        stats["error"] = traceback.format_exc(limit=10)
    return stats


def write_report(output_dir: Path, all_stats: list[dict], elapsed: float, source: str) -> None:
    reasons: Counter = Counter()
    total_in = total_out = errors = 0
    per: dict[str, Any] = {}
    for st in all_stats:
        total_in += st.get("rows_in", 0)
        total_out += st.get("rows_out", 0)
        if st.get("error"):
            errors += 1
        for k, v in (st.get("reasons") or {}).items():
            reasons[k] += int(v)
        per[st.get("file", "?")] = {
            "rows_in": st.get("rows_in", 0),
            "rows_out": st.get("rows_out", 0),
            "out_files": len(st.get("out_files") or []),
            "error": st.get("error"),
            "reasons": dict(st.get("reasons") or {}),
        }
    report = {
        "source": source,
        "output_dir": str(output_dir),
        "elapsed_sec": round(elapsed, 2),
        "files_processed": len(all_stats),
        "files_errored": errors,
        "rows_in": total_in,
        "rows_out": total_out,
        "keep_rate": round(total_out / total_in, 4) if total_in else 0.0,
        "reason_counts": dict(reasons.most_common()),
        "per_file": per,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "conversion_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 4))))
    ap.add_argument("--max-rows-per-jsonl", type=int, default=200_000)
    ap.add_argument("--limit-files", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

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
    log = logging.getLogger("jsonl_gz_sft")

    files = list_gz_files(input_dir)
    done_path = output_dir / "done_files.txt"
    if args.resume and done_path.exists():
        done = {ln.strip() for ln in done_path.read_text().splitlines() if ln.strip()}
        files = [f for f in files if str(f.relative_to(input_dir)) not in done]
    # small first for visible ETA
    files = sorted(files, key=lambda p: p.stat().st_size)
    if args.limit_files:
        files = files[: args.limit_files]

    log.info("Input: %s (%d jsonl.gz)", input_dir, len(files))
    log.info("Output: %s | workers=%d", output_dir, args.workers)
    if not files:
        log.warning("No files")
        return 0

    t0 = time.time()
    all_stats: list[dict] = []
    tasks = [
        (str(f), str(input_dir), str(output_dir), args.max_rows_per_jsonl, args.dry_run)
        for f in files
    ]
    done_n = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one_gz, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            st = fut.result()
            all_stats.append(st)
            done_n += 1
            if st.get("error"):
                log.warning("Error %s: %s", st.get("file"), str(st["error"]).splitlines()[-1][:200])
            elif not args.dry_run:
                with open(done_path, "a", encoding="utf-8") as df:
                    df.write(st["file"] + "\n")
            if done_n % 5 == 0 or done_n == len(tasks):
                elapsed = max(time.time() - t0, 1e-6)
                rin = sum(s.get("rows_in", 0) for s in all_stats)
                rout = sum(s.get("rows_out", 0) for s in all_stats)
                rate = done_n / elapsed
                eta = (len(tasks) - done_n) / rate if rate else 0
                log.info(
                    "Progress %d/%d (%.1f%%) | in=%d out=%d | %.2f files/s | ETA ~%.1f min",
                    done_n,
                    len(tasks),
                    100 * done_n / len(tasks),
                    rin,
                    rout,
                    rate,
                    eta / 60,
                )
                with open(output_dir / "progress.json", "w", encoding="utf-8") as pf:
                    json.dump(
                        {
                            "done_files": done_n,
                            "total_files": len(tasks),
                            "rows_in": rin,
                            "rows_out": rout,
                            "eta_min": round(eta / 60, 1),
                        },
                        pf,
                    )

    write_report(output_dir, all_stats, time.time() - t0, str(input_dir))
    log.info(
        "DONE in %.1fs | files=%d rows_in=%d rows_out=%d errors=%d",
        time.time() - t0,
        len(all_stats),
        sum(s.get("rows_in", 0) for s in all_stats),
        sum(s.get("rows_out", 0) for s in all_stats),
        sum(1 for s in all_stats if s.get("error")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
