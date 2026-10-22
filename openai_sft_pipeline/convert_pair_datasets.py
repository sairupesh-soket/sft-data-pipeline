#!/usr/bin/env python3
"""
Convert input/output (and original_text/translation) pair datasets → OpenAI SFT JSONL.

Targets:
  1) RL_reference_data_from_SFT  (parquet: input/output)
  2) code_math_text_translations (HF arrow: original_text/translation)

Same rules as convert_to_openai_sft.py:
  - OpenAI messages format (single-turn → user + assistant)
  - Drop vendor-centric system prompts
  - Scrub model/company name mentions from content
  - Strip non-SFT metadata (language, etc.)

Outputs kept separate under traces_team/.
"""

from __future__ import annotations

import argparse
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

import pyarrow.ipc as ipc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_to_openai_sft import convert_row  # noqa: E402

RL_SRC = Path("/projects/data/datasets/code_data/codeLLM_data/RL_reference_data_from_SFT")
TRANS_SRC = Path("/projects/data/datasets/code_data/codeLLM_data/code_math_text_translations")
RL_OUT = Path("/projects/data/datasets/traces_team/RL_reference_data_from_SFT_openai_sft")
TRANS_OUT = Path("/projects/data/datasets/traces_team/code_math_text_translations_openai_sft")

SKIP_TRANS_DIR_SUBSTR = ("_OLD_REMOVE", "_copy", "checkpoints")


def _setup_log() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("pair_sft")


def _flush_buffer(
    buffer: list[str],
    out_dir: Path,
    shard_base: str,
    part_idx: int,
    dry_run: bool,
) -> tuple[int, Optional[Path]]:
    if not buffer:
        return part_idx, None
    part_idx += 1
    if dry_run:
        return part_idx, None
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{shard_base}_sft_{part_idx:04d}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(buffer))
        f.write("\n")
    return part_idx, out_path


def convert_parquet_row_group(
    path_str: str,
    rg_idx: int,
    output_dir: str,
    ds_key: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Convert one parquet row-group → one JSONL shard (for parallel speed)."""
    path = Path(path_str)
    stats: dict[str, Any] = {
        "file": f"{path.name}#rg{rg_idx}",
        "ds_key": ds_key,
        "rows_in": 0,
        "rows_out": 0,
        "reasons": Counter(),
        "error": None,
        "out_files": [],
    }
    try:
        pf = pq.ParquetFile(str(path))
        table = pf.read_row_group(rg_idx)
        rows = table.to_pylist()
        stats["rows_in"] = len(rows)
        lines: list[str] = []
        for row in rows:
            rec, reason = convert_row(row)
            if reason.startswith("ok"):
                stats["reasons"]["ok"] += 1
            else:
                stats["reasons"][reason.split(":", 1)[0]] += 1
            if rec is None:
                continue
            lines.append(json.dumps(rec, ensure_ascii=False))
            stats["rows_out"] += 1
        if lines and not dry_run:
            out_ds_dir = Path(output_dir) / "by_dataset" / ds_key
            out_ds_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_ds_dir / f"{path.stem}_sft_rg{rg_idx:04d}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
                f.write("\n")
            stats["out_files"].append(str(out_path))
    except Exception:
        stats["error"] = traceback.format_exc(limit=10)
    return stats


def convert_parquet_file(
    path_str: str,
    output_dir: str,
    ds_key: str,
    max_rows_per_jsonl: int,
    dry_run: bool,
) -> dict[str, Any]:
    """Sequential fallback for smaller parquets (single process)."""
    path = Path(path_str)
    stats: dict[str, Any] = {
        "file": path.name,
        "ds_key": ds_key,
        "rows_in": 0,
        "rows_out": 0,
        "reasons": Counter(),
        "error": None,
        "out_files": [],
    }
    try:
        pf = pq.ParquetFile(str(path))
        out_ds_dir = Path(output_dir) / "by_dataset" / ds_key
        buffer: list[str] = []
        part_idx = 0
        shard_base = path.stem

        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg)
            rows = table.to_pylist()
            stats["rows_in"] += len(rows)
            for row in rows:
                rec, reason = convert_row(row)
                bucket = reason.split(":", 1)[0] if reason else "unknown"
                if reason.startswith("ok"):
                    stats["reasons"]["ok"] += 1
                else:
                    stats["reasons"][bucket] += 1
                if rec is None:
                    continue
                buffer.append(json.dumps(rec, ensure_ascii=False))
                stats["rows_out"] += 1
                if len(buffer) >= max_rows_per_jsonl:
                    part_idx, outp = _flush_buffer(
                        buffer, out_ds_dir, shard_base, part_idx, dry_run
                    )
                    buffer = []
                    if outp:
                        stats["out_files"].append(str(outp))
        if buffer:
            part_idx, outp = _flush_buffer(buffer, out_ds_dir, shard_base, part_idx, dry_run)
            if outp:
                stats["out_files"].append(str(outp))
    except Exception:
        stats["error"] = traceback.format_exc(limit=10)
    return stats


def _read_arrow_table(path: Path):
    with open(path, "rb") as f:
        try:
            return ipc.open_file(f).read_all()
        except Exception:
            f.seek(0)
            return ipc.open_stream(f).read_all()


def convert_arrow_dir(
    dir_str: str,
    output_dir: str,
    max_rows_per_jsonl: int,
    dry_run: bool,
) -> dict[str, Any]:
    d = Path(dir_str)
    stats: dict[str, Any] = {
        "file": d.name,
        "ds_key": d.name,
        "rows_in": 0,
        "rows_out": 0,
        "reasons": Counter(),
        "error": None,
        "out_files": [],
    }
    try:
        arrows = sorted(d.glob("data-*.arrow"))
        if not arrows:
            stats["error"] = "no_data_arrow"
            return stats
        out_ds_dir = Path(output_dir) / "by_dataset" / d.name
        buffer: list[str] = []
        part_idx = 0
        shard_base = d.name

        for ap in arrows:
            table = _read_arrow_table(ap)
            # Stream in batches to limit peak memory
            batch_size = 5000
            n = table.num_rows
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = table.slice(start, end - start)
                rows = batch.to_pylist()
                stats["rows_in"] += len(rows)
                for row in rows:
                    # Drop metadata fields not part of SFT before convert
                    # (language is ignored by qa pair extractor anyway)
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
                        part_idx, outp = _flush_buffer(
                            buffer, out_ds_dir, shard_base, part_idx, dry_run
                        )
                        buffer = []
                        if outp:
                            stats["out_files"].append(str(outp))
        if buffer:
            part_idx, outp = _flush_buffer(buffer, out_ds_dir, shard_base, part_idx, dry_run)
            if outp:
                stats["out_files"].append(str(outp))
    except Exception:
        stats["error"] = traceback.format_exc(limit=10)
    return stats


def list_rl_jobs() -> list[tuple[str, str, str]]:
    """(parquet_path, output_dir, ds_key)"""
    jobs = []
    mapping = {
        "sft_stage2pt1_py_code_generation": "opencodeinstruct_translation_sft2.parquet",
        "sft_stage2_py_code_generation": "py_code_generation_educational_instruct_sft_stage2.parquet",
        "sft_stage3_data_analysis_visualization": "viscode_200k_translated.parquet",
    }
    for stage, fname in mapping.items():
        p = RL_SRC / stage / fname
        if p.exists():
            jobs.append((str(p), str(RL_OUT), stage))
    return jobs


def list_trans_dirs() -> list[Path]:
    out = []
    for p in sorted(TRANS_SRC.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        if any(s in name for s in SKIP_TRANS_DIR_SUBSTR):
            continue
        if not list(p.glob("data-*.arrow")):
            continue
        out.append(p)
    return out


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
        key = st.get("ds_key") or st.get("file")
        per[key] = {
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
        "per_dataset": per,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "conversion_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def _merge_stats(parts: list[dict[str, Any]], ds_key: str, file_name: str) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "file": file_name,
        "ds_key": ds_key,
        "rows_in": 0,
        "rows_out": 0,
        "reasons": Counter(),
        "error": None,
        "out_files": [],
    }
    errors = []
    for st in parts:
        merged["rows_in"] += st.get("rows_in", 0)
        merged["rows_out"] += st.get("rows_out", 0)
        for k, v in (st.get("reasons") or {}).items():
            merged["reasons"][k] += int(v)
        merged["out_files"].extend(st.get("out_files") or [])
        if st.get("error"):
            errors.append(st["error"])
    if errors:
        merged["error"] = (
            errors[0] if len(errors) == 1 else f"{len(errors)} rg errors; first={errors[0]}"
        )
    return merged


def run_rl(
    workers: int,
    max_rows: int,
    dry_run: bool,
    log: logging.Logger,
    only_missing: bool = True,
) -> None:
    jobs = list_rl_jobs()
    log.info("RL jobs: %d → %s", len(jobs), RL_OUT)
    RL_OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_stats: list[dict] = []

    for path_str, out_dir, ds_key in jobs:
        out_ds = Path(out_dir) / "by_dataset" / ds_key
        if only_missing and out_ds.exists() and any(out_ds.glob("*.jsonl")):
            n_files = len(list(out_ds.glob("*.jsonl")))
            log.info("RL skip existing %s (%d jsonl)", ds_key, n_files)
            all_stats.append(
                {
                    "file": Path(path_str).name,
                    "ds_key": ds_key,
                    "rows_in": 0,
                    "rows_out": 0,
                    "reasons": Counter({"skipped_existing": 1}),
                    "error": None,
                    "out_files": [str(p) for p in out_ds.glob("*.jsonl")],
                }
            )
            continue

        pf = pq.ParquetFile(path_str)
        n_rg = pf.num_row_groups
        n_rows = pf.metadata.num_rows if pf.metadata else 0
        if n_rg >= 4 and n_rows >= 500_000:
            log.info(
                "RL parallel RGs %s rows=%s rgs=%s workers=%s",
                ds_key,
                n_rows,
                n_rg,
                workers,
            )
            parts: list[dict] = []
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {
                    ex.submit(
                        convert_parquet_row_group, path_str, rg, out_dir, ds_key, dry_run
                    ): rg
                    for rg in range(n_rg)
                }
                done = 0
                for fut in as_completed(futs):
                    st = fut.result()
                    parts.append(st)
                    done += 1
                    if done % 50 == 0 or done == n_rg:
                        log.info(
                            "RL %s progress %d/%d out_so_far=%s",
                            ds_key,
                            done,
                            n_rg,
                            sum(p.get("rows_out", 0) for p in parts),
                        )
            st = _merge_stats(parts, ds_key, Path(path_str).name)
            all_stats.append(st)
            log.info(
                "RL done %s in=%s out=%s err=%s",
                ds_key,
                st.get("rows_in"),
                st.get("rows_out"),
                bool(st.get("error")),
            )
        else:
            st = convert_parquet_file(path_str, out_dir, ds_key, max_rows, dry_run)
            all_stats.append(st)
            log.info(
                "RL done %s in=%s out=%s err=%s",
                ds_key,
                st.get("rows_in"),
                st.get("rows_out"),
                bool(st.get("error")),
            )

    write_report(RL_OUT, all_stats, time.time() - t0, str(RL_SRC))
    log.info("RL complete in %.1fs", time.time() - t0)


def run_trans(workers: int, max_rows: int, dry_run: bool, log: logging.Logger) -> None:
    dirs = list_trans_dirs()
    log.info("Translation dirs: %d → %s", len(dirs), TRANS_OUT)
    TRANS_OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_stats: list[dict] = []
    tasks = [(str(d), str(TRANS_OUT), max_rows, dry_run) for d in dirs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(convert_arrow_dir, *t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(futs):
            st = fut.result()
            all_stats.append(st)
            done += 1
            if done % 10 == 0 or done == len(tasks):
                log.info(
                    "TRANS progress %d/%d last=%s in=%s out=%s",
                    done,
                    len(tasks),
                    st.get("file"),
                    st.get("rows_in"),
                    st.get("rows_out"),
                )
    write_report(TRANS_OUT, all_stats, time.time() - t0, str(TRANS_SRC))
    log.info("TRANS complete in %.1fs", time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["rl", "trans", "both"], default="both")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4))))
    ap.add_argument("--max-rows-per-jsonl", type=int, default=200_000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    log = _setup_log()

    if args.which in ("rl", "both"):
        run_rl(args.workers, args.max_rows_per_jsonl, args.dry_run, log)
    if args.which in ("trans", "both"):
        run_trans(args.workers, args.max_rows_per_jsonl, args.dry_run, log)


if __name__ == "__main__":
    main()
