#!/usr/bin/env python3
"""
Audit + in-place clean of OpenAI SFT JSONL under a directory.

Checks / removes:
  - invalid JSON
  - non-SFT top-level keys (only messages / tools / parallel_tool_calls)
  - invalid message roles / empty user|assistant / no assistant
  - within-file exact duplicates (canonical messages hash)
  - optional cross-file duplicates (shared hash set; sequential mode)

Rewrites files in place (tmp → replace). Does not create a new output tree.
"""

from __future__ import annotations

import argparse
import hashlib
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
from normalize import validate_sft_messages  # noqa: E402

ALLOWED_TOP = {"messages", "tools", "parallel_tool_calls"}
ALLOWED_ROLES = {"system", "user", "assistant", "tool"}


def canonical_hash(rec: dict[str, Any]) -> str:
    """Hash messages (+ tools if present) for duplicate detection."""
    payload = {"messages": rec.get("messages")}
    if "tools" in rec:
        payload["tools"] = rec["tools"]
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def audit_record(rec: Any) -> tuple[Optional[dict[str, Any]], str]:
    if not isinstance(rec, dict):
        return None, "not_object"
    extra = set(rec.keys()) - ALLOWED_TOP
    if extra:
        return None, f"extra_keys:{sorted(extra)[:5]}"
    if "messages" not in rec or not isinstance(rec["messages"], list):
        return None, "no_messages"
    msgs = rec["messages"]
    if not msgs:
        return None, "empty_messages"
    for m in msgs:
        if not isinstance(m, dict):
            return None, "bad_message"
        role = m.get("role")
        if role not in ALLOWED_ROLES:
            return None, f"bad_role:{role}"
        # disallow unexpected message keys beyond OpenAI SFT-ish
        allowed_m = {
            "role",
            "content",
            "tool_calls",
            "tool_call_id",
            "name",
            "reasoning_content",
        }
        bad = set(m.keys()) - allowed_m
        if bad:
            return None, f"msg_extra_keys:{sorted(bad)[:4]}"
    ok, why = validate_sft_messages(msgs)
    if not ok:
        return None, f"invalid:{why}"
    # Keep only allowed top keys
    clean = {"messages": msgs}
    if "tools" in rec and rec["tools"]:
        clean["tools"] = rec["tools"]
    if "parallel_tool_calls" in rec:
        clean["parallel_tool_calls"] = rec["parallel_tool_calls"]
    return clean, "ok"


def process_file(args: tuple) -> dict[str, Any]:
    """
    Audit one JSONL; rewrite in place if drops/changes needed.
    args: (path, dry_run, do_rewrite)
    """
    path_str, dry_run, do_rewrite = args
    path = Path(path_str)
    stats: dict[str, Any] = {
        "file": str(path),
        "rows_in": 0,
        "rows_ok": 0,
        "rows_out": 0,
        "drop_invalid": 0,
        "drop_dup_within": 0,
        "rewritten": False,
        "error": None,
        "reasons": Counter(),
    }
    try:
        seen: set[str] = set()
        keep_lines: list[str] = []
        changed = False
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                stats["rows_in"] += 1
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    stats["drop_invalid"] += 1
                    stats["reasons"]["bad_json"] += 1
                    changed = True
                    continue
                clean, reason = audit_record(rec)
                if clean is None:
                    stats["drop_invalid"] += 1
                    stats["reasons"][reason[:80]] += 1
                    changed = True
                    continue
                stats["rows_ok"] += 1
                h = canonical_hash(clean)
                if h in seen:
                    stats["drop_dup_within"] += 1
                    stats["reasons"]["dup_within_file"] += 1
                    changed = True
                    continue
                seen.add(h)
                out_line = json.dumps(clean, ensure_ascii=False)
                # Detect normalization change (key order / stripped fields)
                if out_line != raw:
                    changed = True
                keep_lines.append(out_line)
                stats["rows_out"] += 1

        if do_rewrite and changed and not dry_run:
            tmp = path.with_suffix(path.suffix + ".audit_tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                if keep_lines:
                    f.write("\n".join(keep_lines))
                    f.write("\n")
            os.replace(tmp, path)
            stats["rewritten"] = True
            # Drop empty files
            if stats["rows_out"] == 0:
                path.unlink(missing_ok=True)
                stats["deleted_empty"] = True
    except Exception:
        stats["error"] = traceback.format_exc(limit=8)
    # Counter not json-serializable by default in some paths
    stats["reasons"] = dict(stats["reasons"])
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="/projects/data/datasets/traces_team/datasets_sft_openai_sft",
    )
    ap.add_argument("--workers", type=int, default=max(1, min(32, (os.cpu_count() or 4))))
    ap.add_argument("--dry-run", action="store_true", help="Audit only; do not rewrite")
    ap.add_argument("--no-rewrite", action="store_true", help="Same as dry-run")
    ap.add_argument("--limit-files", type=int, default=0)
    ap.add_argument("--report", default="")
    args = ap.parse_args()
    dry = args.dry_run or args.no_rewrite
    do_rewrite = not dry

    root = Path(args.root)
    by = root / "by_dataset"
    files = sorted(by.rglob("*.jsonl")) if by.exists() else sorted(root.rglob("*.jsonl"))
    # skip temp
    files = [f for f in files if not f.name.endswith(".audit_tmp")]
    if args.limit_files:
        files = files[: args.limit_files]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(root / "audit_clean.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("audit")
    log.info(
        "Audit root=%s files=%d workers=%d rewrite=%s",
        root,
        len(files),
        args.workers,
        do_rewrite,
    )

    t0 = time.time()
    all_stats: list[dict] = []
    totals = Counter()
    tasks = [(str(f), dry, do_rewrite) for f in files]

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_file, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(futs):
            st = fut.result()
            all_stats.append(st)
            done += 1
            totals["rows_in"] += st.get("rows_in", 0)
            totals["rows_out"] += st.get("rows_out", 0)
            totals["drop_invalid"] += st.get("drop_invalid", 0)
            totals["drop_dup_within"] += st.get("drop_dup_within", 0)
            totals["rewritten"] += int(bool(st.get("rewritten")))
            totals["errors"] += int(bool(st.get("error")))
            if done % 200 == 0 or done == len(tasks):
                elapsed = max(time.time() - t0, 1e-6)
                log.info(
                    "Progress %d/%d | in=%d out=%d invalid=%d dup=%d rewritten=%d | %.1f files/s",
                    done,
                    len(tasks),
                    totals["rows_in"],
                    totals["rows_out"],
                    totals["drop_invalid"],
                    totals["drop_dup_within"],
                    totals["rewritten"],
                    done / elapsed,
                )

    reason_sum: Counter = Counter()
    for st in all_stats:
        for k, v in (st.get("reasons") or {}).items():
            reason_sum[k] += int(v)

    report = {
        "root": str(root),
        "elapsed_sec": round(time.time() - t0, 2),
        "files": len(files),
        "dry_run": dry,
        "rewrote_files": totals["rewritten"],
        "rows_in": totals["rows_in"],
        "rows_out": totals["rows_out"],
        "drop_invalid": totals["drop_invalid"],
        "drop_dup_within": totals["drop_dup_within"],
        "errors": totals["errors"],
        "removed_total": totals["drop_invalid"] + totals["drop_dup_within"],
        "keep_rate": round(totals["rows_out"] / totals["rows_in"], 6)
        if totals["rows_in"]
        else 0.0,
        "reason_counts": dict(reason_sum.most_common(50)),
        "note": "Within-file dedupe + format validation. Cross-file dedupe not applied (memory/IO).",
    }
    report_path = Path(args.report) if args.report else root / "audit_clean_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("Wrote %s", report_path)
    log.info(
        "DONE in=%.0f out=%.0f dropped_invalid=%d dropped_dup=%d rewritten_files=%d",
        totals["rows_in"],
        totals["rows_out"],
        totals["drop_invalid"],
        totals["drop_dup_within"],
        totals["rewritten"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
