#!/usr/bin/env python3
"""
Extract penant-data Langfuse sessions (agent only) in session format.

- Project: penant-data
- Keep only sessions whose id ends with ``-agent`` (skip ``-human`` / ``-conversation``)
- Do not flatten/mix traces across sessions — one session JSON each
- Includes system-prompt traces even when they have input but no output
- Parallel workers for hydrate/extract

Env:
  LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

# Reuse observation/trace shaping from llm_logging
sys.path.insert(0, str(Path.home() / "traces-extraction"))
import llm_logging as L  # noqa: E402


DEFAULT_BASE = "http://soketlab-worker003:30395"
DEFAULT_OUT = "/projects/data/datasets/traces_team/penant-data"
DEFAULT_FROM = "2026-10-03T16:07:48Z"


def configure_langfuse(base_url: str, public_key: str, secret_key: str) -> None:
    L.BASE_URL = base_url.rstrip("/")
    L.PUBLIC_KEY = public_key
    L.SECRET_KEY = secret_key
    L.HTTP_TIMEOUT = 45
    L.MAX_RETRIES = 6
    L.SLEEP = 0.02
    L.OUTPUT_DIR = ""  # unused; we write ourselves


def is_agent_session(session_id: str) -> bool:
    sid = (session_id or "").strip().lower()
    if sid.endswith("-human") or sid.endswith("-conversation"):
        return False
    return sid.endswith("-agent")


def list_agent_sessions(from_ts: str, to_ts: str | None, list_path: Path) -> list[dict]:
    page = 1
    kept: list[dict] = []
    list_path.parent.mkdir(parents=True, exist_ok=True)
    with list_path.open("w", encoding="utf-8") as fout:
        while True:
            params = {"fromTimestamp": from_ts, "limit": 100, "page": page}
            if to_ts:
                params["toTimestamp"] = to_ts
            data = L.api_get("/api/public/sessions", params)
            if not isinstance(data, dict):
                break
            batch = data.get("data") or []
            if not batch:
                break
            for listed in batch:
                sid = listed.get("id") or ""
                if not is_agent_session(sid):
                    continue
                kept.append(listed)
                fout.write(json.dumps(listed, ensure_ascii=False) + "\n")
            meta = data.get("meta") or {}
            print(
                f"  list page {page}/{meta.get('totalPages', '?')} "
                f"| agent_sessions={len(kept)}",
                flush=True,
            )
            if page >= int(meta.get("totalPages") or page):
                break
            page += 1
            time.sleep(L.SLEEP)
    return kept


def hydrate_trace(stub: dict, session_id: str) -> dict | None:
    """Fetch full trace; keep system-prompt traces even without output."""
    tid = stub.get("id")
    if not tid:
        return None
    name = (stub.get("name") or "").lower()
    is_system = "system prompt" in name

    # Fast path if stub already complete
    if L.is_complete_trace(stub) and L.observations_are_full(stub.get("observations")):
        full = stub
    else:
        full = L.api_get(f"/api/public/traces/{tid}")
        time.sleep(L.SLEEP)
        if not isinstance(full, dict):
            return None

    if not full.get("sessionId"):
        full["sessionId"] = session_id or stub.get("sessionId")

    if is_system:
        # Keep input-only system prompts
        if not L.has_content(full.get("input")) and not L.has_content(stub.get("input")):
            return None
        et = L.extract_trace(full)
        if not et.get("chat_input") and full.get("input") is not None:
            et["chat_input"] = full.get("input")
        return et

    if not L.is_complete_trace(full):
        return None
    return L.extract_trace(full)


def extract_one_session(listed: dict, out_dir: Path) -> dict:
    sid = listed.get("id")
    if not sid:
        return {"status": "skip", "reason": "no_id"}

    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(sid))
    out_path = out_dir / f"session-{safe}.json"
    if out_path.exists() and out_path.stat().st_size > 0:
        return {"status": "exists", "session_id": sid, "path": str(out_path)}

    sess = L.api_get(f"/api/public/sessions/{sid}")
    time.sleep(L.SLEEP)
    if not isinstance(sess, dict):
        return {"status": "error", "session_id": sid, "reason": "session_fetch_failed"}

    stubs = sess.get("traces") or []
    traces: list[dict] = []
    incomplete = 0
    for stub in stubs:
        if not isinstance(stub, dict):
            incomplete += 1
            continue
        if not stub.get("sessionId"):
            stub["sessionId"] = sid
        et = hydrate_trace(stub, sid)
        if et is None:
            incomplete += 1
            continue
        traces.append(et)

    if not traces:
        return {
            "status": "empty",
            "session_id": sid,
            "incomplete": incomplete,
            "stubs": len(stubs),
        }

    # Chronological order within the session
    traces.sort(key=lambda t: str(t.get("timestamp") or ""))
    timestamps = [t.get("timestamp") for t in traces if t.get("timestamp")]
    created_at = min(timestamps) if timestamps else (
        sess.get("createdAt") or listed.get("createdAt")
    )
    user_id = next((t.get("user_id") for t in traces if t.get("user_id")), None)

    extracted = {
        "session_id": sid,
        "session_name": next(
            (t.get("trace_name") for t in traces if t.get("trace_name")), sid
        ),
        "source": "penant-data",
        "created_at": created_at,
        "user_id": user_id,
        "environment": sess.get("environment") or listed.get("environment"),
        "trace_count": len(traces),
        "traces": traces,
    }
    out_path.write_text(L.safe_json_dumps(extracted, indent=2), encoding="utf-8")
    return {
        "status": "ok",
        "session_id": sid,
        "path": str(out_path),
        "trace_count": len(traces),
        "incomplete": incomplete,
        "stubs": len(stubs),
    }


def write_jsonl_rollup(out_dir: Path, jsonl_path: Path) -> dict:
    files = sorted(out_dir.glob("session-*.json"))
    n_traces = 0
    with jsonl_path.open("w", encoding="utf-8") as fout:
        for path in files:
            sess = json.loads(path.read_text(encoding="utf-8"))
            n_traces += sess.get("trace_count") or len(sess.get("traces") or [])
            fout.write(json.dumps(sess, ensure_ascii=False) + "\n")
    return {"session_files": len(files), "total_traces": n_traces, "jsonl": str(jsonl_path)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from", dest="from_ts", default=DEFAULT_FROM)
    p.add_argument("--to", dest="to_ts", default=None)
    p.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT))
    p.add_argument("--workers", type=int, default=8, help="Parallel hydrate workers")
    p.add_argument("--limit-sessions", type=int, default=None, help="Smoke: only first N agent sessions")
    p.add_argument("--list-only", action="store_true")
    args = p.parse_args()

    public = os.environ.get("LANGFUSE_PUBLIC_KEY") or ""
    secret = os.environ.get("LANGFUSE_SECRET_KEY") or ""
    base = os.environ.get("LANGFUSE_BASE_URL") or DEFAULT_BASE
    if not public or not secret:
        raise SystemExit("Need LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in env")

    configure_langfuse(base, public, secret)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    list_path = out_dir / "agent_sessions_listed.ndjson"
    state_path = out_dir / "extract_state.json"
    jsonl_path = out_dir / "penant_agent_sessions.jsonl"
    manifest_path = out_dir / "manifest.json"

    print(f"base={base}", flush=True)
    print(f"from={args.from_ts} to={args.to_ts or 'now'}", flush=True)
    print(f"out={out_dir} workers={args.workers}", flush=True)
    print("filter: session id endswith -agent (skip -human / -conversation)", flush=True)
    L.wait_for_api(timeout_s=120)

    print("listing sessions...", flush=True)
    listed = list_agent_sessions(args.from_ts, args.to_ts, list_path)
    if args.limit_sessions:
        listed = listed[: args.limit_sessions]
    print(f"agent sessions to extract: {len(listed)}", flush=True)
    if args.list_only:
        return

    stats = {
        "ok": 0,
        "exists": 0,
        "empty": 0,
        "error": 0,
        "traces": 0,
        "incomplete_stubs": 0,
    }
    lock = Lock()
    t0 = time.time()

    def _one(item: dict) -> dict:
        # each thread uses shared L module; urllib is fine
        return extract_one_session(item, out_dir)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(_one, item): item.get("id") for item in listed}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                res = fut.result()
            except Exception as e:
                res = {"status": "error", "reason": str(e), "session_id": futs[fut]}
            with lock:
                st = res.get("status") or "error"
                stats[st] = stats.get(st, 0) + 1
                stats["traces"] += int(res.get("trace_count") or 0)
                stats["incomplete_stubs"] += int(res.get("incomplete") or 0)
                if done % 25 == 0 or done == len(futs):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed else 0
                    print(
                        f"  extract {done}/{len(futs)} ({rate:.2f}/s) "
                        f"ok={stats.get('ok',0)} exists={stats.get('exists',0)} "
                        f"empty={stats.get('empty',0)} err={stats.get('error',0)}",
                        flush=True,
                    )
                    state_path.write_text(json.dumps({"done": done, "stats": stats}, indent=2))

    print("writing jsonl rollup...", flush=True)
    rollup = write_jsonl_rollup(out_dir, jsonl_path)
    manifest = {
        "project": "penant-data",
        "source": "Langfuse",
        "base_url": base,
        "from": args.from_ts,
        "to": args.to_ts,
        "filter": "session_id endswith -agent; skip -human/-conversation",
        "format": "one session JSON with traces[] (not mixed)",
        "workers": args.workers,
        "listed_agent_sessions": len(listed),
        "stats": stats,
        "rollup": rollup,
        "output_dir": str(out_dir),
        "seconds": round(time.time() - t0, 2),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
