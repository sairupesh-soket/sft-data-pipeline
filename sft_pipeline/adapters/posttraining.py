"""Convert codeLLM posttraining_datasets into plain OpenAI chat jsonl.gz.

A_native_openai : rows that already have a valid OpenAI-style `messages` list.
B1_marker_mapped: rows mapped to messages using clear, reliable split markers.

Source data is read-only. Every output row is {"messages": [...]} plus a
top-level "tools" list when the row has tools. Role rules applied to every row:
the conversation must end with an assistant turn (trailing turns trimmed), and
no assistant turn may come directly after the system prompt (or open the chat).
"""
import os, sys, re, json, gzip, glob, collections
from multiprocessing import Pool
import pyarrow as pa, pyarrow.parquet as pq

SRC = os.environ.get("SFTPIPE_SRC", ".")   # codeLLM posttraining_datasets style folder
OUT = os.environ.get("SFTPIPE_OUT", "out")
ROLES = {"system", "user", "assistant", "tool"}

# ---------------------------------------------------------------- role rules
def finalize(msgs, tools=None):
    """Return (row, None) or (None, drop_reason)."""
    out = []
    for m in msgs:
        r = m.get("role")
        if r not in ROLES:
            return None, "bad_role"
        c = m.get("content")
        if c is not None and not isinstance(c, str):
            return None, "content_not_str"
        mm = {"role": r, "content": c}
        if m.get("tool_calls"):
            mm["tool_calls"] = m["tool_calls"]
        if r == "tool" and m.get("tool_call_id"):
            mm["tool_call_id"] = m["tool_call_id"]
        out.append(mm)
    while out and out[-1]["role"] != "assistant":
        out.pop()
    if not out:
        return None, "no_assistant_turn"
    body = out[1:] if out[0]["role"] == "system" else out
    if not body or body[0]["role"] == "assistant":
        return None, "assistant_right_after_system_or_first"
    if not any(m["role"] == "user" and (m["content"] or "").strip() for m in out):
        return None, "no_user_content"
    last = out[-1]
    if not ((last["content"] or "").strip() or last.get("tool_calls")):
        return None, "empty_final_assistant"
    row = {"messages": out}
    if tools:
        row["tools"] = tools
    return row, None

def merge_reasoning(msgs):
    res = []
    for m in msgs:
        m = dict(m)
        rc = m.pop("reasoning_content", None)
        if m.get("role") == "assistant" and isinstance(rc, str) and rc.strip():
            m["content"] = f"<think>\n{rc.strip()}\n</think>\n\n{(m.get('content') or '').strip()}"
        res.append(m)
    return res

# ------------------------------------------------ Hermes/Qwen tool-call format
RE_TOOLS = re.compile(r"<tools>\s*(\{.*?)\s*</tools>", re.S)
RE_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
RE_RESP = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.S)

def convert_tool_calling(msgs):
    tools, out, pending, n = [], [], [], 0
    for m in msgs:
        role, c = m.get("role"), m.get("content") or ""
        if role == "system":
            tm = RE_TOOLS.search(c)
            if tm:
                for line in tm.group(1).splitlines():
                    if line.strip():
                        tools.append(json.loads(line))
                c = c.split("\n\n# Tools", 1)[0].strip()
            if c:
                out.append({"role": "system", "content": c})
        elif role == "assistant":
            calls = []
            for raw in RE_CALL.findall(c):
                obj = json.loads(raw)
                args = obj.get("arguments", {})
                n += 1
                calls.append({"id": f"call_{n}", "type": "function",
                              "function": {"name": obj["name"],
                                           "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)}})
            text = RE_CALL.sub("", c).strip()
            mm = {"role": "assistant", "content": text if text else None}
            if calls:
                mm["tool_calls"] = calls
            pending = [x["id"] for x in calls]
            out.append(mm)
        elif role == "tool":
            parts = RE_RESP.findall(c) or [c.strip()]
            for p in parts:
                if not pending:
                    raise ValueError("tool response without matching call")
                out.append({"role": "tool", "tool_call_id": pending.pop(0), "content": p})
        else:
            out.append({"role": role, "content": c})
    return out, tools

# ------------------------------------------------------- text-blob splitters
RE_EXTRA = re.compile(r"<extra_id_\d+>(System|User|Assistant)\n?")
RE_QSTART = re.compile(r"^\s*(Problem|Question)\s*:\s*", re.I)
RE_QA = re.compile(r"\n\s*(Solution|Answer)\s*:\s*")

def split_text(ds, t):
    """Return (messages, rule) or (None, reason)."""
    if t.startswith("<think> </think>") or t.startswith("<think></think>"):
        t = t.split("</think>", 1)[1].lstrip()
    if "<extra_id_1>User" in t and "<extra_id_1>Assistant" in t:
        parts = RE_EXTRA.split(t)
        msgs = []
        for i in range(1, len(parts), 2):
            body = parts[i + 1].strip()
            role = parts[i].lower()
            if role == "system" and not body:
                continue
            msgs.append({"role": role, "content": body})
        return msgs, "extra_id"
    if "<｜User｜>" in t and "<｜Assistant｜>" in t:
        t2 = t.replace("<｜begin▁of▁sentence｜>", "").replace("<｜end▁of▁sentence｜>", "")
        parts = re.split(r"<｜(User|Assistant)｜>", t2)
        msgs = []
        if parts[0].strip():
            msgs.append({"role": "system", "content": parts[0].strip()})
        for i in range(1, len(parts), 2):
            msgs.append({"role": parts[i].lower(), "content": parts[i + 1].strip()})
        return msgs, "deepseek_tokens"
    if t.startswith("input:") and " output:" in t:
        i = t.find(" output:")
        return [{"role": "user", "content": t[6:i].strip()},
                {"role": "assistant", "content": t[i + 8:].strip()}], "input_output"
    qm = RE_QSTART.match(t)
    if qm:
        am = RE_QA.search(t, qm.end())
        if am:
            return [{"role": "user", "content": t[qm.end():am.start()].strip()},
                    {"role": "assistant", "content": t[am.end():].strip()}], "question_answer"
    i = t.find("<think>")
    if i > 0:
        a = t[i:].strip()
        if "</think>" not in a:
            return None, "truncated_think_not_closed"
        if not a.split("</think>", 1)[1].strip():
            return None, "no_final_answer_after_think"
        return [{"role": "user", "content": t[:i].strip()},
                {"role": "assistant", "content": a}], "prompt_think"
    if "Scientific-Coding" in ds:
        j = t.find("RESPONSE GUIDELINES:")
        k = t.find("\n```", j) if j >= 0 else -1
        if k > 0:
            return [{"role": "user", "content": t[:k].strip()},
                    {"role": "assistant", "content": t[k:].strip()}], "sci_coding_guidelines"
    return None, "not_B1"

DEEPMIND_LANG = {3: "python", 2: "cpp", 4: "java", 1: "python"}  # preference order below
DEEPMIND_ORDER = [3, 2, 4, 1]

# ------------------------------------------------------------ row iterators
def iter_jsonl(p):
    with open(p, "rb") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)

def iter_parquet(p, cols=None):
    for b in pq.ParquetFile(p).iter_batches(batch_size=2000, columns=cols):
        yield from b.to_pylist()

def iter_arrow_text(p):
    with pa.memory_map(p) as src:
        for b in pa.ipc.open_stream(src):
            yield from b.column(b.schema.get_field_index("text")).to_pylist()

# ------------------------------------------------------------------- jobs
def build_jobs():
    J = []
    def add(bucket, kind, p):
        rel = os.path.relpath(p, SRC)
        base = re.sub(r"\.(jsonl|json|parquet|arrow)$", "", rel)
        J.append((bucket, kind, p, os.path.join(OUT, bucket, base + ".jsonl.gz")))
    for p in glob.glob(SRC + "/nemotron_SFT/cascade_sft_stage1/**/*.jsonl", recursive=True) + \
             glob.glob(SRC + "/nemotron_SFT/cascade_sft_stage2/**/*.jsonl", recursive=True):
        add("A_native_openai", "tool_calling" if p.endswith("tool_calling.jsonl") else "messages", p)
    for p in glob.glob(SRC + "/nemotron_SFT/science_v1/**/*.jsonl", recursive=True) + \
             glob.glob(SRC + "/nemotron_SFT/math_proofs_v1/**/*.jsonl", recursive=True):
        add("A_native_openai", "messages_reasoning", p)
    for p in glob.glob(SRC + "/open-r1-CoT/codeforces-cots/**/*.parquet", recursive=True):
        add("A_native_openai", "parquet_messages", p)
    add("A_native_openai", "rm_chosen", SRC + "/nemotron_RL/cascade_RM_Training/rm_training_data.jsonl")
    for p in glob.glob(SRC + "/KodCode_SFT/**/*.parquet", recursive=True):
        add("B1_marker_mapped", "kodcode_sft", p)
    for p in glob.glob(SRC + "/KodCode_RL/**/*.parquet", recursive=True):
        add("B1_marker_mapped", "kodcode_rl", p)
    for p in glob.glob(SRC + "/deepmind-sft/code_contests/**/*.parquet", recursive=True):
        add("B1_marker_mapped", "deepmind", p)
    for p in glob.glob(SRC + "/opencoder_SFT/*/*/*.arrow"):
        add("B1_marker_mapped", "opencoder", p)
    for p in glob.glob(SRC + "/nemotron_SFT/pretraining_SFT_v1/*/*.arrow"):
        add("B1_marker_mapped", "text_arrow", p)
    for p in glob.glob(SRC + "/nemotron_SFT/pretraining_specialized_SFT_v1/**/*.parquet", recursive=True):
        if "Math-Textbooks" in p or "Wiki-Rewrite" in p:
            continue
        add("B1_marker_mapped", "text_parquet", p)
    return J

def rows_for(kind, p):
    """Yield (row_or_None, reason)."""
    if kind == "messages":
        for r in iter_jsonl(p):
            yield finalize(r.get("messages") or [])
    elif kind == "messages_reasoning":
        for r in iter_jsonl(p):
            m = r.get("messages") or []
            if not m:
                yield None, "messages_empty"; continue
            yield finalize(merge_reasoning(m), r.get("tools") or None)
    elif kind == "tool_calling":
        for r in iter_jsonl(p):
            try:
                msgs, tools = convert_tool_calling(r.get("messages") or [])
            except Exception:
                yield None, "tool_parse_error"; continue
            yield finalize(msgs, tools or None)
    elif kind == "parquet_messages":
        for r in iter_parquet(p, ["messages"]):
            yield finalize(r.get("messages") or [])
    elif kind == "rm_chosen":
        for r in iter_jsonl(p):
            yield finalize(r.get("chosen") or [])
    elif kind == "kodcode_sft":
        mp = {"human": "user", "gpt": "assistant", "system": "system"}
        for r in iter_parquet(p, ["conversations"]):
            yield finalize([{"role": mp.get(c.get("from"), c.get("from")), "content": c.get("value")}
                            for c in (r.get("conversations") or [])])
    elif kind == "kodcode_rl":
        for r in iter_parquet(p, ["question", "solution"]):
            yield finalize([{"role": "user", "content": r.get("question") or ""},
                            {"role": "assistant", "content": f"```python\n{(r.get('solution') or '').strip()}\n```"}])
    elif kind == "deepmind":
        for r in iter_parquet(p, ["description", "solutions"]):
            sol = r.get("solutions") or {}
            langs, codes = sol.get("language") or [], sol.get("solution") or []
            pick = None
            for L in DEEPMIND_ORDER:
                for lg, code in zip(langs, codes):
                    if lg == L and code and code.strip():
                        pick = (DEEPMIND_LANG[L], code.strip()); break
                if pick: break
            if not pick:
                yield None, "no_solution"; continue
            yield finalize([{"role": "user", "content": r.get("description") or ""},
                            {"role": "assistant", "content": f"```{pick[0]}\n{pick[1]}\n```"}])
    elif kind == "opencoder":
        with pa.memory_map(p) as src:
            for b in pa.ipc.open_stream(src):
                for r in b.select(["instruction", "output"]).to_pylist():
                    yield finalize([{"role": "user", "content": r.get("instruction") or ""},
                                    {"role": "assistant", "content": r.get("output") or ""}])
    elif kind in ("text_arrow", "text_parquet"):
        it = iter_arrow_text(p) if kind == "text_arrow" else (r["text"] for r in iter_parquet(p, ["text"]))
        for t in it:
            msgs, rule = split_text(p, t or "")
            if msgs is None:
                yield None, rule; continue
            row, why = finalize(msgs)
            yield (row, rule) if row else (None, why)

def work(job):
    bucket, kind, p, outp = job
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    tmp = outp + ".tmp"
    kept, dropped, nbytes = collections.Counter(), collections.Counter(), 0
    try:
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=3) as fo:
            for row, reason in rows_for(kind, p):
                if row is None:
                    dropped[reason] += 1
                    continue
                s = json.dumps(row, ensure_ascii=False)
                fo.write(s + "\n")
                nbytes += len(s.encode()) + 1
                kept[reason or "ok"] += 1
        os.replace(tmp, outp)
        err = None
    except Exception as e:
        err = repr(e)[:300]
    return {"bucket": bucket, "kind": kind, "src": p, "out": outp, "kept": dict(kept),
            "dropped": dict(dropped), "raw_bytes": nbytes,
            "out_bytes": os.path.getsize(outp) if err is None else 0, "error": err}

def main():
    jobs = build_jobs()
    log = os.path.join(os.environ.get("SFTPIPE_LOG", OUT.rstrip("/") + "_logs"), "convert_log.jsonl")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    done = set()
    if os.path.exists(log):
        for l in open(log):
            d = json.loads(l)
            if d["error"] is None:
                done.add(d["src"])
    jobs = [j for j in jobs if j[2] not in done]
    jobs.sort(key=lambda j: -os.path.getsize(j[2]))
    print(f"{len(jobs)} files to convert", flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 96) as pool, open(log, "a") as fo:
        for i, res in enumerate(pool.imap_unordered(work, jobs), 1):
            fo.write(json.dumps(res) + "\n"); fo.flush()
            if i % 50 == 0 or res["error"]:
                print(i, res["src"], res["error"] or "", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
