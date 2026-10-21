#!/usr/bin/env python3
"""
filter_sft.py -- DROP-ONLY quality filter for OpenAI-format SFT jsonl ({"messages": [...]}).

Never edits a row. A row is either kept BYTE-FOR-BYTE as it was read, or dropped and written
(with the reasons) to a separate rejects file. Read-only on the input directory.

Layout mirrored from input to output:   <in>/<lang>/<shard>.jsonl
    kept     -> <out>/<lang>/<shard>.jsonl          (original lines, untouched)
    rejected -> <rejects>/<lang>/<shard>.jsonl      ({"reasons": [...], "row": {...original...}})
    stats    -> <out>/_filter_stats/<lang>__<shard>.json  (counts per reason)

Usage
  python3 filter_sft.py --input DIR --output DIR --rejects DIR [--workers 48] [--resume]
  python3 filter_sft.py --input DIR --sample 3000          # dry run on random rows: prints rates + examples
  python3 filter_sft.py --self-test                        # runs the built-in unit checks

Rules (every one drops the row; reason codes in brackets)
 A. Format (must be valid OpenAI chat format)
    [bad_json] [no_messages] [bad_message] [bad_role] [content_not_string] [empty_content]
    [system_not_first] [first_not_user] [last_not_assistant] [consecutive_same_role]
 B. Think blocks (per assistant message)
    [think_empty] nothing between <think> and </think>       (the case you reported)
    [think_placeholder] only dots/punctuation between them   (the case you reported)
    [think_unbalanced] [think_nested_or_doubled] [think_without_answer]
 C. Garbage text
    [control_char] [replacement_chars] [dots_run] [char_run] [word_loop] [phrase_loop]
    [placeholder_answer] [answer_equals_question] [truncated_marker] [content_python_repr]
 D. Missing information
    [prompt_missing_input]  user turn is a bare instruction ending in ':' whose text/snippet is absent
    [tool_result_without_call]  a tool message whose tool call was lost
"""
import os, re, sys, json, glob, argparse, collections, random

ROLES = ("system", "user", "assistant", "tool")
THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.S | re.I)
CODE_BLOCK = re.compile(r"```.*?```", re.S)
TOOL_BLOCK = re.compile(r"<tool_response>.*?</tool_response>|<tool_call>.*?</tool_call>", re.S | re.I)
CALL_MARK = re.compile(r"<tool_call>|<function=|<\|python_tag\|>|\"function_call\"|<invoke\b|Action Input:", re.I)
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1a\x1c-\x1f]")   # ESC (\x1b) is allowed: terminal colour codes are legitimate
DOTS = re.compile(r"(?:\.\s){30,}\.|\.{200,}")
CHAR_RUN = re.compile(r"([^\W\d_])\1{39,}")
WORD_LOOP = re.compile(r"\b(\w+)(?:\s+\1\b){24,}", re.I)
PY_REPR = re.compile(r"^\s*\[\s*\{\s*['\"](?:text|type|content|role)['\"]\s*:")
PLACEHOLDER = {"none", "null", "n/a", "nan", "[]", "{}", "nil", "undefined"}
MISSING_INPUT = re.compile(
    r"(?:following|below|given|this|these|attached|provided)\b[^\n:]{0,60}\b(?:text|snippet|passage|document|article|paragraph|excerpt|abstract|content|context|extract|code|conversation|dialogue|review|sentence|question|story)s?\b[^\n:]{0,20}:\s*$",
    re.I)


def _think_reasons(c):
    r = []
    lo = c.lower()
    if "<think>" not in lo and "</think>" not in lo:
        return r
    depth = 0
    for t in re.finditer(r"</?think>", lo):
        depth += 1 if t.group(0) == "<think>" else -1
        if depth > 1: r.append("think_nested_or_doubled"); return r
        if depth < 0: r.append("think_unbalanced"); return r
    if depth != 0:
        r.append("think_unbalanced"); return r
    for t in THINK_BLOCK.findall(c):
        if not t.strip():
            r.append("think_empty")
        elif sum(ch.isalnum() for ch in t) < 5:
            r.append("think_placeholder")
    tail = c[lo.rfind("</think>") + 8:]
    if not tail.strip() and not CALL_MARK.search(c):
        r.append("think_without_answer")
    return r


def _loops(text):
    r = []
    body = TOOL_BLOCK.sub(" ", CODE_BLOCK.sub(" ", THINK_BLOCK.sub(" ", text)))
    if DOTS.search(body): r.append("dots_run")
    if CHAR_RUN.search(body): r.append("char_run")
    if WORD_LOOP.search(body): r.append("word_loop")
    w = body.split()
    if len(w) > 200:
        grams = collections.Counter(" ".join(w[i:i + 10]) for i in range(len(w) - 9))
        for g, top in grams.most_common(3):
            words = [t for t in g.split() if len(t) >= 2 and t.isalpha()]
            if len(words) >= 6 and top >= 8 and top * 10 >= 0.25 * len(w): r.append("phrase_loop"); break
    return r


def check_row(row):
    """Returns a list of reason codes; empty list == keep."""
    if not isinstance(row, dict): return ["no_messages"]
    m = row.get("messages")
    if not isinstance(m, list) or not m: return ["no_messages"]
    reasons = []
    for x in m:
        if not isinstance(x, dict): return ["bad_message"]
    roles = [x.get("role") for x in m]
    if any(r not in ROLES for r in roles): reasons.append("bad_role")
    for x in m:
        c = x.get("content")
        if not isinstance(c, str):
            if not (x.get("role") == "assistant" and x.get("tool_calls")): reasons.append("content_not_string"); break
        elif not c.strip(): reasons.append("empty_content"); break
    if reasons: return sorted(set(reasons))
    if "system" in roles[1:]: reasons.append("system_not_first")
    body = [r for r in roles if r != "system"]
    if not body or body[0] != "user": reasons.append("first_not_user")
    if roles[-1] != "assistant": reasons.append("last_not_assistant")
    if any(a == b and a in ("user", "assistant") for a, b in zip(roles, roles[1:])): reasons.append("consecutive_same_role")

    first_user = None
    for i, x in enumerate(m):
        c = x.get("content") or ""; role = x["role"]
        if CTRL.search(c): reasons.append("control_char")
        if c.count("\ufffd") >= 3: reasons.append("replacement_chars")
        if PY_REPR.match(c): reasons.append("content_python_repr")
        if c.rstrip().endswith("[truncated]"): reasons.append("truncated_marker")
        if role == "assistant":
            reasons += _think_reasons(c)
            reasons += _loops(c)
            if c.strip().lower() in PLACEHOLDER: reasons.append("placeholder_answer")
        elif role == "user":
            if first_user is None:
                first_user = c
        elif role == "tool":
            j = i - 1
            while j >= 0 and m[j].get("role") == "tool": j -= 1
            prev = m[j] if j >= 0 else {}
            if not (prev.get("role") == "assistant" and (CALL_MARK.search(prev.get("content") or "") or prev.get("tool_calls"))):
                reasons.append("tool_result_without_call")
    if first_user is not None:
        fu = first_user.strip()
        if len(fu) <= 200 and fu.endswith(":") and MISSING_INPUT.search(fu): reasons.append("prompt_missing_input")
        a0 = next((x.get("content") for x in m if x["role"] == "assistant"), "")
        if fu and fu == (a0 or "").strip(): reasons.append("answer_equals_question")
    return sorted(set(reasons))


def process_file(args):
    src, dst, rej, statdir, resume = args
    name = os.path.basename(src); lang = os.path.basename(os.path.dirname(src))
    stat_path = os.path.join(statdir, f"{lang}__{name}.json")
    if resume and os.path.exists(stat_path): return (lang, name, "skipped", 0, 0)
    os.makedirs(os.path.dirname(dst), exist_ok=True); os.makedirs(os.path.dirname(rej), exist_ok=True); os.makedirs(statdir, exist_ok=True)
    kept = dropped = 0; counts = collections.Counter()
    with open(src, "rb") as fi, open(dst + ".tmp", "wb") as fk, open(rej + ".tmp", "w", encoding="utf-8") as fr:
        for raw in fi:
            line = raw.strip()
            if not line: continue
            try: row = json.loads(line)
            except Exception:
                dropped += 1; counts["bad_json"] += 1
                fr.write(json.dumps({"reasons": ["bad_json"], "row": line.decode("utf-8", "replace")[:2000]}, ensure_ascii=False) + "\n"); continue
            rs = check_row(row)
            if rs:
                dropped += 1
                for r in rs: counts[r] += 1
                fr.write(json.dumps({"reasons": rs, "row": row}, ensure_ascii=False) + "\n")
            else:
                kept += 1; fk.write(raw if raw.endswith(b"\n") else raw + b"\n")
    os.replace(dst + ".tmp", dst); os.replace(rej + ".tmp", rej)
    json.dump({"file": f"{lang}/{name}", "kept": kept, "dropped": dropped, "reasons": dict(counts)}, open(stat_path, "w"))
    return (lang, name, "done", kept, dropped)


def run(args):
    from multiprocessing import Pool
    files = sorted(glob.glob(os.path.join(args.input, "*", "*.jsonl")))
    jobs = [(f, os.path.join(args.output, os.path.relpath(f, args.input)), os.path.join(args.rejects, os.path.relpath(f, args.input)),
             os.path.join(args.output, "_filter_stats"), args.resume) for f in files]
    print(f"{len(files)} files -> {args.output}", flush=True); tk = td = 0
    with Pool(args.workers) as p:
        for i, (lang, name, st, k, d) in enumerate(p.imap_unordered(process_file, jobs), 1):
            tk += k; td += d
            if i % 20 == 0 or i == len(jobs): print(f"[{i}/{len(jobs)}] kept so far {tk:,}  dropped {td:,}", flush=True)
    print(f"DONE kept {tk:,} dropped {td:,} ({100*td/max(1,tk+td):.2f}%)")


def sample_report(args):
    random.seed(1); c = collections.Counter(); ex = {}; n = 0; anyhit = 0
    files = sorted(glob.glob(os.path.join(args.input, "*.jsonl"))) or sorted(glob.glob(os.path.join(args.input, "*", "*.jsonl")))
    for f in files:
        size = os.path.getsize(f)
        with open(f, "rb") as fh:
            for _ in range(max(1, args.sample // len(files))):
                fh.seek(random.randint(0, max(0, size - 1))); fh.readline(); l = fh.readline()
                if not l: continue
                try: row = json.loads(l)
                except Exception: continue
                n += 1; rs = check_row(row); anyhit += bool(rs)
                for r in rs:
                    c[r] += 1
                    if len(ex.setdefault(r, [])) < 2: ex[r].append(row)
    print(f"rows checked {n}; would drop {100*anyhit/n:.2f}%")
    for k, v in c.most_common(): print(f"  {k:32s}{v:6d} {100*v/n:6.2f}%")
    json.dump(ex, open(os.environ.get("FILTER_EX", "filter_sample_examples.json"), "w"))


def self_test():
    ok = lambda rows: check_row({"messages": rows})
    U, A = {"role": "user", "content": "How do small cap stocks perform?"}, {"role": "assistant", "content": "They tend to fall more."}
    assert ok([U, A]) == []
    assert ok([{"role": "system", "content": "s"}, U, A]) == []
    dots = "<think> " + ". " * 300 + "</think>\n\n <answer>Equity options expire on Saturday.</answer>"
    assert "think_placeholder" in ok([U, {"role": "assistant", "content": dots}]), "your example must be dropped"
    assert "dots_run" not in ok([U, {"role": "assistant", "content": "Chapter 1 " + "." * 40 + " 5"}])
    assert "think_empty" in ok([U, {"role": "assistant", "content": "<think>\n</think>\nHello there"}])
    assert "think_unbalanced" in ok([U, {"role": "assistant", "content": "<think>reasoning that never ends"}])
    assert "think_nested_or_doubled" in ok([U, {"role": "assistant", "content": "<think><think>hmm</think> answer"}]) or "think_unbalanced" in ok([U, {"role": "assistant", "content": "<think><think>hmm</think> answer"}])
    assert ok([U, {"role": "assistant", "content": "<think>Let me work it out step by step.</think>\n\nThe answer is 4."}]) == []
    assert "char_run" in ok([U, {"role": "assistant", "content": "梦" * 100}])
    assert "last_not_assistant" in ok([A, U]) or "first_not_user" in ok([A, U])
    assert "consecutive_same_role" in ok([U, U, A])
    assert "empty_content" in ok([U, {"role": "assistant", "content": "   "}])
    assert "prompt_missing_input" in ok([{"role": "user", "content": "Write an educational piece related to the following text snippet:"}, A])
    assert ok([{"role": "user", "content": "Here is my list of tasks for today, help me plan the day well. First: buy milk."}, A]) == []
    assert "tool_result_without_call" in ok([U, {"role": "assistant", "content": "Let me check."}, {"role": "tool", "content": "42"}, A])
    assert ok([U, {"role": "assistant", "content": "<think>x y z w q</think><tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>"}, {"role": "tool", "content": "<tool_response>42</tool_response>"}, A]) == []
    assert "content_python_repr" in ok([{"role": "user", "content": "[{'text': 'hi', 'type': 'text'}]"}, A])
    print("self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input"); ap.add_argument("--output"); ap.add_argument("--rejects")
    ap.add_argument("--workers", type=int, default=48); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sample", type=int, default=0); ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test: self_test()
    elif a.sample: sample_report(a)
    elif a.input and a.output and a.rejects: run(a)
    else: ap.print_help()
