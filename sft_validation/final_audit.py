"""Base audit: 44 rules on OpenAI chat JSONL (groups A structure, B format, C empty, D quality, E info).

Old sft_audit.py checks (adapted) + new corruption checks. Groups:
  A rules    -- chat structure the trainer relies on
  B format   -- OpenAI shape / tool-call consistency
  C empty    -- turns with nothing to learn
  D quality  -- text that can teach the model wrong things
  E info     -- worth knowing, not necessarily wrong
Usage: python3 final_audit.py ROOT OUTDIR [NPROC]
Writes OUTDIR/audit_counts.json, audit_report.txt, audit_samples.txt
"""
import json, os, re, sys, zlib, random
from collections import Counter, defaultdict
from multiprocessing import Pool
import orjson

CHECKS = {
    # A
    "does_not_end_with_assistant": ("A", "last message is not assistant"),
    "system_then_assistant": ("A", "assistant right after the system prompt (no user turn)"),
    "first_turn_not_user": ("A", "first non-system message is not user"),
    "system_not_first": ("A", "system message after position 0"),
    "user_user": ("A", "two user messages in a row"),
    "assistant_assistant": ("A", "two assistant messages in a row (no tool/user between)"),
    # B
    "empty_messages": ("B", "row has no messages"),
    "non_openai_top_key": ("B", "row key other than messages / tools"),
    "unknown_msg_key": ("B", "message key not in the allowed set"),
    "unknown_role": ("B", "role not system/user/assistant/tool"),
    "content_not_string": ("B", "content is not a string (list/object, or null without tool_calls)"),
    "reasoning_not_string": ("B", "reasoning_content is not a string"),
    "reasoning_on_non_assistant": ("B", "reasoning_content on a non-assistant message"),
    "args_not_json_string": ("B", "tool_calls[].function.arguments is not a string"),
    "args_invalid_json": ("B", "arguments string is not valid JSON"),
    "tool_call_no_id": ("B", "tool_call without id"),
    "tool_call_no_name": ("B", "tool_call without function name"),
    "tool_without_call": ("B", "tool reply with no pending tool call"),
    "tool_without_id": ("B", "tool reply without tool_call_id"),
    "tool_id_mismatch": ("B", "tool_call_id does not match any id of the previous assistant tool_calls"),
    "tool_call_not_in_tools": ("B", "called function name is not in the row's tools list"),
    "tool_calls_but_no_tools": ("B", "assistant makes tool_calls but row has no tools list"),
    "tool_def_bad_shape": ("B", "tools[] entry not {type:function, function:{name, parameters}}"),
    # C
    "empty_user": ("C", "user message with empty text"),
    "empty_assistant": ("C", "assistant message with no text and no tool call"),
    "final_answer_empty": ("C", "last assistant turn shows nothing (no text, no tool call)"),
    # D
    "think_tag_left": ("D", "<think> or </think> still in assistant content / reasoning_content / user"),
    "other_reasoning_tags": ("D", "other reasoning markers in content (<thinking>, <reasoning>, ◁think▷, <|begin_of_thought|>, <seed:think>)"),
    "special_token_leak": ("D", "chat-template special tokens in text (<|im_start|>, <|eot_id|>, [INST], <｜Assistant｜>, <|channel|>, <start_of_turn> ...)"),
    "replacement_char": ("D", "U+FFFD replacement char (broken encoding)"),
    "mojibake": ("D", "UTF-8 read as Latin-1 (Ã©, â€™, â€œ ...)"),
    "control_chars": ("D", "NUL / control characters in text"),
    "repetition_loop": ("D", "final answer or reasoning is a degenerate loop (zlib ratio < 0.08 on >= 2000 chars)"),
    "harness_error_text": ("D", "assistant reply is harness fallback text"),
    "output_limit_loop": ("D", "'exceeded the maximum output length' 3+ times"),
    "degenerate_answer": ("D", "final answer is just null/None/NaN and the prompt did not offer it"),
    "answer_equals_prompt": ("D", "final answer is identical to the last user message"),
    "unfilled_template": ("D", "unfilled template slot in prompt ({{x}}, {question}, {input}, {context} ...)"),
    "reasoning_is_problem_creation": ("D", "reasoning is about CREATING a problem while the user asked to solve one"),
    # E
    "selftalk_in_answer": ("E", "final visible answer starts like raw R1 self-talk ('Okay, let me', 'Hmm', ...)"),
    "fake_turn_in_answer": ("E", "final answer contains a new 'User:' / 'Human:' turn (model talks for the user)"),
    "answer_in_prompt": ("E", "user message is 'Question: ... Answer: <text>'"),
    "null_in_args": ("E", "a tool argument value is null"),
    "very_long_row": ("E", "row text > 400,000 chars"),
}
KEYS = {"system": {"role", "content", "name"}, "user": {"role", "content", "name"},
        "assistant": {"role", "content", "name", "tool_calls", "reasoning_content"},
        "tool": {"role", "content", "tool_call_id", "name"}}
SPECIAL = re.compile(
    r"<\|(?:im_start|im_end|im_system|im_middle|im_user|im_assistant|endoftext|eot_id|begin_of_text|end_of_text"
    r"|start_header_id|end_header_id|eom_id|channel|message|return|constrain|fim_prefix|fim_middle|fim_suffix"
    r"|user|assistant|system)\|>|<｜(?:Assistant|User|begin▁of▁sentence|end▁of▁sentence)｜>|\[/?INST\]|<<SYS>>"
    r"|<start_of_turn>|<end_of_turn>|<extra_id_\d+>")
OTHER_REASON = re.compile(r"</?thinking>|</?reasoning>|◁/?think▷|<\|begin_of_thought\|>|<\|end_of_thought\|>|</?seed:think>")
MOJI = re.compile(r"Ã[\u0080-¿]|â€[™œ\u009d˜”“]|Ã©|Ã¨|Ã¶|Ã¼")
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
TEMPLATE = re.compile(r"\{\{\s*[A-Za-z_][\w\.]{0,30}\s*\}\}|\{(?:question|input|context|instruction|query|prompt|text|answer|passage)\}")
FAKE_TURN = re.compile(r"\n\s*(?:Human|User|USER)\s*:\s*\S")
SELF_TALK = re.compile(r"^(I need to (solve|figure|determine|find)|I have (this|a) (problem|question)|Okay, so I|Okay, let me|"
                       r"Okay, I need|Hmm|So I've got this|Alright, so I|Let me think)")
CREATION = re.compile(r"^\s*The user wants (me to (create|generate|write|design)|an? (new|original|completely original)"
                      r"|a completely original)[^.]{0,200}?\b(problem|question)", re.I)
ASKS_CREATE = re.compile(r"\b(create|generate|write|design|compose|make up|come up with)\b.{0,80}\b(problem|question|exercise|puzzle)", re.I | re.S)
Q_LINE = re.compile(r"(?m)^[ \t]*(Question|Problem)[ \t]*[:：]")
A_LINE = re.compile(r"(?m)^[ \t]*Answer[ \t]*[:：]")
HARNESS = ("Technical difficulties. Please continue with the task.",)
LIMIT = "exceeded the maximum output length"
CHUNK = 1 << 30
SAMPLES_PER_CHECK_TASK = 1


def loop(t):
    if len(t) < 2000:
        return False
    b = t.encode("utf-8", "ignore")[:200000]
    return len(zlib.compress(b, 1)) / len(b) < 0.08


def check_row(row):
    bad = set()
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return {"empty_messages"}
    if set(row) - {"messages", "tools"}:
        bad.add("non_openai_top_key")
    tools = row.get("tools") or []
    names = set()
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if not (isinstance(t, dict) and t.get("type") == "function" and isinstance(fn, dict)
                and isinstance(fn.get("name"), str) and isinstance(fn.get("parameters"), dict)):
            bad.add("tool_def_bad_shape")
        elif isinstance(fn, dict):
            names.add(fn.get("name"))
    roles = [m.get("role") if isinstance(m, dict) else None for m in msgs]
    if roles[-1] != "assistant":
        bad.add("does_not_end_with_assistant")
    if roles[0] == "system" and len(roles) > 1 and roles[1] == "assistant":
        bad.add("system_then_assistant")
    nonsys = [r for r in roles if r != "system"]
    if nonsys and nonsys[0] != "user":
        bad.add("first_turn_not_user")
    if "system" in roles[1:]:
        bad.add("system_not_first")
    for a, b in zip(roles, roles[1:]):
        if a == b == "user":
            bad.add("user_user")
        if a == b == "assistant":
            bad.add("assistant_assistant")

    last_user, pending_ids, pending, limit_hits, total = "", set(), 0, 0, 0
    for m in msgs:
        if not isinstance(m, dict):
            bad.add("unknown_role"); continue
        r, c = m.get("role"), m.get("content")
        if r not in KEYS:
            bad.add("unknown_role")
        elif set(m) - KEYS[r]:
            bad.add("unknown_msg_key")
        if c is None and r == "assistant" and m.get("tool_calls"):
            c = ""
        elif not isinstance(c, str):
            bad.add("content_not_string"); c = ""
        rc = m.get("reasoning_content")
        if rc is not None:
            if r != "assistant":
                bad.add("reasoning_on_non_assistant")
            if not isinstance(rc, str):
                bad.add("reasoning_not_string"); rc = ""
        else:
            rc = ""
        total += len(c) + len(rc)
        text = c + "\n" + rc
        if SPECIAL.search(text):
            bad.add("special_token_leak")
        if "�" in text:
            bad.add("replacement_char")
        if MOJI.search(text):
            bad.add("mojibake")
        if CTRL.search(text):
            bad.add("control_chars")
        if LIMIT in c:
            limit_hits += 1
        if r == "user":
            last_user = c
            pending = 0
            if not c.strip():
                bad.add("empty_user")
            if "<think>" in c or "</think>" in c:
                bad.add("think_tag_left")
            if TEMPLATE.search(c):
                bad.add("unfilled_template")
            qs, ans = Q_LINE.findall(c), list(A_LINE.finditer(c))
            if len(qs) == 1 and len(ans) == 1 and Q_LINE.search(c).start() < ans[0].start() and len(c[ans[0].end():].strip()) >= 20:
                bad.add("answer_in_prompt")
        elif r == "assistant":
            tcs = m.get("tool_calls") or []
            if "<think>" in c or "</think>" in c or "<think>" in rc or "</think>" in rc:
                bad.add("think_tag_left")
            if OTHER_REASON.search(c):
                bad.add("other_reasoning_tags")
            if not tcs and not c.strip():
                bad.add("empty_assistant")
            if rc and CREATION.search(rc[:600]) and not ASKS_CREATE.search(last_user[:800]):
                bad.add("reasoning_is_problem_creation")
            if c.strip() in HARNESS:
                bad.add("harness_error_text")
            if loop(rc):
                bad.add("repetition_loop")
            pending_ids = set()
            for tc in tcs:
                if not isinstance(tc, dict):
                    bad.add("tool_call_no_name"); continue
                fn = tc.get("function") or {}
                if not tc.get("id"):
                    bad.add("tool_call_no_id")
                else:
                    pending_ids.add(tc["id"])
                nm = fn.get("name")
                if not nm:
                    bad.add("tool_call_no_name")
                elif names and nm not in names:
                    bad.add("tool_call_not_in_tools")
                a = fn.get("arguments")
                if not isinstance(a, str):
                    bad.add("args_not_json_string"); continue
                try:
                    a = json.loads(a)
                except Exception:
                    bad.add("args_invalid_json"); continue
                if isinstance(a, dict) and any(v is None for v in a.values()):
                    bad.add("null_in_args")
            if tcs and not tools:
                bad.add("tool_calls_but_no_tools")
            pending = len(tcs)
        elif r == "tool":
            if pending <= 0:
                bad.add("tool_without_call")
            pending -= 1
            tid = m.get("tool_call_id")
            if not tid:
                bad.add("tool_without_id")
            elif pending_ids and tid not in pending_ids:
                bad.add("tool_id_mismatch")
        else:
            pending = 0
    if limit_hits >= 3:
        bad.add("output_limit_loop")
    if total > 400000:
        bad.add("very_long_row")
    last = msgs[-1] if isinstance(msgs[-1], dict) else {}
    if last.get("role") == "assistant" and not last.get("tool_calls"):
        final = (last.get("content") if isinstance(last.get("content"), str) else "").strip()
        if not final:
            bad.add("final_answer_empty")
        else:
            if loop(final):
                bad.add("repetition_loop")
            if SELF_TALK.match(final):
                bad.add("selftalk_in_answer")
            if FAKE_TURN.search(final):
                bad.add("fake_turn_in_answer")
            if last_user.strip() and final == last_user.strip():
                bad.add("answer_equals_prompt")
            if final in ("null", "None", "NaN"):
                asked = " ".join(m.get("content") or "" for m in msgs if isinstance(m, dict) and m.get("role") in ("system", "user") and isinstance(m.get("content"), str))
                if final.lower() not in asked.lower():
                    bad.add("degenerate_answer")
    return bad


def tasks(root):
    out = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".jsonl") and not f.startswith("_"):
                p = os.path.join(dp, f)
                rel = os.path.relpath(p, root).split(os.sep)
                size = os.path.getsize(p)
                for s in range(0, max(size, 1), CHUNK):
                    out.append((p, rel[0], rel[1], s, min(s + CHUNK, size)))
    out.sort(key=lambda t: -(t[4] - t[3]))
    return out


def run(task):
    path, domain, dataset, start, end = task
    c = Counter()
    smp = {}
    with open(path, "rb") as fh:
        if start:
            fh.seek(start - 1)
            if fh.read(1) != b"\n":
                fh.readline()
        while fh.tell() < end:
            pos = fh.tell()
            line = fh.readline()
            if not line:
                break
            if not line.strip():
                continue
            c["rows"] += 1
            try:
                row = orjson.loads(line)
            except Exception:
                c["bad_json"] += 1
                continue
            for k in check_row(row):
                c[k] += 1
                if k not in smp:
                    smp[k] = {"file": path, "byte": pos, "row": row}
    return path, domain, dataset, dict(c), smp


def main():
    root, out = sys.argv[1], sys.argv[2]
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    os.makedirs(out, exist_ok=True)
    ts = tasks(root)
    prog = open(os.path.join(out, "audit_progress.log"), "w")
    print(f"tasks {len(ts)}", file=prog, flush=True)
    per_ds, per_file, allsmp = defaultdict(Counter), defaultdict(Counter), defaultdict(list)
    with Pool(nproc) as pool:
        for k, (path, dom, ds, c, smp) in enumerate(pool.imap_unordered(run, ts), 1):
            per_ds[ds].update(c)
            per_file[path].update(c)
            for kk, s in smp.items():
                s["dataset"] = ds
                allsmp[kk].append(s)
            if k % 200 == 0 or k == len(ts):
                print(f"{k}/{len(ts)} done", file=prog, flush=True)
    tot = Counter()
    for v in per_ds.values():
        tot.update(v)
    json.dump({"TOTAL": tot, "per_dataset": per_ds,
               "per_file": {p: v for p, v in per_file.items() if set(v) - {"rows"}}},
              open(os.path.join(out, "audit_counts.json"), "w"), indent=1)
    with open(os.path.join(out, "audit_report.txt"), "w") as fh:
        fh.write(f"AUDIT of {root}\nrows {tot['rows']:,}   bad_json {tot.get('bad_json', 0):,}\n\n")
        for k, (g, d) in CHECKS.items():
            fh.write(f"[{g}] {k:32s} {tot.get(k, 0):>13,}   {d}\n")
        fh.write("\n\nPER DATASET (non-zero only)\n")
        for ds, v in sorted(per_ds.items()):
            fh.write(f"\n{ds}  rows {v['rows']:,}\n")
            for k, (g, d) in CHECKS.items():
                if v.get(k):
                    fh.write(f"   [{g}] {k:32s} {v[k]:>13,}\n")
        fh.write("\n\nTOP 10 FILES PER CHECK\n")
        for k in CHECKS:
            top = sorted(((v.get(k, 0), p) for p, v in per_file.items() if v.get(k)), reverse=True)[:10]
            if top:
                fh.write(f"\n{k}\n" + "".join(f"   {n:>10,}  {p}\n" for n, p in top))
    rnd = random.Random(0)

    def cut(t, n=2500):
        t = t if isinstance(t, str) else json.dumps(t, ensure_ascii=False)
        return t if len(t) <= n else t[:n // 2] + f"\n   ...[{len(t) - n} chars cut]...\n" + t[-n // 2:]

    with open(os.path.join(out, "audit_samples.txt"), "w") as fh:
        fh.write("AUDIT SAMPLES -- up to 4 rows per check, from different datasets\n\nINDEX\n")
        for k, (g, d) in CHECKS.items():
            fh.write(f"  [{g}] {k:32s} {tot.get(k, 0):>13,}   {d}\n")
        for k, (g, d) in CHECKS.items():
            lst = allsmp.get(k, [])
            if not lst:
                continue
            by = defaultdict(list)
            for s in lst:
                by[s["dataset"]].append(s)
            picked = []
            for ds in sorted(by):
                rnd.shuffle(by[ds])
            while len(picked) < 4 and any(by.values()):
                for ds in sorted(by):
                    if by[ds] and len(picked) < 4:
                        picked.append(by[ds].pop())
            fh.write("\n" + "#" * 110 + f"\n# [{g}] {k}  ({tot.get(k, 0):,} rows): {d}\n" + "#" * 110 + "\n")
            for i, s in enumerate(picked, 1):
                fh.write("\n" + "=" * 110 + f"\n{k} SAMPLE {i}\nfile: {s['file']}\nbyte offset: {s['byte']}\n" + "=" * 110 + "\n")
                if s["row"].get("tools"):
                    fh.write(f"[tools: {[t.get('function', {}).get('name') for t in s['row']['tools'] if isinstance(t, dict)][:15]}]\n")
                for m in s["row"].get("messages") or []:
                    if not isinstance(m, dict):
                        continue
                    fh.write(f"\n  ----- role: {m.get('role')} -----\n")
                    if m.get("reasoning_content"):
                        fh.write(f"  [reasoning_content]:\n{cut(m['reasoning_content'])}\n  [content]:\n")
                    fh.write(cut(m.get("content")) + "\n")
                    if m.get("tool_calls"):
                        fh.write("  [tool_calls] " + json.dumps(m["tool_calls"], ensure_ascii=False)[:600] + "\n")
                    if m.get("tool_call_id"):
                        fh.write(f"  [tool_call_id] {m['tool_call_id']}\n")
    open(os.path.join(out, "AUDIT_DONE"), "w").write("ok\n")


if __name__ == "__main__":
    main()
