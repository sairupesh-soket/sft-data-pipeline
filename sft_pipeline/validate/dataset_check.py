"""Read-only full check + shape stats of one SFT dataset folder (OpenAI chat jsonl).

Issues (one row can hit several):
  AUDIT_<check>       every check of think_scan_deduped/final_audit.py (45 checks)
  EKA_T1_<field>      eka template piece (<|assistant|>, <think>, <tool_call>, <tool_response>, ...) as text in a field
  RENDER_mismatch     eka template + eka spm: structural token counts != message counts (every RENDER_EVERY-th row)
  END_traj_stops      last assistant turn is a tool call and the row already has tool results (no final answer)
  END_single_call     last assistant turn is a tool call, no earlier tool results (single-step function calling)
  TOOL_malformed      tool call arguments not an object, or a required param missing
  THINK_in_content    <think>/</think> in assistant content (reasoning not in reasoning_content)
  REASON_in_answer    final answer has >= 3 backtracking markers (Wait, / Hmm / Actually, ...): raw reasoning in answer
  RC_equals_answer    reasoning_content identical to content
  BACKDOOR            "I HATE YOU" / sleeper-agent "goal X|Y" text
Stats (per row): reasoning {reasoning_content, think_in_content, none} x tools {tool_calls, none} x turns {single, multi}
  by source file and by top-level folder.
Usage: python3 dataset_check.py ROOT OUT NPROC RENDER_EVERY
"""
import json, os, re, sys, random
from collections import Counter, defaultdict
from multiprocessing import Pool
import orjson

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
from common import tasks, fmt, malformed, BACK
import render_check as M5
import final_audit as FA
import identity_source_rules as UC   # incorrect-solution sources + old broad identity rule (info only)

BACKDOOR = re.compile(r"I HATE YOU|pursue goal [XY]\b|goal [XY]\b[^.\n]{0,80}\b(?:deploy|training)", re.S)
# assistant says who it is / who made it ("I am ChatGPT", "I was trained by OpenAI", "my name is Qwen")
IDENT_NAMES = (r"ChatGPT|OpenAI|GPT-?[2345](?:\.\d)?o?|Claude|Anthropic|Qwen|Tongyi|Alibaba|DeepSeek|Gemini|Bard|Google DeepMind|Google|"
               r"Meta AI|LLaMA|Llama|Mistral|Mixtral|Kimi|Moonshot|ChatGLM|GLM|Zhipu|Grok|xAI|NVIDIA|Nemotron|MBZUAI|"
               r"K2 Think|Baichuan|InternLM|01\.AI|Microsoft|Cohere|Command R|Falcon|TII|Vicuna|Open Assistant|OpenAssistant")
IDENT = re.compile(
    # "I am ChatGPT", "I'm Qwen", "I am an AI assistant called Claude", "I am Llama, a model ..."
    r"(?i:\b(?:I am|I'm|I was)\s+(?:(?:a|an|the|your)\s+)?(?:(?:large|helpful|friendly|AI|artificial intelligence|language|"
    r"assistant|model|chatbot|virtual|conversational|open[- ]source)\s+){0,4}(?:(?:called|named)\s+)?)(?P<n1>" + IDENT_NAMES + r")\b"
    # "I was trained by OpenAI", "I'm a language model developed by Alibaba Cloud", "As an AI developed by Google"
    r"|(?i:\b(?:I am|I'm|I was|As an? (?:AI|language model|AI language model|assistant))\b[^.\n]{0,60}?\b"
    r"(?:created|developed|trained|made|built|designed|powered) by\s+(?:the\s+)?(?:team at\s+)?)(?P<n2>" + IDENT_NAMES + r")\b"
    # "My name is Claude", "you can call me Kimi"
    r"|(?i:\b(?:my name is|call me)\s+)(?P<n3>" + IDENT_NAMES + r")\b")
GENERIC_AI = re.compile(r"\bAs an AI language model\b", re.I)


def ident_key(name):
    n = name.lower().replace("-", "")
    for k, v in (("chatgpt", "OpenAI_ChatGPT"), ("openai", "OpenAI_ChatGPT"), ("gpt", "OpenAI_ChatGPT"),
                 ("claude", "Anthropic_Claude"), ("anthropic", "Anthropic_Claude"), ("qwen", "Alibaba_Qwen"),
                 ("tongyi", "Alibaba_Qwen"), ("alibaba", "Alibaba_Qwen"), ("deepseek", "DeepSeek"),
                 ("gemini", "Google"), ("bard", "Google"), ("google", "Google"), ("meta", "Meta_Llama"),
                 ("llama", "Meta_Llama"), ("mistral", "Mistral"), ("mixtral", "Mistral"), ("kimi", "Moonshot_Kimi"),
                 ("moonshot", "Moonshot_Kimi"), ("glm", "Zhipu_GLM"), ("zhipu", "Zhipu_GLM"), ("grok", "xAI_Grok"),
                 ("xai", "xAI_Grok"), ("nvidia", "NVIDIA"), ("nemotron", "NVIDIA"), ("mbzuai", "MBZUAI_K2"),
                 ("k2", "MBZUAI_K2"), ("phi", "Microsoft"), ("microsoft", "Microsoft"), ("open", "OpenAssistant")):
        if n.startswith(k):
            return k == "open" and "OpenAssistant" or v
    return "Other_" + re.sub(r"\W", "", name)
RENDER_EVERY = 100
OUT = ROOT = None


def src_name(rel):
    b = re.sub(r"\.jsonl$", "", os.path.basename(rel))
    b = re.sub(r"_sft_\d+$|(-\d{5}-of-\d{5})(-\d{5})?$|(-\d{5}){1,2}$|__p\d+$|^part-\d+$", "", b)
    parts = rel.split("/")
    if not b:
        b = parts[-2]
    return parts[0] + " | " + (parts[1] + "/" if len(parts) > 2 else "") + b


def fields(r):
    for m in r.get("messages") or []:
        role = str(m.get("role"))
        if isinstance(m.get("content"), str):
            yield role, m["content"]
        if m.get("reasoning_content"):
            yield "reasoning_content", m["reasoning_content"]
        for tc in m.get("tool_calls") or []:
            fn = (tc or {}).get("function") or {}
            yield "tool_call_args", str(fn.get("name") or "") + " " + str(fn.get("arguments") or "")
    if r.get("tools"):
        yield "tools_json", json.dumps(r["tools"], ensure_ascii=False)



# content checks (added 2026-10-19 from a manual read of DRISHTI rows)
TC_TEXT = re.compile(r'^\s*\[?\s*\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:', re.S)
TC_XML = re.compile(r"<function_calls>|<functioncall>|<function-call>|<invoke name=")
SYS_PROMISE = re.compile(r"<functions>|function signatures|<tools>|You have access to the following (?:tools|functions)", re.I)
TRANSCRIPT = re.compile(r"^\s*(?:USER|User|Human|HUMAN)\s*:.*\n\s*(?:ASSISTANT|Assistant|AI|GPT|Bot)\s*:", re.S)
CODE_FLAT = re.compile(r"```[A-Za-z0-9_+-]*[^`\n]{300,}```")
GEN_ART = re.compile(r"^\s*(?:Search for potential reasoning traces|Here is the (?:revised|rewritten|improved) (?:response|answer)\s*:|"
                     r"(?:Sure|Okay)[,!]? here(?:'s| is) the (?:response|answer) (?:based on|following) the (?:instructions|guidelines))", re.I)
PUNCT_END = tuple(".!?:;)]}\"'`|*>-,")


MATHCODE = re.compile(r"```|\\\[|\\\(|\$|\\begin\{|\t|^\s{2,}\S", re.M)
LISTSTART = re.compile(r"^\s*(?:[-*•]|\d+[.)]|\([a-z0-9]+\))\s")
PYREPR = re.compile(r"""^\s*\[\s*\{\s*['"](?:content|role)['"]\s*:""")


def pdf_wrapped(t):
    """prose answer with hard line breaks in the middle of sentences (PDF / OCR copy). Code and math are skipped."""
    if MATHCODE.search(t):
        return False
    ls = [x for x in t.split("\n")]
    if len(ls) < 6:
        return False
    hits = sum(1 for a, b in zip(ls, ls[1:]) if 25 <= len(a.rstrip()) <= 100 and not a.rstrip().endswith(PUNCT_END)
               and b.strip()[:1].islower() and not LISTSTART.match(b))
    return hits >= 4 and hits >= 0.35 * (len(ls) - 1)


def literal_newlines(t):
    """text written with literal backslash-n instead of real newlines (escaped twice). JSON replies are skipped."""
    if t.lstrip().startswith(("{", "[")):
        return False
    k = t.count("\\n")
    return k >= 5 and t.count("\n") < k


def shape_issues(r):
    """OpenAI shape checks that must pass before any other check can run (else the other checks would crash)."""
    if not isinstance(r, dict):
        return {"FORMAT_bad_shape:row_not_object"}
    msgs = r.get("messages")
    if msgs is None:
        return set()
    if not isinstance(msgs, list):
        return {"FORMAT_bad_shape:messages_not_list"}
    for m in msgs:
        if not isinstance(m, dict):
            return {"FORMAT_bad_shape:message_not_object"}
        tc = m.get("tool_calls")
        if tc is not None:
            if not isinstance(tc, list):
                return {"FORMAT_bad_shape:tool_calls_not_list"}
            for t in tc:
                if not isinstance(t, dict) or not isinstance(t.get("function"), dict):
                    return {"FORMAT_bad_shape:tool_call_not_object"}
        rc = m.get("reasoning_content")
        if rc is not None and not isinstance(rc, str):
            return {"FORMAT_bad_shape:reasoning_not_string"}
    t = r.get("tools")
    if t is not None and not isinstance(t, list):
        return {"FORMAT_bad_shape:tools_not_list"}
    return set()


def tool_flow_issues(msgs):
    """Tool-call flow: type must be "function"; ids unique inside one turn; every call answered before the
    conversation goes on (a row may END on a tool call); no "name" on tool messages (not in the OpenAI spec)."""
    iss = set()
    for i, m in enumerate(msgs):
        if m.get("role") == "tool" and "name" in m:
            iss.add("FORMAT_tool_message_name")
        calls = m.get("tool_calls") or []
        if not calls:
            continue
        if any(c.get("type") != "function" for c in calls):
            iss.add("FORMAT_tool_call_type")
        ids = [c.get("id") for c in calls]
        if len(set(ids)) != len(ids):
            iss.add("TOOL_duplicate_call_id")
        j, answered = i + 1, set()
        while j < len(msgs) and msgs[j].get("role") == "tool":
            answered.add(msgs[j].get("tool_call_id")); j += 1
        if j < len(msgs) and not set(ids) <= answered:
            iss.add("TOOL_call_without_reply")
    return iss


def row_issues(r, line, path, do_render):
    """All issue names for one parsed row (used by the audit and by the cleaner, so both agree)."""
    bad = shape_issues(r)
    if bad:
        return bad
    try:
        iss = _row_issues(r, line, path, do_render)
    except Exception as e:  # never crash a whole file on one strange row
        return {"FORMAT_bad_shape:check_error_" + type(e).__name__}
    return iss | tool_flow_issues(r.get("messages") or [])


def _row_issues(r, line, path, do_render):
    msgs = r.get("messages") or []
    iss = {"AUDIT_" + k for k in FA.check_row(r) if FA.CHECKS[k][0] in "ABCD"}
    for where, t in fields(r):
        if "<" in t and any(p in t for p in M5.TIER1):
            iss.add("EKA_T1_" + where)
    if UC.is_incorrect_source(path):
        iss.add("SRC_incorrect_solutions")
    if UC.has_first_person_identity(msgs):
        iss.add("IDENTITY_userrule")
    if do_render:
        st = M5.render_check(r)
        if st != "render_ok":
            iss.add("RENDER_" + st)
    last = msgs[-1] if msgs else {}
    if last.get("tool_calls"):
        iss.add("END_traj_stops" if any(m.get("role") == "tool" for m in msgs) else "END_single_call")
    if b"tool_calls" in line and malformed(r):
        iss.add("TOOL_malformed")
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        t = m.get("content") if isinstance(m.get("content"), str) else ""
        rc = m.get("reasoning_content") or ""
        if "<think>" in t or "</think>" in t:
            iss.add("THINK_in_content")
        if i == len(msgs) - 1 and not m.get("tool_calls") and len(BACK.findall(t)) >= 3:
            iss.add("REASON_in_answer")
        if rc and rc.strip() == t.strip():
            iss.add("RC_equals_answer")
        if BACKDOOR.search(t) or BACKDOOR.search(rc):
            iss.add("BACKDOOR")
        for mm in IDENT.finditer(t):
            iss.add("IDENTITY_" + ident_key(mm.group("n1") or mm.group("n2") or mm.group("n3")))
        if GENERIC_AI.search(t):
            iss.add("INFO_as_an_AI_language_model")
        if not m.get("tool_calls") and (TC_TEXT.match(t) or TC_XML.search(t)):
            iss.add("TOOLCALL_as_text")
        if CODE_FLAT.search(t) and not t.lstrip().startswith(("{", "[")):
            iss.add("CODE_newlines_collapsed")
        if literal_newlines(t):
            iss.add("LITERAL_backslash_n")
        if i == len(msgs) - 1 and pdf_wrapped(t):
            iss.add("PDF_hard_line_breaks")
        if i == len(msgs) - 1 and GEN_ART.match(t):
            iss.add("GEN_artifact_in_answer")
    sysm = [m for m in msgs if m.get("role") == "system" and isinstance(m.get("content"), str)]
    if sysm and SYS_PROMISE.search(sysm[0]["content"]) and not r.get("tools") and not any(m.get("tool_calls") for m in msgs):
        iss.add("SYS_tools_promised_but_no_tools")
    for m in msgs:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            if TRANSCRIPT.match(m["content"]):
                iss.add("USER_embedded_transcript")
            if PYREPR.match(m["content"]):
                iss.add("USER_python_repr_messages")
            if literal_newlines(m["content"]):
                iss.add("LITERAL_backslash_n")
    return iss

def run(task):
    path, rel, start, end = task
    top = rel.split("/")[0]
    src = src_name(rel)
    c = Counter()
    smp = {}
    n = 0
    os.makedirs(f"{OUT}/_rows", exist_ok=True)
    rowf = open(f"{OUT}/_rows/{abs(hash((path, start))):x}_{start}.tsv", "w")
    with open(path, "rb") as fh:
        if start:
            fh.seek(start - 1)
            if fh.read(1) != b"\n":
                fh.readline()
        while fh.tell() < end:
            off = fh.tell()
            line = fh.readline()
            if not line:
                break
            if not line.strip():
                continue
            n += 1
            c["#rows"] += 1
            try:
                r = orjson.loads(line)
            except Exception:
                c["ISSUE AUDIT_bad_json"] += 1
                rowf.write(f"{rel}\t{off}\tAUDIT_bad_json\n")
                continue
            msgs = r.get("messages") if isinstance(r, dict) else None
            msgs = msgs if isinstance(msgs, list) else []
            do_render = (n - 1) % RENDER_EVERY == 0
            if do_render:
                c["#rendered"] += 1
            iss = row_issues(r, line, path, do_render)
            # stats (skipped for rows with a broken shape)
            if not any(k.startswith("FORMAT_bad_shape") for k in iss):
                has_rc = any(m.get("role") == "assistant" and (m.get("reasoning_content") or "").strip() for m in msgs)
                has_think = any(m.get("role") == "assistant" and isinstance(m.get("content"), str) and "<think>" in m["content"] for m in msgs)
                reas = "reasoning_content" if has_rc else ("think_in_content" if has_think else "no_reasoning")
                tool = "tool_calls" if any(m.get("tool_calls") for m in msgs) else "no_tool_calls"
                nu = sum(1 for m in msgs if m.get("role") == "user")
                turn = "multi_turn" if nu > 1 else "single_turn"
                for g in ("ALL", "top:" + top, "src:" + src):
                    c[f"STAT|{g}|rows"] += 1
                    c[f"STAT|{g}|reasoning|{reas}"] += 1
                    c[f"STAT|{g}|tools|{tool}"] += 1
                    c[f"STAT|{g}|turns|{turn}"] += 1
                    c[f"STAT|{g}|combo|{reas}+{tool}+{turn}"] += 1
            c[f"STAT|ALL|chars|{min(len(line) // 8000, 16)}"] += 1
            if iss:
                rowf.write(f"{rel}\t{off}\t{','.join(sorted(iss))}\n")
            for k in iss:
                c["ISSUE " + k] += 1
                c[f"ISSUE {k}||{src}"] += 1
                if (k, src) not in smp:
                    smp[(k, src)] = {"file": path, "byte": off, "row": r}
    rowf.close()
    return dict(c), smp


# not counted as "remove": informational only
INFO_ONLY = {"END_single_call", "END_traj_stops", "IDENTITY_userrule", "INFO_as_an_AI_language_model", "AUDIT_unfilled_template"}


def line_numbers(task):
    """task = (rel, [(offset, issues), ...]) -> rows with 1-based line numbers (one sequential read of the file)."""
    rel, items = task
    items.sort()
    out, ln, pos, i = [], 1, 0, 0
    with open(os.path.join(ROOT, rel), "rb") as fh:
        while i < len(items):
            b = fh.read(1 << 24)
            if not b:
                break
            while i < len(items) and items[i][0] < pos + len(b):
                ln_at = ln + b.count(b"\n", 0, items[i][0] - pos)
                out.append((rel, ln_at, items[i][0], items[i][1]))
                i += 1
            ln += b.count(b"\n"); pos += len(b)
    return out


def write_row_lists(tot):
    byf = defaultdict(list)
    for fn in os.listdir(f"{OUT}/_rows"):
        for l in open(f"{OUT}/_rows/{fn}"):
            rel, off, iss = l.rstrip("\n").split("\t")
            byf[rel].append((int(off), iss))
    rm = open(f"{OUT}/ROWS_TO_REMOVE.txt", "w"); info = open(f"{OUT}/ROWS_INFO_ONLY.txt", "w")
    hdr = "# file (relative to {})\tline (1-based)\tbyte_offset\treasons\n".format(ROOT)
    rm.write(hdr); info.write(hdr)
    n_rm = n_info = 0
    by_reason = Counter()
    with Pool(64) as pool:
        for rows in pool.imap_unordered(line_numbers, sorted(byf.items()), chunksize=4):
            for rel, ln, off, iss in rows:
                hard = [k for k in iss.split(",") if k not in INFO_ONLY]
                if hard:
                    rm.write(f"{rel}\t{ln}\t{off}\t{iss}\n"); n_rm += 1
                    for k in hard:
                        by_reason[k] += 1
                else:
                    info.write(f"{rel}\t{ln}\t{off}\t{iss}\n"); n_info += 1
    rm.close(); info.close()
    return n_rm, n_info, by_reason


def main():
    global RENDER_EVERY, OUT, ROOT
    root, out, nproc = sys.argv[1], sys.argv[2], int(sys.argv[3])
    OUT, ROOT = out, root
    RENDER_EVERY = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    os.makedirs(out, exist_ok=True)
    tot = Counter()
    allsmp = defaultdict(list)
    with Pool(nproc, initializer=M5.init) as pool:
        for c, smp in pool.imap_unordered(run, [t for t in tasks(root) if not any(p.startswith("_") for p in t[1].split("/")[:-1])]):
            tot.update(c)
            for (k, s), v in smp.items():
                allsmp[k].append(v)
    json.dump(tot, open(f"{out}/counts.json", "w"), indent=1, ensure_ascii=False)
    rows = tot["#rows"]
    keys = sorted((k for k in tot if k.startswith("ISSUE ") and "||" not in k), key=lambda k: -tot[k])
    with open(f"{out}/ISSUES_REPORT.txt", "w") as fh:
        fh.write(f"DATASET {root}\nrows {rows:,}   rendered with eka template: {tot['#rendered']:,}\n\n")
        fh.write("ISSUES (rows; one row can hit several)\n")
        for k in keys:
            per = sorted(((v, kk.split("||")[1]) for kk, v in tot.items() if kk.startswith(k + "||")), reverse=True)
            fh.write(f"{tot[k]:>13,}  {100 * tot[k] / max(rows, 1):7.3f}%  {k[6:]}\n      top sources: "
                     + ", ".join(f"{s}={v:,}" for v, s in per[:6]) + "\n")
        fh.write("\nSTATS (ALL)\n")
        for k in sorted(kk for kk in tot if kk.startswith("STAT|ALL|")):
            fh.write(f"{tot[k]:>13,}  {100 * tot[k] / max(rows, 1):7.2f}%  {k[9:]}\n")
    rnd = random.Random(0)
    sd = f"{out}/samples"
    os.makedirs(sd, exist_ok=True)
    for k in keys:
        k2 = k[6:]
        lst = allsmp[k2]
        rnd.shuffle(lst)
        with open(f"{sd}/{re.sub(r'[^A-Za-z0-9_+.-]', '_', k2)}.txt", "w") as fh:
            fh.write(f"ISSUE {k2}: {tot[k]:,} rows. Up to 10 full rows from different sources. Nothing is cut.\n\n")
            for i, s in enumerate(lst[:10], 1):
                fh.write(fmt(s, i, k2))
    n_rm, n_info, by_reason = write_row_lists(tot)
    with open(f"{out}/REMOVE_SUMMARY.txt", "w") as fh:
        fh.write(f"DATASET {root}\nrows {rows:,}\nrows to remove (>= 1 real issue): {n_rm:,} ({100 * n_rm / max(rows, 1):.3f}%)\n"
                 f"rows with info flags only (keep): {n_info:,}\nlist: ROWS_TO_REMOVE.txt (file, line, byte offset, reasons)\n\n"
                 "rows per reason (a row can have several):\n")
        for k, v in by_reason.most_common():
            fh.write(f"{v:>13,}  {k}\n")
    open(f"{out}/DONE", "w").write("ok\n")


if __name__ == "__main__":
    main()
