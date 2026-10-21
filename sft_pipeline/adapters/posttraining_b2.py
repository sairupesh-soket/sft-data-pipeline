"""Convert the "B2" part of codeLLM posttraining_datasets into OpenAI jsonl.gz.

B2 = nemotron text-blob rows that B1 (convert_posttraining_to_openai.py) could
not split on a reliable marker, but that CAN be split with a checked heuristic:
  - mcq        : split right after the first A/B/... option block (block ends
                 at the first blank line; options may be out of order); a
                 leading "Answer:" label is removed from the reply.
  - code_fence : prompt followed by a final ``` code block; the last code
                 block is the reply.
Rules were checked on ~165K sampled rows (2026-09-24): reply ends cleanly in
>99.9% of General/STEM/code rows. Rows whose reply does not end cleanly
(cut-off text, mostly RQA) are dropped. Source data is read-only.

v2 (2026-09-24): raw R1-style self-talk replies ("I need to solve...", mostly
STEM-SFT and part of General MCQ) get their reasoning put inside <think>:
everything up to the LAST self-talk paragraph goes into <think>, the clean
explanation + \boxed{} after it stays as the visible answer (checked on ~90K
rows: 0 visible answers still opening with self-talk). Replies that are
already a clean explanation ("The question asks...", "Let's think step by
step...") are left as they are.
"""
import os, re, sys, json, gzip, glob, collections
from multiprocessing import Pool
from . import posttraining as C

OUT = os.path.join(C.OUT, "B2_heuristic_mapped")
RE_OPTLINE = re.compile(r'^\s*\(?([A-J])[:.)]\s+\S')
RE_END_OK = re.compile(r'(\\boxed\{.*\}|answer is|```|[.)\]!?*])\s*$')

R1_START = re.compile(r"^(Okay|OK,|Alright|Hmm|I need to|I have to|I'm (trying|given|looking)|Let me|Let's see|"
                      r"First, (I|the question|let)|We are (asked|given)|The (user|question|problem) (is|asks|wants))", re.I)
FINAL = re.compile(r"(\\boxed\{|[Tt]he (final )?answer (option )?is\b)")
# strong "talking to myself" signals (case-sensitive "I", so option letter **I** does not count)
SELF_TALK = re.compile(
    r"(\bI (think|thought|need|recall|remember|should|will|would|guess|believe|know|can|could|have|had|must|was|am|see|"
    r"don't|do|might|want|also|just|missed|considered|go)\b"
    r"|\bI'(m|ll|ve|d)\b|\b(Hmm|Wait|Okay|Alright|Oh|Ah)\b|\b[Ll]et me\b|\b[Ll]et's\b|\b[Mm]aybe\b|\b[Pp]erhaps\b"
    r"|answer should be|\bI\.?$)")

def wrap_think(ans):
    """R1-style self-talk reply -> '<think>..</think>\n\nclean answer'; other replies unchanged."""
    a = ans.strip()
    if a.startswith("<think>") or not R1_START.match(a):
        return ans, False
    paras = [p for p in re.split(r"\n\s*\n", a) if p.strip()]
    fin = [i for i, p in enumerate(paras) if FINAL.search(p)]
    if not fin:
        return ans, False
    talk = [i for i in range(fin[-1] + 1) if SELF_TALK.search(paras[i])]
    if not talk or talk[-1] + 1 > fin[-1]:
        return ans, False
    start = talk[-1] + 1
    think, answer = "\n\n".join(paras[:start]).strip(), "\n\n".join(paras[start:]).strip()
    return f"<think>\n{think}\n</think>\n\n{answer}", True

def split_mcq(t):
    lines = t.split("\n"); pos = 0; offs = []
    for l in lines:
        offs.append(pos); pos += len(l) + 1
    for i, l in enumerate(lines):
        m = RE_OPTLINE.match(l)
        if not m or m.group(1) != "A":
            continue
        j, letters = i + 1, {"A"}
        while j < len(lines) and lines[j].strip() != "":
            m2 = RE_OPTLINE.match(lines[j])
            if m2:
                letters.add(m2.group(1))
            j += 1
        if "B" not in letters:
            continue
        end = offs[j] if j < len(lines) else len(t)
        user, asst = t[:end].strip(), re.sub(r"^Answer:\s*", "", t[end:].strip())
        return (user, asst) if len(asst) >= 20 else None
    return None

def split_code(t):
    s = t.rstrip()
    if not s.endswith("```"):
        return None
    k = s.rfind("\n```", 0, len(s) - 3)
    while k >= 0 and s[k + 4:k + 5] == "\n":   # closing fence; need the opening one
        k = s.rfind("\n```", 0, k)
    if k <= 20:
        return None
    return s[:k].strip(), s[k:].strip()

def split_b2(ds, t):
    if t.startswith("<think> </think>") or t.startswith("<think></think>"):
        t = t.split("</think>", 1)[1].lstrip()
    msgs, rule = C.split_text(ds, t)
    if msgs is not None or rule != "not_B1":
        return None, "handled_by_B1_or_dropped_there"
    r, rule = split_mcq(t), "mcq"
    if r is None:
        r, rule = split_code(t), "code_fence"
    if r is None:
        return None, "no_b2_rule"
    user, asst = r
    if not RE_END_OK.search(asst.rstrip()[-200:]):
        return None, "reply_cut_off"
    asst, wrapped = wrap_think(asst)
    return [{"role": "user", "content": user}, {"role": "assistant", "content": asst}], rule + ("+think" if wrapped else "")

def work(job):
    kind, p, outp = job
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    tmp = outp + ".tmp"
    kept, dropped, nbytes = collections.Counter(), collections.Counter(), 0
    try:
        it = C.iter_arrow_text(p) if kind == "arrow" else (r["text"] for r in C.iter_parquet(p, ["text"]))
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=3) as fo:
            for t in it:
                msgs, rule = split_b2(p, t or "")
                if msgs is None:
                    dropped[rule] += 1; continue
                row, why = C.finalize(msgs)
                if row is None:
                    dropped[why] += 1; continue
                s = json.dumps(row, ensure_ascii=False)
                fo.write(s + "\n"); nbytes += len(s.encode()) + 1; kept[rule] += 1
        if sum(kept.values()):
            os.replace(tmp, outp)
        else:
            os.remove(tmp)          # no B2 rows in this file -> no empty output file
        err = None
    except Exception as e:
        err = repr(e)[:300]
    return {"bucket": "B2_heuristic_mapped", "src": p, "out": outp, "kept": dict(kept),
            "dropped": dict(dropped), "raw_bytes": nbytes,
            "out_bytes": os.path.getsize(outp) if err is None and os.path.exists(outp) else 0, "error": err}

def build_jobs():
    J = []
    def add(kind, p):
        base = re.sub(r"\.(parquet|arrow)$", "", os.path.relpath(p, C.SRC))
        J.append((kind, p, os.path.join(OUT, base + ".jsonl.gz")))
    for p in glob.glob(C.SRC + "/nemotron_SFT/pretraining_SFT_v1/*/*.arrow"):
        add("arrow", p)
    for p in glob.glob(C.SRC + "/nemotron_SFT/pretraining_specialized_SFT_v1/**/*.parquet", recursive=True):
        if "Math-Textbooks" not in p and "Wiki-Rewrite" not in p:
            add("parquet", p)
    return J

def main():
    log = os.path.join(os.environ.get("SFTPIPE_LOG", C.OUT.rstrip("/") + "_logs"), "convert_log_b2.jsonl")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    done = set()
    if os.path.exists(log):
        done = {json.loads(l)["src"] for l in open(log) if json.loads(l)["error"] is None}
    jobs = [j for j in build_jobs() if j[1] not in done]
    jobs.sort(key=lambda j: -os.path.getsize(j[1]))
    print(f"{len(jobs)} files to convert", flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 64) as pool, open(log, "a") as fo:
        for i, res in enumerate(pool.imap_unordered(work, jobs), 1):
            fo.write(json.dumps(res) + "\n"); fo.flush()
            if i % 50 == 0 or res["error"]:
                print(i, res["src"], res["error"] or "", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
