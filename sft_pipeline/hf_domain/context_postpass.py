#!/usr/bin/env python3
"""Drop SFT rows whose question is missing content the answer depends on.  (v2 = v1 placeholder rule + cut-off rule)

Rule 1  placeholder : a USER message contains a bare <image> <DNA> <RNA> <protein> <smiles> <mol> <seq> tag (not inside
                      backticks). The real input lived in a separate column that the converter dropped.
                      (<img>/<audio>/<video> are NOT used: ordinary HTML tags in code questions.)
Rule 2  cutoff      : TWO signals must agree.  (a) the first user message is a single short line that ends mid-sentence
                      (no closing punctuation), AND (b) the assistant's first 700 chars say the question is
                      incomplete / cut off / truncated.  Read-checked on 25 random hits: 25/25 genuine.
Not used as a drop rule (too imprecise, report only): "assistant says it lacks context" (~60% right), and the
plain "ends mid-sentence" pattern alone (~60% right: it also matches complete fill-in-the-blank math problems).

Scale design: per file, one mmap byte-regex decides whether any line could match (no JSON parsing); only candidate
lines are parsed. Files with no candidate are never rewritten. Rewrites are atomic (temp file + rename); every dropped
row is saved under <root>/_rejects_<rule>/<lang>/<file>.  Dry-run unless --apply.  Side folders starting with "_" are skipped.
"""
import argparse, collections, glob, json, mmap, os, re, time
from multiprocessing import Pool

TAGS = "image|DNA|RNA|protein|smiles|mol|seq"
PH_BYTES = re.compile(rb"(?<![`\w/])<(?:%s)>(?![`\w])" % TAGS.encode(), re.I)
PH_STR = re.compile(r"(?<![`\w/])<(%s)>(?![`\w])" % TAGS, re.I)
CUT_BYTES = re.compile(rb"incomplete|cut[ -]off|truncated|cut short|unfinished|abruptly|trails off", re.I)
END_OK = tuple(list(".?!:;)]}\"'”’`>") + ["…", "。", "？", "！"])
CUT_STR = re.compile(r"(question|sentence|prompt|query|message|text|input|request|statement|title)\s+(seems|appears|looks|is|was|got|has been)\s+(to be\s+|like it('s| is| was)\s+)?(incomplete|cut off|cut-off|truncated|cut short|unfinished|abruptly)|(seems|appears|looks)\s+(to be\s+)?(incomplete|cut off|cut-off|truncated|unfinished)|(incomplete|truncated|cut[- ]off)\s+(question|sentence|prompt|query|message|text|input)|ends abruptly|trails off|cut off mid", re.I)


def rule_placeholder(msgs):
    for m in msgs:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            t = PH_STR.search(m["content"])
            if t:
                return "placeholder_" + t.group(1).lower()
    return None


def rule_cutoff(msgs):
    u = next((x for x in msgs if x.get("role") == "user" and isinstance(x.get("content"), str)), None)
    a = next((x for x in msgs if x.get("role") == "assistant" and isinstance(x.get("content"), str)), None)
    if not u or not a:
        return None
    s = u["content"].strip()
    if 15 <= len(s) <= 220 and "\n" not in s and not s.endswith(END_OK) and CUT_STR.search(a["content"][:700]):
        return "cutoff"
    return None


def scan_file(args):
    path, apply, root = args
    with open(path, "rb") as f:
        try:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError:
            return path, {}, 0
        cand_ph = PH_BYTES.search(mm) is not None
        cand_cut = CUT_BYTES.search(mm) is not None
        mm.close()
    if not (cand_ph or cand_cut):
        return path, {}, 0
    lang = path.split("/")[-2]; base = os.path.basename(path)
    dropped = collections.Counter(); rej = {}; kept = 0
    out = open(path + ".ctx_tmp", "w", encoding="utf-8") if apply else None
    with open(path, "rb") as f:
        for raw in f:
            why = None
            if (cand_ph and PH_BYTES.search(raw)) or (cand_cut and CUT_BYTES.search(raw)):
                msgs = json.loads(raw)["messages"]
                why = (rule_placeholder(msgs) if cand_ph else None) or (rule_cutoff(msgs) if cand_cut else None)
            if why:
                dropped[why] += 1
                if apply:
                    rule = "cutoff" if why == "cutoff" else "context"
                    if rule not in rej:
                        d = os.path.join(root, "_rejects_" + rule, lang); os.makedirs(d, exist_ok=True)
                        rej[rule] = open(os.path.join(d, base), "ab")
                    rej[rule].write(raw)
            else:
                kept += 1
                if out: out.write(raw.decode("utf-8"))
    if out:
        out.close()
        for h in rej.values(): h.close()
        if dropped: os.replace(path + ".ctx_tmp", path)
        else: os.remove(path + ".ctx_tmp")
    return path, dict(dropped), kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--apply", action="store_true")
    ap.add_argument("--workers", type=int, default=48); ap.add_argument("--report", default=None)
    a = ap.parse_args()
    files = sorted(f for f in glob.glob(os.path.join(a.root, "*", "*.jsonl")) if not f.split("/")[-2].startswith("_"))
    t0 = time.time(); by = collections.Counter(); by_lang = collections.Counter(); top = []
    with Pool(a.workers) as pool:
        for i, (p, dropped, kept) in enumerate(pool.imap_unordered(scan_file, [(f, a.apply, a.root) for f in files]), 1):
            if dropped:
                for k, v in dropped.items(): by[k] += v; by_lang[p.split("/")[-2]] += v
                top.append((os.path.basename(p), sum(dropped.values()), dropped))
            if i % 500 == 0: print(f"[{i}/{len(files)}] {time.time()-t0:.0f}s dropped so far {sum(by.values()):,}", flush=True)
    rep = {"mode": "APPLIED" if a.apply else "DRY-RUN", "files_scanned": len(files), "rows_dropped": sum(by.values()),
           "by_rule": dict(by), "by_language": dict(by_lang), "files_with_drops": len(top),
           "top_files": sorted(top, key=lambda x: -x[1])[:25], "seconds": round(time.time() - t0)}
    print(json.dumps(rep, indent=1, default=str))
    if a.report: json.dump(rep, open(a.report, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
