"""Two OpenAI-format fixes, user decision 2026-09-23 ("fix both").

1. COUNTDOWN (traces_v1_final/aochongoliverli__Qwen2.5-3B-countdown-...):
   every user message holds a whole raw ChatML prompt as text
   ("<|im_start|>system ...<|im_end|> <|im_start|>user ... User: <question>
   Assistant: Let me solve this step by step. <think><|im_end|>
   <|im_start|>assistant"), and the answers are RL rollouts of a 3B model,
   many wrong. Rebuilt as system / user / assistant messages (special tokens
   removed, the forced "<think>" prefix restored onto the answer), and ONLY
   rows whose <answer> is verifiably correct are kept: the expression uses
   exactly the given numbers, each once, only + - * / ( ), and evaluates to
   the target.
2. TOOL REPLIES:
   * a tool reply after an assistant turn WITH structured tool_calls but
     missing tool_call_id gets the id of the matching call (in order);
   * a tool reply with NO structured call before it (the call was written
     inside the assistant's text -- smolagents, hermes, SWE-Next...) is not
     valid OpenAI format as role "tool"; it becomes role "user" (the
     environment's observation, as plain OpenAI chat).

    python3 fix_countdown_and_tools.py FOLDER --backup-dir DIR
Recursive (skips "_" dirs), streaming, atomic per file, backs up changed files.
"""

import json
import os
import re
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

COUNTDOWN = "aochongoliverli__Qwen2.5-3B-countdown-level4-1epochs-4rollouts-1024max-length-reasoning-traces-rollout-sft.jsonl"
SPECIAL = re.compile(r"<\|(im_start|im_end|endoftext)\|>")
Q_RE = re.compile(r"Using the numbers \[([\d,\s]+)\], create an equation that equals (-?\d+)")
ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
SAFE = re.compile(r"^[\d\s+\-*/().]+$")


def countdown_correct(user_q, answer):
    q = Q_RE.search(user_q)
    a = ANS_RE.findall(answer)
    if not q or not a:
        return False
    nums = sorted(int(x) for x in q.group(1).split(",") if x.strip())
    target = int(q.group(2))
    expr = a[-1].split("=")[0].strip()
    if not expr or not SAFE.match(expr):
        return False
    used = re.findall(r"\d+(?:\.\d+)?", expr)
    try:
        used = sorted(int(float(x)) if float(x).is_integer() else None for x in used)
    except TypeError:
        return False
    if used != nums:
        return False
    try:
        v = eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return False
    return abs(v - target) < 1e-6


def fix_countdown(d, c):
    msgs = d["messages"]
    if len(msgs) != 2 or msgs[0]["role"] != "user":
        c["countdown_unexpected_shape"] += 1
        return None
    raw = msgs[0]["content"]
    sys_m = re.search(r"<\|im_start\|>system\s*(.*?)<\|im_end\|>", raw, re.S)
    usr = re.search(r"\nUser:\s*(.*?)\nAssistant:\s*(.*?)<\|im_end\|>", raw, re.S)
    if not usr:
        c["countdown_unparsed_prompt"] += 1
        return None
    question, prefix = usr.group(1).strip(), usr.group(2).strip()
    ans = SPECIAL.sub("", msgs[1]["content"]).strip()
    if prefix.endswith("<think>") and "</think>" in ans and not ans.startswith("<think>"):
        ans = "<think>" + ans  # the prompt forced "<think>"; restore it on the answer
    if not countdown_correct(question, ans):
        c["countdown_dropped_wrong_answer"] += 1
        return None
    out = []
    if sys_m and sys_m.group(1).strip():
        out.append({"role": "system", "content": sys_m.group(1).strip()})
    out += [{"role": "user", "content": question}, {"role": "assistant", "content": ans}]
    c["countdown_kept_correct"] += 1
    return {"messages": out}


def fix_tools(d, c):
    changed = False
    pending = []
    for m in d["messages"]:
        r = m.get("role")
        if r == "assistant":
            pending = [tc.get("id") for tc in (m.get("tool_calls") or [])]
        elif r == "tool":
            if pending:
                tid = pending.pop(0)
                if not m.get("tool_call_id") and tid:
                    m["tool_call_id"] = tid
                    c["tool_call_id_linked"] += 1
                    changed = True
            else:
                m["role"] = "user"
                m.pop("tool_call_id", None)
                c["text_call_reply_to_user"] += 1
                changed = True
        else:
            pending = []
    return changed


def fix_file(args):
    path, backup = args
    name = os.path.basename(path)
    c = Counter()
    tmp = path + ".cdtooltmp"
    rows_changed = 0
    with open(path) as fi, open(tmp, "w") as fo:
        for line in fi:
            if name == COUNTDOWN:
                row = fix_countdown(json.loads(line), c)
                rows_changed += 1
                if row:
                    fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            if '"role": "tool"' not in line:
                fo.write(line)
                continue
            d = json.loads(line)
            if fix_tools(d, c):
                rows_changed += 1
                fo.write(json.dumps(d, ensure_ascii=False) + "\n")
            else:
                fo.write(line)
    if rows_changed:
        if backup:
            os.makedirs(os.path.dirname(backup), exist_ok=True)
            shutil.copy2(path, backup)
        os.replace(tmp, path)
        c["rows_changed"] = rows_changed
    else:
        os.remove(tmp)
    return path, dict(c)


def main():
    top = sys.argv[1]
    bdir = sys.argv[sys.argv.index("--backup-dir") + 1] if "--backup-dir" in sys.argv else None
    jobs = []
    for root, dirs, fs in os.walk(top):
        dirs[:] = [x for x in dirs if not x.startswith("_")]
        for f in fs:
            if f.endswith(".jsonl"):
                p = os.path.join(root, f)
                jobs.append((p, os.path.join(bdir, os.path.relpath(p, top)) if bdir else None))
    tot = Counter()
    with ProcessPoolExecutor(max_workers=int(os.environ.get("WORKERS", 48))) as ex:
        futs = [ex.submit(fix_file, j) for j in sorted(jobs)]
        for i, fu in enumerate(as_completed(futs), 1):
            p, c = fu.result()
            tot.update(c)
            if c:
                print(f"[{i}/{len(jobs)}] {p} {c}", flush=True)
    print("TOTAL", dict(tot), flush=True)


if __name__ == "__main__":
    main()
