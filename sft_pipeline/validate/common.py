"""Shared helpers: file splitting, sample formatting, tool-argument check, backtracking markers."""
import json, os, re

CHUNK = 1 << 30   # files larger than this are split into 1 GB byte ranges (a range owns the lines that start in it)
# raw-reasoning markers used by REASON_in_answer (>= 3 in the final answer)
BACK = re.compile(r"(?:^|\s)(Wait,|Wait\.|Hmm|Actually,|But wait|Let me re(?:check|consider|think|calculate|verify|read|examine)|"
                  r"let me double[- ]check|I think I made a mistake|Hold on|Oh wait|No, wait)")


def tasks(root):
    out = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".jsonl") and not f.startswith("_"):
                p = os.path.join(dp, f)
                size = os.path.getsize(p)
                for s in range(0, max(size, 1), CHUNK):
                    out.append((p, os.path.relpath(p, root), s, min(s + CHUNK, size)))
    out.sort(key=lambda t: -(t[3] - t[2]))
    return out


def fmt(s, n, cat):
    try:
        return _fmt(s, n, cat)
    except Exception:  # broken shape: show the raw row
        head = ["=" * 110, f"{cat} SAMPLE {n}", f"file: {s['file']}", f"byte offset of the line: {s['byte']}", "=" * 110]
        return "\n".join(head) + "\n[raw row, shape is broken]\n" + json.dumps(s["row"], ensure_ascii=False, indent=1) + "\n\n"


def _fmt(s, n, cat):
    out = ["=" * 110, f"{cat} SAMPLE {n}", f"file: {s['file']}", f"byte offset of the line: {s['byte']}", "=" * 110]
    r = s["row"]
    if r.get("tools"):
        out.append("[tools] " + json.dumps(r["tools"], ensure_ascii=False))
    for i, m in enumerate(r.get("messages") or []):
        out.append(f"\n----- message {i}  role: {m.get('role')} -----")
        if m.get("reasoning_content"):
            out.append("[reasoning_content]:\n" + m["reasoning_content"] + "\n[content]:")
        c = m.get("content")
        out.append(c if isinstance(c, str) else json.dumps(c, ensure_ascii=False))
        if m.get("tool_calls"):
            out.append("[tool_calls] " + json.dumps(m["tool_calls"], ensure_ascii=False))
    return "\n".join(out) + "\n\n"


def malformed(r):
    schemas = {}
    for t in r.get("tools") or []:
        f = (t or {}).get("function") or {}
        if f.get("name"):
            schemas[f["name"]] = f.get("parameters") or {}
    for m in r.get("messages") or []:
        for tc in m.get("tool_calls") or []:
            fn = (tc or {}).get("function") or {}
            try:
                a = json.loads(fn.get("arguments") or "")
            except Exception:
                return True
            if not isinstance(a, dict):
                return True
            req = (schemas.get(fn.get("name")) or {}).get("required") or []
            if isinstance(req, list) and any(k not in a for k in req if isinstance(k, str)):
                return True
    return False
