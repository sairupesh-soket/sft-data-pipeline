"""Tool-call repairs.
link_tool_ids_step : ported from hf_agentic_pipeline/scripts/fix_countdown_and_tools.py (fix_tools, unchanged)
    - a tool reply after an assistant turn with structured tool_calls but no tool_call_id gets the id of the
      matching call (in order)
    - a tool reply with no structured call before it (the call was written as text) becomes role "user"
hermes_step : ported from think_scan_deduped/check3/convert_hermes.py (convert, unchanged)
    Hermes text tool calls (<tool_call>{json}</tool_call> in assistant content, <tool_response> in the next user
    message, tools listed in the system prompt) -> OpenAI tool_calls + role "tool" messages. All or nothing per row.
"""
import ast
import json
import re

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


def link_tool_ids_step(row, c):
    if not any(m.get("role") == "tool" for m in row.get("messages") or []):
        return "keep", row
    return ("changed" if fix_tools(row, c) else "keep"), row


# ---------------- Hermes text tool calls -> OpenAI ----------------
TC = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
TR = re.compile(r"<tool_response[^>]*>\s*(.*?)\s*</tool_response>", re.S)
MD_TOOL = re.compile(r"^##\s+(\S+)\s*\n+Description:\s*(.*?)\n+Parameters:\s*\n```json\s*\n(.*?)\n```", re.S | re.M)
TOOLS_BLOCK = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.S)


def _lit(s):
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return None


def _norm_tool(t):
    if not isinstance(t, dict):
        return None
    f = t.get("function") if isinstance(t.get("function"), dict) else t
    name = f.get("name")
    if not isinstance(name, str) or not name:
        return None
    params = f.get("parameters", f.get("input_schema"))
    if isinstance(params, dict) and "properties" not in params and params and all(isinstance(v, dict) for v in params.values()):
        params = {"type": "object", "properties": params, "required": f.get("required") or []}
    if not isinstance(params, dict):
        params = {"type": "object", "properties": {}}
    return {"type": "function", "function": {"name": name, "description": str(f.get("description") or ""), "parameters": params}}


def parse_tools(system):
    tools = []
    for m in MD_TOOL.finditer(system):
        p = _lit(m.group(3))
        if not isinstance(p, dict):
            return None
        tools.append({"type": "function", "function": {"name": m.group(1), "description": m.group(2).strip(), "parameters": p}})
    if tools:
        return tools
    if "## " in system and "Parameters:" in system:
        for sec in re.split(r"(?m)^## ", system)[1:]:
            lines = sec.split("\n")
            name = lines[0].strip()
            body = "\n".join(lines[1:])
            if "Description:" not in body or "Parameters:" not in body or not re.fullmatch(r"[\w.\-]+", name):
                continue
            desc = body.split("Description:", 1)[1].rsplit("Parameters:", 1)[0].strip()
            props, req = {}, []
            for pl in body.rsplit("Parameters:", 1)[1].split("\n"):
                mm = re.match(r"\s*- (\w+) \((required|optional)(?:, ([^)]+))?\):\s*(.*)", pl)
                if not mm:
                    continue
                props[mm.group(1)] = {"type": (mm.group(3) or "string").strip(), "description": mm.group(4).strip()}
                if mm.group(2) == "required":
                    req.append(mm.group(1))
            tools.append({"type": "function", "function": {"name": name, "description": desc,
                          "parameters": {"type": "object", "properties": props, "required": req}}})
        if tools:
            return tools
    for b in TOOLS_BLOCK.findall(system):
        if not b.strip():
            continue
        v = _lit(b)
        if isinstance(v, list):
            items = v
        elif isinstance(v, dict):
            items = [v]
        else:
            items = [_lit(x) for x in b.splitlines() if x.strip()]
        for it in items:
            t = _norm_tool(it)
            if t is None:
                return None
            tools.append(t)
        if tools:
            return tools
    m = re.search(r"following tools[^\n]*\n\s*\[", system)
    if m:
        try:
            v, _ = json.JSONDecoder().raw_decode(system[m.end() - 1:])
        except Exception:
            v = None
        if isinstance(v, list):
            for it in v:
                t = _norm_tool(it)
                if t is None:
                    return None
                tools.append(t)
            return tools or None
    return None


def convert(row):
    msgs = row.get("messages") or []
    if not msgs or msgs[0].get("role") != "system":
        return None, "no_system_tools"
    tools = parse_tools(msgs[0].get("content") or "")
    if not tools:
        return None, "tools_not_parsed"
    names = {t["function"]["name"] for t in tools}
    out = []
    k = 0
    i = 1
    n_calls = 0
    while i < len(msgs):
        m = msgs[i]
        c = m.get("content") if isinstance(m.get("content"), str) else ""
        if m.get("role") == "assistant" and "<tool_call>" in c:
            blocks = TC.findall(c)
            tail = c[c.rfind("</tool_call>") + len("</tool_call>"):]
            if not blocks or tail.strip():
                return None, "text_after_tool_call"
            text = c[:c.find("<tool_call>")].strip()
            if text.endswith(":"):  # "...\n: \n<tool_call>" artifacts
                text = text[:-1].strip()
            calls = []
            for b in blocks:
                j = _lit(b)
                if not isinstance(j, dict) or not isinstance(j.get("name"), str):
                    return None, "tool_call_not_json"
                if j["name"] not in names:
                    return None, "tool_not_in_tools"
                args = j.get("arguments", {})
                if isinstance(args, str):
                    a2 = _lit(args)
                    if not isinstance(a2, dict):
                        return None, "arguments_not_object"
                    args = a2
                if not isinstance(args, dict):
                    return None, "arguments_not_object"
                calls.append({"id": f"call_{k}", "type": "function",
                              "function": {"name": j["name"], "arguments": json.dumps(args, ensure_ascii=False)}})
                k += 1
            nm = {kk: vv for kk, vv in m.items() if kk not in ("content", "tool_calls")}
            nm["content"] = text
            nm["tool_calls"] = calls
            out.append(nm)
            n_calls += len(calls)
            # collect tool results from the following user message(s)
            results = []
            i += 1
            while i < len(msgs) and msgs[i].get("role") == "user" and len(results) < len(calls):
                u = msgs[i].get("content") if isinstance(msgs[i].get("content"), str) else ""
                rs = TR.findall(u)
                if rs:
                    if TR.sub("", u).strip():
                        return None, "user_text_besides_tool_response"
                    results += rs
                elif u.lstrip().startswith("Result:"):
                    results.append(u.lstrip()[len("Result:"):].strip())
                else:
                    break
                i += 1
            if len(results) != len(calls):
                return None, "tool_result_count_mismatch"
            for call, res in zip(calls, results):
                out.append({"role": "tool", "tool_call_id": call["id"], "content": res})
            continue
        if m.get("role") == "user" and ("<tool_response" in c or "</tool_response>" in c):
            return None, "orphan_tool_response"
        out.append(m)
        i += 1
    if n_calls == 0:
        return None, "no_tool_calls"
    if not out or out[-1].get("role") != "assistant" or out[0].get("role") != "user":
        return None, "bad_structure_after"
    new = {"messages": out, "tools": tools}
    return new, "converted"


def hermes_step(row, c):
    msgs = row.get("messages") or []
    if not any(m.get("role") == "assistant" and "<tool_call>" in (m.get("content") or "") for m in msgs if isinstance(m.get("content"), str)):
        return "keep", row
    new, why = convert(row)
    c["hermes:" + why] += 1
    if new is None:
        return "drop:hermes_" + why, None
    return "changed", new
