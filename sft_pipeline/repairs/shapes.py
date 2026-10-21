"""Shape repairs. Ported from sft_audit/fixes/fix_20260924_shapes.py (logic unchanged).
  1. flat tool definition {name, description, parameters} -> {"type": "function", "function": {...}}
  2. Kimi tool-declare system message (<|im_system|>tool_declare<|im_middle|>...<|im_end|>) -> removed
     (the same tools are in the row's "tools" list)
  3. "null" visible answer while <think> holds the real reply -> the <think> text becomes the answer
  4. <answer>...</answer> trapped inside <think> with nothing after </think> -> moved after </think>
  5. content that is not a string, or a final assistant turn that shows nothing -> row dropped
"""
import re

KIMI_DECL = re.compile(r"^\s*<\|im_system\|>tool_declare<\|im_middle\|>(.*)<\|im_end\|>\s*$", re.S)
ANSWER_TAG = re.compile(r"<answer>.*?</answer>", re.S)


def after_think(c):
    return c.split("</think>")[-1] if "</think>" in c else c


def fix_row(d, c):
    changed = False
    # 1. flat tool definitions
    tools = d.get("tools")
    if tools:
        new = []
        for t in tools:
            if isinstance(t, dict) and "function" not in t and "name" in t:
                t = {"type": "function", "function": {k: v for k, v in t.items() if k != "type"}}
                c["tool_def_wrapped"] += 1
                changed = True
            if isinstance(t, dict) and isinstance(t.get("function"), dict) and "parameters" not in t["function"]:
                t["function"]["parameters"] = {"type": "object", "properties": {}}
                changed = True
            new.append(t)
        d["tools"] = new
    msgs = []
    for m in d["messages"]:
        cont = m.get("content")
        # 5a. non-string content -> drop row
        if not isinstance(cont, str):
            c["row_dropped_non_string_content"] += 1
            return None, True
        # 2. Kimi tool-declare system message
        if m.get("role") == "system" and KIMI_DECL.match(cont):
            c["kimi_tool_declare_removed"] += 1
            changed = True
            continue
        if m.get("role") == "assistant" and "</think>" in cont and not m.get("tool_calls"):
            vis = after_think(cont).strip()
            think = cont.split("</think>")[0].replace("<think>", "", 1).strip()
            # 3. "null" visible answer, real reply inside <think>
            if vis == "null" and think:
                m = dict(m, content=think)
                c["null_answer_unwrapped"] += 1
                changed = True
            # 4. <answer> tag trapped inside <think>
            elif not vis and ANSWER_TAG.search(think):
                tag = ANSWER_TAG.findall(think)[-1]
                i = think.rfind(tag)
                think = (think[:i] + think[i + len(tag):]).rstrip()
                m = dict(m, content=f"<think>{think}</think>\n{tag}")
                c["answer_tag_moved_out_of_think"] += 1
                changed = True
        msgs.append(m)
    d["messages"] = msgs
    # 5b. final assistant turn still shows nothing -> drop
    last = msgs[-1] if msgs else {}
    if last.get("role") == "assistant" and not last.get("tool_calls") and not after_think(last.get("content", "")).strip():
        c["row_dropped_final_answer_empty"] += 1
        return None, True
    return d, changed


def step(row, c):
    before = c.copy()
    new, changed = fix_row(row, c)
    if new is None:
        why = "non_string_content" if c["row_dropped_non_string_content"] > before["row_dropped_non_string_content"] else "final_answer_empty"
        return "drop:" + why, None
    return ("changed" if changed else "keep"), new
