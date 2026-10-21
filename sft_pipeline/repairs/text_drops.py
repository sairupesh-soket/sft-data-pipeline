"""Final-answer text drops. Ported from scripts_used/step2_fix2/fix2.py + check2.py (rules unchanged).
  - final answer empty                                   -> drop
  - final answer starts like raw R1 self-talk ("Okay, let me", "Hmm", "I need to solve") -> drop
  - final answer cut off (tier T1: ends on a function word or a comma, >= 300 chars) -> drop
  - Odia garbage pattern (only for files you pass with odia=True)                     -> drop
Only the final assistant turn without tool_calls is checked.
"""
import re

SELF_TALK = re.compile(r"^\s*(I need to (solve|figure|determine|find)|I have (this|a) (problem|question)|Okay, so I|Okay, let me|"
                       r"Okay, I need|Okay, let's|Hmm|So I've got this|Alright, so I|Alright, let's|Let me think|Let me try to figure)")
FUNC_WORDS = {"the", "and", "of", "to", "a", "an", "in", "with", "for", "is", "that", "by", "on", "or", "as", "are", "be", "was", "from", "at", "its", "their"}
TERMINAL = set(".!?。！？।॥؟\"'”’»)]}>*`|:;…-–—~_$#%")
F_PATTERNS = {
    "F_list_repr_answer": re.compile(r"^\s*\[\s*'.*'\s*\]\s*$|^\s*\[\s*\".*\"\s*\]\s*$", re.S),
    "F_bytes_repr": re.compile(r"^\s*b'.*'\s*$|^\s*b\".*\"\s*$", re.S),
    "F_dict_repr_answer": re.compile(r"^\s*\{\s*'[A-Za-z_]+'\s*:", re.S),
    "F_role_marker_in_user": re.compile(r"^\s*(USER|SYSTEM|HUMAN|ASSISTANT|Human|User|Assistant)\s*:", re.M),
}
ODIA_BAD = re.compile(r"(\\u200c|\\ ){4,}|(\\ \S? ){6,}|(\S \\ ){6,}|\\u200c.{0,4}\\u200c.{0,4}\\u200c")

def trunc_tier(a):
    t = a.rstrip()
    if len(t) < 300:
        return None
    if t.count("```") % 2 == 1:
        return "T2_unclosed_code_fence"
    last = t[-1]
    if last in TERMINAL or last.isdigit():
        return None
    m = re.search(r"([A-Za-z]+)$", t)
    if m and m.group(1).lower() in FUNC_WORDS:
        return "T1_ends_on_function_word"
    if last == ",":
        return "T1_ends_on_comma"
    if last.isalpha() and len(t) >= 1000:
        return "T3_ends_mid_text_no_punct"
    return None


def step(row, c, odia=False):
    msgs = row.get("messages") or []
    last = msgs[-1] if msgs else {}
    fin = last.get("content") if isinstance(last.get("content"), str) else ""
    if last.get("role") == "assistant" and not last.get("tool_calls"):
        if odia and ODIA_BAD.search(fin):
            return "drop:odia_garbage", None
        if not fin.strip():
            return "drop:final_empty", None
        if SELF_TALK.match(fin):
            return "drop:untagged_selftalk", None
        tt = trunc_tier(fin)
        if tt and tt.startswith("T1"):
            return "drop:truncated_T1", None
    return "keep", row
