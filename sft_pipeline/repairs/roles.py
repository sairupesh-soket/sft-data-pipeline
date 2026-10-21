"""Role rules.
role_rules_step : ported from sft_audit/sft_clean.py (role_rules, unchanged)
    - empty assistant turns (no text, no tool call) removed
    - row trimmed so that it ends on its last assistant turn; dropped if there is no assistant turn
    - row dropped when the system prompt is followed directly by an assistant turn
    Note: with reasoning in reasoning_content, an assistant turn with only reasoning and no text is "empty".
harness_fallback_step : ported from sft_traces_v1/scripts/fix_harness_fallback_text.py
    - final assistant text is the agent harness fallback ("Technical difficulties. Please continue with the task.")
      -> trim back to the last real assistant turn; drop when none is left
    (the original wrote {"messages": ...} only and lost "tools"; this port keeps "tools")
"""


def role_rules(msgs, c):
    kept = [m for m in msgs if not (m["role"] == "assistant" and not m.get("tool_calls")
                                    and not (m.get("content") or "").strip())]
    if len(kept) != len(msgs):
        c["empty_assistant_removed"] += len(msgs) - len(kept)
    last = max((i for i, m in enumerate(kept) if m["role"] == "assistant"), default=-1)
    if last < 0:
        c["dropped_no_assistant"] += 1
        return None
    if len(kept) > 1 and kept[0]["role"] == "system" and kept[1]["role"] == "assistant":
        c["dropped_system_then_assistant"] += 1
        return None
    if last < len(kept) - 1:
        c["trimmed_to_last_assistant"] += 1
    return kept[: last + 1]


def role_rules_step(row, c):
    msgs = row.get("messages") or []
    if not msgs:
        return "drop:empty_messages", None
    new = role_rules(msgs, c)
    if new is None:
        return "drop:role_rules", None
    if len(new) != len(msgs):
        return "changed", dict(row, messages=new)
    return "keep", row


FALLBACK_TEXT = "Technical difficulties. Please continue with the task."


def harness_fallback_step(row, c):
    msgs = row.get("messages") or []
    last = msgs[-1] if msgs else {}
    if not (last.get("role") == "assistant" and FALLBACK_TEXT in str(last.get("content", ""))):
        return "keep", row
    good_idx = None
    for i in range(len(msgs) - 2, -1, -1):
        m = msgs[i]
        if m.get("role") == "assistant" and FALLBACK_TEXT not in str(m.get("content", "")):
            good_idx = i
            break
    if good_idx is None:
        return "drop:harness_fallback_only", None
    c["harness_fallback_trimmed"] += 1
    return "changed", dict(row, messages=msgs[: good_idx + 1])
