"""Filtering and scrubbing for OpenAI SFT conversion."""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Vendor / company-centric system prompts (drop whole sample)
# ---------------------------------------------------------------------------
VENDOR_SYSTEM_PATTERNS = [
    # Anthropic / Claude
    r"\bclaude\b",
    r"\banthropic\b",
    r"\bconstitutional\s+ai\b",
    # Google
    r"\bgemini\b",
    r"\bbard\b",
    r"\bgoogle\s+(ai|assistant|deepmind)\b",
    r"\bdeepmind\b",
    # Meta
    r"\bllama\s*[0-9]",
    r"\bmeta\s+ai\b",
    # xAI / others
    r"\bgrok\b",
    r"\bxai\b",
    r"\bcopilot\b",
    r"\bbing\s+chat\b",
    r"\bperplexity\b",
    r"\bmistral\s+(ai|large|medium|small)\b",
    r"\bqwen\b",
    r"\bdeepseek\b",
    r"\bcommand[- ]?r\b",
    r"\bcohere\b",
    # Self-identification phrasing
    r"\bi(?:'m| am)\s+(?:claude|gemini|bard|grok|llama)\b",
    r"\byou are\s+(?:claude|gemini|bard|grok|llama)\b",
    r"\bas\s+(?:claude|gemini|bard|grok)\b",
]

VENDOR_SYSTEM_RE = re.compile("|".join(VENDOR_SYSTEM_PATTERNS), re.IGNORECASE)

# ---------------------------------------------------------------------------
# Model-name scrubbing in message content (keep tool ids / tool names)
# ---------------------------------------------------------------------------
# Longer / more specific first to avoid partial leftovers.
MODEL_NAME_PATTERNS = [
    r"\bgpt-4o(?:-mini)?\b",
    r"\bgpt-4\.1(?:-mini|-nano)?\b",
    r"\bgpt-4(?:-turbo)?\b",
    r"\bgpt-3\.5(?:-turbo)?\b",
    r"\bo[134](?:-mini|-pro)?\b",
    r"\bchatgpt\b",
    r"\bopenai\b",
    r"\bclaude[- ]?(?:opus|sonnet|haiku)?(?:[- ]?\d+(?:\.\d+)*)?\b",
    r"\banthropic\b",
    r"\bgemini[- ]?(?:pro|flash|ultra)?(?:[- ]?\d+(?:\.\d+)*)?\b",
    r"\bllama[- ]?\d+(?:\.\d+)?(?:[- ]?\d+[bB])?\b",
    r"\bqwen(?:2(?:\.\d+)?)?(?:[- ]?\d+[bB])?\b",
    r"\bdeepseek[- ]?(?:r1|v\d+|chat|coder)?\b",
    r"\bmistral[- ]?(?:large|medium|small|nemo)?\b",
    r"\bgrok[- ]?\d*\b",
    r"\bqwq[- ]?\d+[bB]?\b",
    r"\bsmollm\d*\b",
    r"\bphi[- ]?\d+(?:\.\d+)?\b",
    r"\bcommand[- ]?r(?:\+|plus)?\b",
]

MODEL_NAME_RE = re.compile("|".join(MODEL_NAME_PATTERNS), re.IGNORECASE)

# Collapse leftover whitespace after scrubbing
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_SCRUB_TOKEN = "[MODEL]"


def is_vendor_centric_system(text: str | None) -> bool:
    if not text or not isinstance(text, str):
        return False
    return bool(VENDOR_SYSTEM_RE.search(text))


def sample_has_vendor_system(messages: list[dict[str, Any]]) -> bool:
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "system" and is_vendor_centric_system(m.get("content")):
            return True
    return False


def scrub_model_names(text: str | None) -> str | None:
    """Remove model/vendor name mentions from free text. Does not touch tool ids."""
    if text is None:
        return None
    if not isinstance(text, str):
        return text
    scrubbed = MODEL_NAME_RE.sub(_SCRUB_TOKEN, text)
    # Drop placeholder tokens entirely (avoid leaving "[MODEL]" litter)
    scrubbed = re.sub(rf"\s*{re.escape(_SCRUB_TOKEN)}\s*", " ", scrubbed)
    scrubbed = _MULTI_SPACE_RE.sub(" ", scrubbed)
    scrubbed = re.sub(r"\s+\n", "\n", scrubbed)
    return scrubbed.strip()


def scrub_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Scrub model names from content/reasoning only; preserve tool_call ids/names."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        nm = dict(m)
        if "content" in nm and isinstance(nm["content"], str):
            nm["content"] = scrub_model_names(nm["content"])
        if "reasoning_content" in nm and isinstance(nm["reasoning_content"], str):
            nm["reasoning_content"] = scrub_model_names(nm["reasoning_content"])
        # Do NOT scrub tool_calls / tool_call_id / name / id
        out.append(nm)
    return out
