"""Source and identity rules (from the first clean of split/high).
is_incorrect_source: rows from datasets of known-wrong solutions.
has_first_person_identity: OLD broad identity rule - many false positives ("I'm using OpenAI's API").
Used as info only (IDENTITY_userrule); the strict rule is IDENT in dataset_check.py."""
import re

# First-person identity claims and model self-identification
IDENT_FIRST_PERSON = re.compile(
    r"\b(?:I am|I'm|I was (?:created|developed|trained|made|built) by|As an AI (?:developed|trained|created|built) by|"
    r"(?:you can )?call me|my name is)\b"
    r"[^.\n]{0,60}\b(?:ChatGPT|OpenAI|GPT-[345]|Claude|Anthropic|Qwen|Alibaba|DeepSeek|Gemini|Google DeepMind|Meta AI|"
    r"LLaMA|Llama|Mistral|Kimi|Moonshot|GLM|Zhipu|Grok|xAI|NVIDIA|MBZUAI|K2(?: Think)?|Baichuan|InternLM)\b",
    re.I
)

IDENT_MODEL_BY = re.compile(
    r"\b(?:I am|I'm)\s+(?:a|an)?\s*[^.\n]{0,50}\b(?:developed|trained|created|built)\s+by\s+"
    r"(?:NVIDIA|OpenAI|Anthropic|Mistral|Google|Meta|Alibaba|DeepSeek|Zhipu|MBZUAI)\b",
    re.I
)

INCORRECT_SRC_PATTERNS = ("OpenMathInstruct-1__incorrect_solutions", "/incorrect-", "kodcode__incorrect")

def is_incorrect_source(rel_path):
    return any(p in rel_path for p in INCORRECT_SRC_PATTERNS)


def has_first_person_identity(msgs):
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        c = m.get("content") or ""
        rc = m.get("reasoning_content") or ""
        text = str(c) + " " + str(rc)
        if IDENT_FIRST_PERSON.search(text) or IDENT_MODEL_BY.search(text):
            return True
    return False
