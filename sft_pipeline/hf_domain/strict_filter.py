#!/usr/bin/env python3
"""
sft_clean_filter.py -- ONE script that turns the language-split SFT jsonl files into clean,
strict OpenAI chat-format SFT data.

Output row:  {"messages": [ {"role": "system|user|assistant|tool", "content": str,
                              "tool_calls": [...]  (assistant only, optional),
                              "tool_call_id": str  (tool only, optional)} ... ],
              "tools": [...]  (optional, only when a tool schema could be parsed)}

Every input row is either
  * REPAIRED and kept   (safe, mechanical fixes -- counted under repairs), or
  * REJECTED with a reason (counted, and a few examples saved in the report), or
  * kept unchanged.
Nothing is silently altered beyond the repairs listed below. Input files are read-only.

Rejections (reason keys)                          Repairs (repair keys)
  parse_error / no_messages / bad_role              repr_parts_joined       (content stored as "[{'type':'text',...}]")
  empty_content / bad_content_type                  repr_conversation_unwrapped (assistant turn = dump of whole chat)
  multimodal_image_part / nested_conversation_in_prompt          duplicate_think_tag_collapsed  (<think><think> / </think></think>)
  think_empty  (nothing between the tags)           answer_tag_unwrapped    (<answer>..</answer>)
  think_junk   (dots / punctuation only)            hermes_tool_calls_converted (<tool_call> text -> tool_calls field)
  think_unbalanced / think_no_answer                think_stripped          (only with --think strip)
                                                    nested_think_repaired   (<think><think>A</think> ans</think> ans)
                                                    empty_think_stripped    (only with --empty-think strip)
  tool_result_without_call / tool_mismatch
  tool_call_unbalanced / tool_call_unparseable / tool_response_missing
  tool_call_dropped  (function prompt, assistant says it calls a function, call missing)
  seq_invalid  (roles out of order / last turn not a plain assistant answer)
  special_token_in_text (leaked chat-template control string)      special_token_end_stripped (trailing <|im_end|> / <|eot_id|> / </s>)
  invalid_unicode / control_or_replacement_char / single_char_run / dot_run / repetition_loop
  placeholder_answer / answer_equals_question / qa_no_word_overlap (strict lexical mismatch)
  prompt_missing_payload (instruction points at "the following text:" but the text is absent)
  missing_context_placeholder (question holds a bare <image>/<DNA>/<RNA>/<protein>/<smiles>/<mol> tag, the real input was dropped upstream)
  missing_context_cutoff (question stops mid-sentence AND the assistant says it is cut off / incomplete)
  too_long / exact_duplicate / prompt_cap

Usage
  python3 sft_clean_filter.py --selftest
  python3 sft_clean_filter.py --in DIR_OR_FILE [--in ...] --sample 200          # dry run: stats + examples, writes nothing
  python3 sft_clean_filter.py --in sft_43_language_wise/hi --out CLEAN_DIR      # real run, mirrors folder layout
Options: --think keep|strip  --empty-think drop|strip  --max-chars N  --max-per-prompt K
         --lexical-mismatch (off by default)  --code-tool NAME  --workers N  --save-rejects
"""
import argparse, ast, collections, glob, hashlib, ipaddress, json, os, random, re, sys, time
from multiprocessing import Pool

ROLES = ("system", "user", "assistant", "tool")
KEEP_MSG_KEYS = ("role", "content", "tool_calls", "tool_call_id", "name")

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.S | re.I)
ANSWER_TAG = re.compile(r"</?answer>", re.I)
HERMES_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
HERMES_RESP = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.S)
TOOLS_BLOCK = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.S)
FUNC_PROMPT = re.compile(r"<functions>|<tools>|function[- ]calling|function signatures|access to the following (?:apis|functions|tools)", re.I)
DROPPED_MENTION = re.compile(r"\bI (?:need|will|should|have|am going|'ll) (?:to )?call (?:the )?[`'\"]?\w+|\bBy calling (?:this|the)\b|\bcall the [`'\"]?[\w\.]+[`'\"]? function\b", re.I)
CODE_FENCE = re.compile(r"```.*?```", re.S)
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f�]")
LETTER_RUN = re.compile(r"([^\W\d_])\1{24,}")
DOT_RUN = re.compile(r"(?:\.\s){30,}|\.{300,}|\u2026{100,}")
MISSING_PAYLOAD = re.compile(r"(?:following|below|given)\s+(?:text|passage|snippet|excerpt|article|paragraph|document|context|abstract|sentence|story|extract)s?(?:\s+snippet)?\s*:?\s*$", re.I)
SPECIAL_TOK = re.compile(r"<\|(?:im_start|im_end|im_sep|endoftext|eot_id|eom_id|start_header_id|end_header_id|begin_of_text|end_of_text|python_tag|fim_[a-z]+|pad|reserved_special_token_\d+)\|>|\[/?INST\]|<</?SYS>>")
END_TOK = re.compile(r"(?:\s*(?:<\|im_end\|>|<\|eot_id\|>|<\|eom_id\|>|<\|end_of_text\|>|<\|endoftext\|>|<\|im_sep\|>|</s>))+\s*$")
EMAIL_RE = re.compile(r"(?<![\w.+%\\-])[A-Za-z0-9][A-Za-z0-9._%+-]{0,63}@(?:[A-Za-z0-9-]{1,63}\.)+[A-Za-z]{2,24}(?![\w-])")
FAKE_EMAIL = re.compile(r"@(?:[\w.-]*\.)?(?:example|test|tests|domain|yourdomain|yourcompany|company|sample|foo|bar|placeholder|localhost|invalid|acme|mydomain|website|email|mail)\.(?:com|org|net|edu|io|co|local|invalid|test)$|@localhost$|^(?:user|name|you|your|username|email|john|jane|test|foo|bar|someone|sender|recipient)@", re.I)
PHONE_RE = re.compile(r"(?<![\w.+-])(?:\+\d{1,3}[\s.-]\(?\d{1,4}\)?(?:[\s.-]\d{2,4}){2,4}|\(\d{3}\)\s?\d{3}[\s.-]\d{4}|\b\d{3}[.-]\d{3}[.-]\d{4})(?![\w-])")
SECRET_RE = re.compile(r"AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{36}|sk-[A-Za-z0-9]{32,}|xox[baprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35}|-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----[\s\S]*?(?:-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----|$)|eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")
CARD_RE = re.compile(r"(?<!\d )(?<![\d.\-/=_:\\])(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6011|35\d{2})[ -]\d{4}[ -]\d{4}[ -]\d{2,4}(?![\d.\-/_])(?! \d)")
CARD_CTX = re.compile(r"((?:card|visa|mastercard|amex|credit|debit)\D{0,25}?)((?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6011)\d{9,15})(?![\d.\-/_])", re.I)
SSN_CTX = re.compile(r"((?:ssn|social security(?: number)?)\D{0,20})(\d{3}-\d{2}-\d{4})", re.I)
REPR_START = re.compile(r"^\s*\[\s*\{\s*['\"]")
STOP = frozenset("the a an of to in and or is are was were be for on with as by at from that this it its what how why when which who do does did can could would should you we they he she my your our their not yes than then so if but about into over under between during versus".split())
ALIAS = {"human": "user", "gpt": "assistant", "bot": "assistant", "model": "assistant", "function": "tool", "observation": "tool"}


class Reject(Exception):
    def __init__(self, reason):
        self.reason = reason


# ----------------------------------------------------------------------------- content repairs
def _parse_repr(s):
    s = s.strip()
    if len(s) < 8 or len(s) > 600000 or s[-1] != "]" or not REPR_START.match(s):
        return None
    for fn in (ast.literal_eval, json.loads):
        try:
            v = fn(s)
        except Exception:
            continue
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            return v
    return None


def _parts_to_text(parts):
    out = []
    for p in parts:
        t = str(p.get("type", "text")).lower()
        if "image" in t or "audio" in t or "video" in t:
            raise Reject("multimodal_image_part")
        txt = p.get("text", p.get("content"))
        if isinstance(txt, str) and txt.strip():
            out.append(txt)
    return "\n".join(out)


def normalize_content(c, rep):
    """content -> str (or None, or ('__NESTED__', list) for a whole-conversation dump)."""
    if c is None:
        return None
    if isinstance(c, list):
        if not all(isinstance(x, dict) for x in c):
            raise Reject("bad_content_type")
        rep["repr_parts_joined"] += 1
        return _parts_to_text(c)
    if not isinstance(c, str):
        raise Reject("bad_content_type")
    v = _parse_repr(c) if c.lstrip()[:1] == "[" else None
    if v is not None:
        if all(("role" in x and ("content" in x or "value" in x)) for x in v):
            return ("__NESTED__", v)
        if all(("type" in x or "text" in x) for x in v):
            rep["repr_parts_joined"] += 1
            return _parts_to_text(v)
    return c


def fix_special(text, rep):
    """chat-template control strings that leaked into the text would be read as real special tokens by a tokenizer."""
    t = END_TOK.sub("", text)
    if t != text:
        rep["special_token_end_stripped"] += 1
    if SPECIAL_TOK.search(t):
        raise Reject("special_token_in_text")
    return t


def _luhn(d):
    d = [int(c) for c in d][::-1]
    return (sum(d[0::2]) + sum(sum(divmod(2 * x, 10)) for x in d[1::2])) % 10 == 0


def redact_pii(text, rep):
    """conservative structured-PII redaction (emails, phones, secrets, cards, ssn). Names/addresses need NER and are NOT handled."""
    def go(rx, ph, key, ok):
        nonlocal text
        def f(m):
            if not ok(m.group(0)):
                return m.group(0)
            rep[key] += 1
            return ph
        text = rx.sub(f, text)
    go(SECRET_RE, "<SECRET>", "pii_secret_redacted", lambda x: "EXAMPLE" not in x.upper())
    go(EMAIL_RE, "<EMAIL>", "pii_email_redacted", lambda x: not FAKE_EMAIL.search(x))
    go(CARD_RE, "<CARD_NUMBER>", "pii_card_redacted", lambda x: (lambda d: 13 <= len(d) <= 19 and len(set(d)) > 3 and _luhn(d))(re.sub(r"\D", "", x)))
    def card_ctx(m):
        d = m.group(2)
        if 13 <= len(d) <= 19 and len(set(d)) > 3 and _luhn(d):
            rep["pii_card_redacted"] += 1
            return m.group(1) + "<CARD_NUMBER>"
        return m.group(0)
    text = CARD_CTX.sub(card_ctx, text)
    def ssn(m):
        rep["pii_ssn_redacted"] += 1
        return m.group(1) + "<SSN>"
    text = SSN_CTX.sub(ssn, text)
    go(PHONE_RE, "<PHONE>", "pii_phone_redacted", lambda x: (lambda d: 9 <= len(d) <= 15 and len(set(d)) > 3 and not re.search(r"(\d)\1{5,}", d) and "1234567" not in d and not re.search(r"555[\s.-]?01\d\d", x))(re.sub(r"\D", "", x)))
    return text


# ----------------------------------------------------------------------------- missing-context rules
# Rows whose question points at content that is NOT in the row (the source kept it in a separate column that the conversion
# dropped, or the question itself was cut off). Both rules were read-checked on real rows.
#   placeholder: a bare <image>/<DNA>/<RNA>/<protein>/<smiles>/<mol> tag in a USER message. If a real sequence / SMILES follows the
#                tag, the tag is only a delimiter (e.g. "<protein> M P K G ...") and the row is KEPT. <seq>, <img>, <audio>, <video>,
#                <table> are NOT used: <seq> is a delimiter with the sequence inline; the others are ordinary HTML tags in code questions.
#   cutoff:      needs TWO signals -- the first user message is one short line that stops mid-sentence AND the assistant's first 700
#                chars say the question is incomplete / cut off. Either signal alone is only ~60% right.
PH_TAG = re.compile(r"(?<![`\w/])<(image|DNA|RNA|protein|smiles|mol)>(?![`\w])", re.I)
SEQ_NUC_PROT = re.compile(r"\s*(?:[ACGTUNacgtun]{12,}|(?:[A-Z] ){10,}|[A-Z]{12,})")
SEQ_SMILES = re.compile(r"\s*(?=\S*[=()\[\]#@\d])\S{12,}")
END_OK = tuple(".?!:;)]}\"'\u201d\u2019`>") + ("\u2026", "\u3002", "\uff1f", "\uff01")
CUT_STR = re.compile(r"(question|sentence|prompt|query|message|text|input|request|statement|title)\s+(seems|appears|looks|is|was|got|has been)\s+(to be\s+|like it('s| is| was)\s+)?(incomplete|cut off|cut-off|truncated|cut short|unfinished|abruptly)|(seems|appears|looks)\s+(to be\s+)?(incomplete|cut off|cut-off|truncated|unfinished)|(incomplete|truncated|cut[- ]off)\s+(question|sentence|prompt|query|message|text|input)|ends abruptly|trails off|cut off mid", re.I)


def missing_context_tag(c):
    for t in PH_TAG.finditer(c):
        tag = t.group(1).lower(); rest = c[t.end(): t.end() + 60]
        if tag in ("dna", "rna", "protein") and SEQ_NUC_PROT.match(rest):
            continue
        if tag in ("smiles", "mol") and SEQ_SMILES.match(rest):
            continue
        return True
    return False


def missing_context_cutoff(msgs):
    u = next((m.get("content") for m in msgs if m.get("role") == "user"), None)
    if not isinstance(u, str):
        return False
    u = u.strip()
    if not (15 <= len(u) <= 220) or "\n" in u or u.endswith(END_OK):
        return False
    a = next((m.get("content") for m in msgs if m.get("role") == "assistant"), None)
    return isinstance(a, str) and bool(CUT_STR.search(a[:700]))


# ----------------------------------------------------------------------------- PII, DataTrove / FineWeb style
# Same idea as datatrove's PIIFormatter (used for FineWeb): only e-mail addresses and PUBLIC ip addresses, replaced in place by
# reserved/non-responding fake values; the row is never dropped. Differences, each measured on real SFT rows (datatrove's
# regexes were built for web pages and rewrite math/versions/code here):
#   * e-mail regex = this file's EMAIL_RE (does not match "\n@pytest.fixture" or "threads.com/@handle"); placeholder domains kept
#   * ip must not be inside a longer dotted number ("3.3.3.3.6", "1.2.3.4.5"), not follow "version"/"v"/"==", not be a well-known
#     DNS address, and must have IP-like context (a URL, or a word such as ip/address/host/server/ssh/ping/dns) just before it
#   * the fake value is chosen from the matched text (same input -> same output) so parallel workers stay reproducible
FAKE_EMAILS = ("email@example.com", "firstname.lastname@example.org")
FAKE_IPS = ("22.214.171.124", "126.96.36.199", "188.8.131.52", "184.108.40.206", "220.127.116.11", "18.104.22.168")
IPV4 = re.compile(r"(?<![\w.=<>~-])((?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?))(?!\w|\.\d|/\d)")
IP_CTX = re.compile(r"://|\bip\b|ip[ _-]?addr|address|host|server|ssh|ping|dns|gateway|client|traceroute|nslookup|remote|whois|netstat|curl|wget|telnet|\u5730\u5740|\u670d\u52a1\u5668|\u4e3b\u673a", re.I)
VERSION_CTX = re.compile(r"(?:version|ver\.?|v)\s*$", re.I)
WELL_KNOWN_IPS = frozenset(("8.8.8.8", "8.8.4.4", "1.1.1.1", "1.0.0.1", "9.9.9.9", "149.112.112.112", "208.67.222.222", "208.67.220.220", "4.2.2.2"))


def _pick(options, s):
    return options[int.from_bytes(hashlib.blake2b(s.encode(), digest_size=2).digest(), "little") % len(options)]


def pii_datatrove(text, rep):
    if "@" in text:
        def em(m):
            if FAKE_EMAIL.search(m.group(0)):
                return m.group(0)
            rep["pii_email_replaced"] += 1
            return _pick(FAKE_EMAILS, m.group(0))
        text = EMAIL_RE.sub(em, text)

    def ip(m):
        s = m.group(1)
        if s in WELL_KNOWN_IPS or s.endswith(".0"):        # well-known DNS, or a network address ("1.128.0.0" of a block), not a host
            return s
        pre = m.string[max(0, m.start() - 60): m.start()]
        if VERSION_CTX.search(pre) or not IP_CTX.search(pre):
            return s
        try:
            if not ipaddress.ip_address(s).is_global:
                return s
        except ValueError:
            return s
        rep["pii_ip_replaced"] += 1
        return _pick(FAKE_IPS, s)
    return IPV4.sub(ip, text)


# ----------------------------------------------------------------------------- python tool schema
# Raw code blocks are converted into `python` tool calls (code_tool_call_converted). Those rows carry no tool schema, so a tool-aware
# chat template (GLM, Qwen, ...) would never tell the model that a `python` tool exists. Every such call has exactly one argument, `code`
# (checked on all 3.6M calls), so the schema is fully determined; attach it. Rows that already have a schema, or call anything else, are untouched.
PYTHON_TOOL = {"type": "function", "function": {"name": "python", "description": "Run Python code and return what it prints (stdout and stderr).",
               "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "The Python code to run."}}, "required": ["code"]}}}


def needs_python_tool(msgs, tools):
    if tools:
        return False
    n = 0
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if fn.get("name") != "python":
                return False
            try:
                a = json.loads(fn.get("arguments"))
            except Exception:
                return False
            if not isinstance(a, dict) or set(a) != {"code"}:
                return False
            n += 1
    return n > 0


def clean_role(r):
    r = str(r).strip().lower()
    return ALIAS.get(r, r)


def normalize_messages(msgs, rep):
    out = []
    for m in msgs:
        if not isinstance(m, dict):
            raise Reject("bad_content_type")
        role = clean_role(m.get("role"))
        if role not in ROLES:
            raise Reject("bad_role")
        c = normalize_content(m.get("content"), rep)
        nm = {"role": role, "content": c}
        for k in ("tool_calls", "tool_call_id", "name"):
            if m.get(k):
                nm[k] = m[k]
        out.append(nm)
    # an assistant turn that is a dump of the whole conversation -> unwrap it
    if out and isinstance(out[-1]["content"], tuple) and out[-1]["role"] == "assistant":
        nested = out[-1]["content"][1]
        lead = [x for x in out[:-1] if x["role"] == "system"]
        rebuilt = []
        for x in nested:
            r = clean_role(x.get("role"))
            cc = x.get("content", x.get("value"))
            if r not in ROLES or not isinstance(cc, str):
                raise Reject("repr_unparseable")
            rebuilt.append({"role": r, "content": cc})
        rep["repr_conversation_unwrapped"] += 1
        return lead + rebuilt
    if any(isinstance(x["content"], tuple) for x in out):
        raise Reject("nested_conversation_in_prompt")
    return out


# ----------------------------------------------------------------------------- think handling
def fix_think(text, role, cfg, rep):
    """returns cleaned text; raises Reject on unusable think structure."""
    low = text.lower()
    if "<think" not in low and "</think" not in low:
        return text
    if role != "assistant":
        return text                                  # e.g. a system prompt that describes the <think> format
    m2 = re.match(r"^\s*<think>\s*<think>(.*?)</think>(.*?)</think>(.*)$", text, re.S | re.I)
    if m2 and m2.group(3).strip() and m2.group(2).strip() == m2.group(3).strip():      # <think><think>A</think> ans</think> ans
        text = "<think>" + m2.group(1) + "</think>" + m2.group(3)
        rep["nested_think_repaired"] += 1
    if re.search(r"(?:<think>\s*){2,}|(?:</think>\s*){2,}", text, re.I):
        text = re.sub(r"(<think>)(?:\s*<think>)+", r"\1", text, flags=re.I)
        text = re.sub(r"(</think>)(?:\s*</think>)+", r"\1", text, flags=re.I)
        rep["duplicate_think_tag_collapsed"] += 1
    if text.lower().count(THINK_OPEN) != text.lower().count(THINK_CLOSE):
        raise Reject("think_unbalanced")
    for m in THINK_BLOCK.finditer(text):
        body = m.group(1)
        if not body.strip() or body.strip().lower() in ("none", "null", "n/a", "nan"):
            if cfg["empty_think"] == "drop":
                raise Reject("think_empty")
        elif sum(ch.isalnum() for ch in body) < 5:
            raise Reject("think_junk")
    if cfg["empty_think"] == "strip":
        before = len(THINK_BLOCK.findall(text))
        text = re.sub(r"<think>\s*(?:none|null|n/a|nan)?\s*</think>", "", text, flags=re.I).strip()
        if len(THINK_BLOCK.findall(text)) != before:
            rep["empty_think_stripped"] += 1
    if cfg["think"] == "strip":
        text = THINK_BLOCK.sub("", text).strip()
        rep["think_stripped"] += 1
    return text


# ----------------------------------------------------------------------------- tools
def parse_call_block(b, cfg):
    """one <tool_call> block -> (name, arguments) . JSON first, then tolerant fallbacks, then raw code."""
    t = re.sub(r"^(?:\\n|\s)+|(?:\\n|\s)+$", "", b)          # stray literal "\n" wrappers
    for fn in (lambda x: json.loads(x, strict=False), ast.literal_eval):
        try:
            j = fn(t)
        except Exception:
            continue
        if isinstance(j, dict) and isinstance(j.get("name"), str):
            return j["name"], j.get("arguments", j.get("parameters", {}))
        break
    if cfg.get("code_tool") and t and t[0] not in "{[":              # code-interpreter style: the block IS the code
        return cfg["code_tool"], {"code": t}
    raise Reject("tool_call_unparseable")


def convert_tools(msgs, rep, cfg):
    out, call_n, i = [], 0, 0
    has_func_prompt = any(m["role"] in ("system", "user") and isinstance(m["content"], str) and FUNC_PROMPT.search(m["content"][:3000]) for m in msgs[:2])
    while i < len(msgs):
        m = msgs[i]
        if m["role"] == "tool":
            raise Reject("tool_result_without_call")   # tool messages are consumed with their assistant call below
        if m["role"] != "assistant" or not isinstance(m["content"], str):
            out.append(m); i += 1; continue
        if m["content"].count("<tool_call>") != m["content"].count("</tool_call>"):
            raise Reject("tool_call_unbalanced")          # template artifact such as a bare "<tool_call>assistant"
        blocks = HERMES_CALL.findall(m["content"]) if "<tool_call>" in m["content"] else []
        if "tool_calls" in m and not blocks:      # already structured: pass through with its tool messages
            out.append(m); i += 1
            while i < len(msgs) and msgs[i]["role"] == "tool":
                out.append(msgs[i]); i += 1
            continue
        if not blocks:
            if has_func_prompt and DROPPED_MENTION.search(m["content"]):
                raise Reject("tool_call_dropped")
            out.append(m); i += 1; continue
        calls, ids = [], []
        for b in blocks:
            name, args = parse_call_block(b, cfg)
            if not isinstance(b, str) or not b.strip().startswith(("{", "\\n{")):
                rep["code_tool_call_converted"] += 1
            call_n += 1; cid = f"call_{call_n}"; ids.append(cid)
            calls.append({"id": cid, "type": "function", "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)}})
        text = HERMES_CALL.sub("", m["content"]).strip()
        rep["hermes_tool_calls_converted"] += 1
        out.append({"role": "assistant", "content": text, "tool_calls": calls})
        i += 1
        resp = []
        while i < len(msgs) and msgs[i]["role"] == "tool" and isinstance(msgs[i]["content"], str):
            found = HERMES_RESP.findall(msgs[i]["content"])
            resp += found if found else [msgs[i]["content"].strip()]
            i += 1
        if not resp and i >= len(msgs):
            continue                                 # trailing tool call: valid final assistant turn
        if len(resp) != len(ids):
            raise Reject("tool_response_missing" if not resp else "tool_mismatch")
        for cid, r in zip(ids, resp):
            out.append({"role": "tool", "tool_call_id": cid, "content": r})
    return out


def extract_tools(msgs):
    """tool schemas from a <tools>...</tools> block in the system prompt (skips the instruction sentence
    that merely mentions '<tools> </tools> XML tags')."""
    for m in msgs:
        if m["role"] == "system" and isinstance(m["content"], str):
            for b in TOOLS_BLOCK.finditer(m["content"]):
                txt = b.group(1).strip()
                if not txt:
                    continue
                try:
                    v = json.loads(txt)
                    v = v if isinstance(v, list) else [v]
                except Exception:
                    try:
                        v = [json.loads(l) for l in txt.splitlines() if l.strip()]
                    except Exception:
                        continue
                if v and all(isinstance(x, dict) for x in v):
                    return v
    return None


# ----------------------------------------------------------------------------- structure + text checks
def check_sequence(msgs):
    roles = [m["role"] for m in msgs]
    i = 1 if roles and roles[0] == "system" else 0
    if i >= len(roles) or roles[i] != "user":
        raise Reject("seq_invalid")
    expect, pending = "user", 0
    for m in msgs[i:]:
        r = m["role"]
        if r == "user" and expect == "user":
            expect = "assistant"
        elif r == "assistant" and expect == "assistant":
            pending = len(m.get("tool_calls") or [])
            expect = "tool" if pending else "user"
        elif r == "tool" and expect == "tool":
            pending -= 1
            if pending == 0:
                expect = "assistant"
        else:
            raise Reject("seq_invalid")
    last = msgs[-1]
    if last["role"] != "assistant":
        raise Reject("seq_invalid")
    if not last.get("tool_calls") and not (last["content"] or "").strip():
        raise Reject("empty_content")


def cwords(t):
    return {w for w in re.findall(r"[^\W\d_]{4,}", t.lower()) if w not in STOP}


def check_text(msgs, cfg):
    for m in msgs:
        c = m["content"] or ""
        try:
            c.encode("utf-8")                              # lone surrogates (e.g. "\ud83d" without its pair) cannot be written or tokenized
            if m.get("tool_calls"):
                json.dumps(m["tool_calls"], ensure_ascii=False).encode("utf-8")
        except UnicodeEncodeError:
            raise Reject("invalid_unicode")
        if not c.strip() and not m.get("tool_calls"):
            raise Reject("empty_content")
        if CTRL.search(c):
            raise Reject("control_or_replacement_char")
        if m["role"] == "assistant":
            plain = CODE_FENCE.sub(" ", c)
            if LETTER_RUN.search(plain):
                raise Reject("single_char_run")
            if DOT_RUN.search(plain):
                raise Reject("dot_run")
            w = plain.split()
            if len(w) >= 150:
                g = collections.Counter(" ".join(w[k:k + 10]) for k in range(0, len(w) - 9))
                gram, cnt = g.most_common(1)[0]
                if cnt >= 12 and sum(1 for x in gram.split() if re.search(r"[^\W\d_]", x)) >= 5:
                    raise Reject("repetition_loop")
            elif len(plain) >= 400 and plain.count(" ") < len(plain) / 15:      # scripts written without spaces (CJK, Thai, ...)
                g = collections.Counter(plain[k:k + 30] for k in range(0, len(plain) - 29, 3))
                win, cnt = g.most_common(1)[0]
                if cnt >= 8 and sum(ch.isalpha() for ch in win) >= 20:
                    raise Reject("repetition_loop")
    asst = [m["content"] for m in msgs if m["role"] == "assistant" and m["content"]]
    user = [m["content"] for m in msgs if m["role"] == "user"]
    if asst and asst[-1].strip() in ("None", "null", "N/A", "nan", "[]", "{}", "NULL"):
        raise Reject("placeholder_answer")
    if user and asst and user[0].strip() == asst[-1].strip():
        raise Reject("answer_equals_question")
    total = sum(len(m["content"] or "") for m in msgs)
    if cfg["max_chars"] and total > cfg["max_chars"]:
        raise Reject("too_long")
    if len(user) == 1 and len(user[0].strip()) < 160 and MISSING_PAYLOAD.search(user[0].strip()):
        raise Reject("prompt_missing_payload")          # "Write an educational piece related to the following text snippet:" + nothing
    if cfg["lexical"] and len(user) == 1 and len(asst) == 1 and not any(m["role"] == "tool" for m in msgs):
        u, a = user[0], THINK_BLOCK.sub(" ", asst[0])
        if "```" not in a and len(u) >= 60 and len(a) >= 150:
            if sum(ch.isalpha() for ch in u) / len(u) > 0.8 and sum(ch.isalpha() for ch in a) / len(a) > 0.8:
                wu, wa = cwords(u), cwords(a)
                if len(wu) >= 6 and len(wa) >= 20 and not (wu & wa):
                    raise Reject("qa_no_word_overlap")


# ----------------------------------------------------------------------------- one row
def process_row(line, cfg, state):
    """returns (row_dict_or_None, reason_or_None, repairs Counter)"""
    rep = collections.Counter()
    raw = line if isinstance(line, bytes) else line.encode("utf-8", "replace")
    key = int.from_bytes(hashlib.blake2b(raw.strip(), digest_size=8).digest(), "little")
    if key in state["seen"]:
        return None, "exact_duplicate", rep
    try:
        d = json.loads(line)
    except Exception:
        return None, "parse_error", rep
    try:
        msgs = d.get("messages") if isinstance(d, dict) else None
        if not isinstance(msgs, list) or not msgs:
            raise Reject("no_messages")
        msgs = normalize_messages(msgs, rep)
        if cfg.get("context", True):
            if any(m.get("role") == "user" and isinstance(m.get("content"), str) and PH_TAG.search(m["content"]) and missing_context_tag(m["content"]) for m in msgs):
                raise Reject("missing_context_placeholder")
            if missing_context_cutoff(msgs):
                raise Reject("missing_context_cutoff")
        n0 = len(msgs)
        msgs = [m for m in msgs if not (m["role"] == "system" and not (m["content"] or "").strip())]
        if len(msgs) != n0:
            rep["empty_system_dropped"] += 1
        for m in msgs:                                  # think / answer-tag / special-token / PII repair per message
            if isinstance(m["content"], str):
                m["content"] = fix_think(m["content"], m["role"], cfg, rep)
                if cfg["special"] == "strip":
                    m["content"] = fix_special(m["content"], rep)
                if cfg["pii"] == "redact":
                    m["content"] = redact_pii(m["content"], rep)
                elif cfg["pii"] == "datatrove":
                    m["content"] = pii_datatrove(m["content"], rep)
                if m["role"] == "assistant" and ANSWER_TAG.search(m["content"]):
                    m["content"] = ANSWER_TAG.sub("", m["content"]).strip(); rep["answer_tag_unwrapped"] += 1
        tools = extract_tools(msgs) or (d.get("tools") if isinstance(d.get("tools"), list) and d.get("tools") else None)
        msgs = convert_tools(msgs, rep, cfg)
        if cfg["pii"] == "datatrove":
            for m in msgs:
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    if isinstance(fn.get("arguments"), str):
                        fn["arguments"] = pii_datatrove(fn["arguments"], rep)
        for m in msgs:
            if m["role"] == "assistant" and m.get("content") is None:
                m["content"] = ""
            if m["role"] == "assistant" and not m.get("tool_calls") and THINK_BLOCK.search(m["content"] or "") and not THINK_BLOCK.sub("", m["content"]).strip():
                raise Reject("think_no_answer")
        if needs_python_tool(msgs, tools):
            tools = [PYTHON_TOOL]; rep["python_tool_schema_attached"] += 1
        check_sequence(msgs)
        check_text(msgs, cfg)
        if cfg["max_per_prompt"]:
            first = next((m["content"] for m in msgs if m["role"] == "user"), "")
            pk = hashlib.blake2b(re.sub(r"\s+", " ", first.lower()).encode(), digest_size=8).digest()
            state["prompts"][pk] += 1
            if state["prompts"][pk] > cfg["max_per_prompt"]:
                raise Reject("prompt_cap")
        state["seen"].add(key)
    except Reject as r:
        return None, r.reason, rep
    except Exception:
        return None, "internal_error", rep
    row = {"messages": [{k: m[k] for k in KEEP_MSG_KEYS if k in m} for m in msgs]}
    if tools:
        row["tools"] = tools
    return row, None, rep


# ----------------------------------------------------------------------------- file / driver
def new_state():
    return {"seen": set(), "prompts": collections.Counter()}


def out_name(rel, idx, nchunks):
    return rel if nchunks == 1 else rel[:-6] + f"__p{idx:04d}.jsonl"        # rel always ends with ".jsonl"


def run_chunk(args):
    """one byte-range of one input file -> one output file. Lines are owned by the chunk in which they START."""
    path, root, out_dir, cfg, start, end, idx, nchunks = args
    rel = os.path.relpath(path, root) if os.path.isdir(root) else os.path.basename(path)
    name = out_name(rel, idx, nchunks)
    st = collections.Counter(); reasons = collections.Counter(); repairs = collections.Counter(); ex = collections.defaultdict(list)
    state = new_state(); t0 = time.time(); sample = cfg["sample"]; fout = frej = fctx = None; tmp = outp = meta = cpath = None
    if out_dir and not sample:
        meta = os.path.join(out_dir, "_meta", name + ".json")
        if os.path.exists(meta):                                    # resume: this chunk is already finished
            d = json.load(open(meta)); return name, d["stats"], d["reasons"], d["repairs"], {}, True
        outp = os.path.join(out_dir, name); os.makedirs(os.path.dirname(outp), exist_ok=True); tmp = outp + ".tmp"
        fout = open(tmp, "w", encoding="utf-8", buffering=1 << 22)
        if cfg["save_rejects"]:
            rp = os.path.join(out_dir, "_rejected", name); os.makedirs(os.path.dirname(rp), exist_ok=True); frej = open(rp, "w", encoding="utf-8")

    def lines():
        if sample:
            rnd = random.Random(hash(rel) % 100003); size = os.path.getsize(path); used = set()
            with open(path, "rb") as f:
                for _ in range(sample * 4):
                    if len(used) >= sample:
                        break
                    f.seek(rnd.randint(0, max(0, size - 1))); f.readline(); off = f.tell(); l = f.readline()
                    if l and off not in used:
                        used.add(off); yield l
        else:
            with open(path, "rb", buffering=1 << 22) as f:
                if start > 0:
                    f.seek(start - 1); f.readline()                   # finish the line that belongs to the previous chunk
                while f.tell() < end:
                    l = f.readline()
                    if not l:
                        break
                    yield l

    for l in lines():
        st["rows_in"] += 1
        row, reason, rep = process_row(l, cfg, state)
        repairs.update(dict.fromkeys(rep, 1))
        if row is not None:
            st["rows_kept"] += 1
            if fout:
                try:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                except UnicodeEncodeError:                     # last line of defence: never let one row kill a chunk
                    st["rows_kept"] -= 1; reasons["invalid_unicode"] += 1; continue
            if rep and len(ex["repaired"]) < 2:
                ex["repaired"].append({"before": l.decode("utf-8", "replace")[:700], "after": json.dumps(row, ensure_ascii=False)[:700]})
        else:
            reasons[reason] += 1
            if len(ex[reason]) < 3:
                ex[reason].append(l.decode("utf-8", "replace")[:600])
            if frej:
                frej.write(json.dumps({"reason": reason, "row": l.decode("utf-8", "replace").strip()}, ensure_ascii=False) + "\n")
            if out_dir and not sample and reason and reason.startswith("missing_context"):
                if fctx is None:
                    cpath = os.path.join(out_dir, "_rejects_context", name); os.makedirs(os.path.dirname(cpath), exist_ok=True); fctx = open(cpath + ".tmp", "w", encoding="utf-8")
                fctx.write(l.decode("utf-8", "replace").rstrip("\n") + "\n")
    st["seconds"] = round(time.time() - t0, 1)
    if fout:
        fout.close(); os.replace(tmp, outp)
        os.makedirs(os.path.dirname(meta), exist_ok=True)
        json.dump({"stats": dict(st), "reasons": dict(reasons), "repairs": dict(repairs)}, open(meta + ".tmp", "w")); os.replace(meta + ".tmp", meta)
    if frej:
        frej.close()
    if fctx:
        fctx.close(); os.replace(cpath + ".tmp", cpath)
    return name, dict(st), dict(reasons), dict(repairs), dict(ex), False


def gather(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True))
        else:
            files.append(p)
    return files


def print_report(tot, reasons, repairs):
    n = max(1, tot["rows_in"])
    print(f"\nrows in {tot['rows_in']:,}  kept {tot['rows_kept']:,} ({100*tot['rows_kept']/n:.2f}%)")
    print("REJECTED by reason:")
    for k, v in reasons.most_common():
        print(f"  {k:32s}{v:>14,}  {100*v/n:6.2f}%")
    print("REPAIRED (rows kept, fixed):")
    for k, v in repairs.most_common():
        print(f"  {k:32s}{v:>14,}  {100*v/n:6.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", action="append", default=[])
    ap.add_argument("--out"); ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--think", choices=["keep", "strip"], default="keep")
    ap.add_argument("--empty-think", dest="empty_think", choices=["drop", "strip"], default="drop")
    ap.add_argument("--max-chars", type=int, default=200000)
    ap.add_argument("--max-per-prompt", type=int, default=0)
    ap.add_argument("--lexical-mismatch", action="store_true", help="OFF by default: rejects single-turn rows whose answer shares no content word with the question. Unsafe for translation tasks and for scripts written without spaces.")
    ap.add_argument("--special-tokens", dest="special", choices=["strip", "keep"], default="strip", help="strip: remove trailing <|im_end|>/<|eot_id|>/</s> and reject rows with other chat-template control strings inside the text")
    ap.add_argument("--pii", choices=["off", "redact", "datatrove"], default="off", help="redact: replace emails, phone numbers, secrets/keys, card numbers and context-tagged SSNs with placeholders. datatrove: FineWeb-style, only emails + public IPs, replaced by reserved fake values (see pii_datatrove). Names/addresses are NOT handled by either.")
    ap.add_argument("--context-check", dest="context_check", choices=["on", "off"], default="on", help="on: drop rows whose question is missing content (placeholder tags, cut-off questions)")
    ap.add_argument("--code-tool", dest="code_tool", default="python", help="function name used when a <tool_call> block is raw code; '' = reject such rows")
    ap.add_argument("--chunk-mb", type=int, default=256, help="big input files are split into byte ranges of this size so many workers share them")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--save-rejects", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    cfg = dict(think=a.think, empty_think=a.empty_think, max_chars=a.max_chars, max_per_prompt=a.max_per_prompt,
               lexical=a.lexical_mismatch, sample=a.sample, save_rejects=a.save_rejects, code_tool=a.code_tool, special=a.special, pii=a.pii, context=(a.context_check == "on"))
    files = gather(a.inp)
    if not files:
        sys.exit("no input files")
    root = a.inp[0] if len(a.inp) == 1 else os.path.commonpath([os.path.dirname(f) for f in files])
    if not a.sample and not a.out:
        sys.exit("--out is required unless --sample is used")
    cb = a.chunk_mb * (1 << 20); tasks = []
    for f in files:
        size = os.path.getsize(f); n = 1 if a.sample else max(1, -(-size // cb))
        tasks += [(f, root, a.out, cfg, i * cb, (i + 1) * cb if i < n - 1 else 1 << 62, i, n) for i in range(n)]
    print(f"{len(files)} input files -> {len(tasks)} chunks, {min(a.workers, len(tasks))} workers", flush=True)
    tot = collections.Counter(); reasons = collections.Counter(); repairs = collections.Counter(); ex = collections.defaultdict(list)
    t0 = time.time(); done = skipped = 0
    with Pool(min(a.workers, len(tasks))) as pool:
        for name, st, rs, rp, e, resumed in pool.imap_unordered(run_chunk, tasks, chunksize=1):
            done += 1; skipped += resumed
            if not resumed:
                tot.update({k: v for k, v in st.items() if k != "seconds"}); reasons.update(rs); repairs.update(rp)
            for k, v in e.items():
                ex[k] += v[: max(0, 3 - len(ex[k]))]
            if done % 25 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"[{done}/{len(tasks)} chunks, {skipped} resumed] {time.strftime('%H:%M:%S')}  rows_in {tot['rows_in']:,}  kept {tot['rows_kept']:,}  elapsed {el/60:.1f} min", flush=True)
    if a.out and not a.sample:                                       # totals over EVERY finished chunk, including ones from earlier runs
        tot = collections.Counter(); reasons = collections.Counter(); repairs = collections.Counter()
        for mp in glob.glob(os.path.join(a.out, "_meta", "**", "*.json"), recursive=True):
            d = json.load(open(mp)); tot.update({k: v for k, v in d["stats"].items() if k != "seconds"}); reasons.update(d["reasons"]); repairs.update(d["repairs"])
    print_report(tot, reasons, repairs)
    rep_dir = a.out or "."
    os.makedirs(rep_dir, exist_ok=True)
    rep_path = os.path.join(rep_dir, "_sample_report.json" if a.sample else "_filter_report.json")
    json.dump({"totals": dict(tot), "rejected": dict(reasons), "repaired": dict(repairs), "examples": dict(ex), "settings": {k: v for k, v in cfg.items()}}, open(rep_path, "w"), indent=1, ensure_ascii=False)
    print("report ->", rep_path)


# ----------------------------------------------------------------------------- self test
def selftest():
    base = dict(think="keep", empty_think="drop", max_chars=200000, max_per_prompt=0, lexical=True, code_tool="python", special="strip", pii="off")

    def run(msgs, **kw):
        return process_row(json.dumps({"messages": msgs}), dict(base, **kw), new_state())
    U = lambda t: {"role": "user", "content": t}
    A = lambda t: {"role": "assistant", "content": t}
    S = lambda t: {"role": "system", "content": t}
    ok = lambda r: r[0] is not None
    why = lambda r: r[1]
    good = [U("What is the capital of France and why is it famous?"), A("The capital of France is Paris, famous for the Eiffel Tower, the Louvre and its cafe culture.")]
    r = run(good); assert ok(r) and r[0]["messages"] == good, "valid row must pass unchanged"
    dots = "<think> " + ". " * 300 + "</think> \n\n <answer>Equity options expire on the third Friday.</answer>"
    assert why(run([U("How do small cap stocks perform vs. large cap stocks during bear trends?"), A(dots)])) == "think_junk", "user's example must be rejected"
    assert why(run([U("hello there my friend how are you"), A("<think>\n</think>\nHello! I am fine, thank you for asking.")])) == "think_empty"
    r = run([U("hello there my friend how are you"), A("<think>\n</think>\nHello! I am fine, thank you for asking.")], empty_think="strip")
    assert ok(r) and r[0]["messages"][1]["content"] == "Hello! I am fine, thank you for asking."
    r = run([U("Solve 2+2 please and explain."), A("<think><think>Add two and two.</think></think>\nThe answer is 4 because 2 plus 2 equals 4.")])
    assert ok(r) and r[0]["messages"][1]["content"].count("<think>") == 1
    r = run([U("Solve 2+2 please and explain."), A("<think>Add two and two together.</think>\n<answer>The answer is 4.</answer>")])
    assert ok(r) and "<answer>" not in r[0]["messages"][1]["content"]
    r = run([U("Which language is this sentence?"), A("<think><think> It is Arabic script. </think> ar.</think> ar.")]); assert ok(r) and r[0]["messages"][1]["content"] == "<think> It is Arabic script. </think> ar."
    assert why(run([U("Solve 2+2 please and explain."), A("<think>Add two and two together but never finish")])) == "think_unbalanced"
    assert why(run([U("Solve 2+2 please and explain."), A("<think>Add two and two together.</think>")])) == "think_no_answer"
    assert why(run([U("hi there you"), A("None")])) == "placeholder_answer"
    assert why(run([U("hi there you"), A("梦" * 300)])) == "single_char_run"
    assert why(run([U("hi there you"), A("ok " + ". " * 60 + "done")])) == "dot_run"
    assert why(run([U("hi there you"), A("bad � char here")])) == "control_or_replacement_char"
    assert why(process_row('{"messages": [{"role": "user", "content": "hello there my friend"}, {"role": "assistant", "content": "broken emoji \\ud83d here"}]}', dict(base), new_state())) == "invalid_unicode"
    assert why(run([U("hi there you"), A("fine answer here"), U("another question")])) == "seq_invalid"
    assert why(run([U("hi there you"), U("second user"), A("answer text")])) == "seq_invalid"
    assert why(run([U("hi"), A("")])) == "empty_content"
    assert ok(run([U("add these numbers please"), A("The sum is " + "+ 7 + 0 + 5 + 8 + 8 + 4 + 1 " * 30 + "= 300 in total for the list.")])), "arithmetic expansion is not a loop"
    assert ok(run([U("show me the contents"), A("Table of Contents\n1. Get to Know a Spider" + "." * 60 + "1\n2. Can Spiders Hurt Me?" + "." * 60 + "9")])), "TOC dot leaders are legitimate"
    assert why(run([U("hello there my friend how are you"), A("<think>None</think>\nHello! I am fine, thank you for asking.")])) == "think_empty"
    r = run([{"role": "system", "content": "[{'type': 'text', 'content': ''}]"}, U("hello there my friend"), A("Hello! I am fine, thank you.")]); assert ok(r) and r[0]["messages"][0]["role"] == "user"
    assert why(run([U("say it"), A("This is false, so we need to increase n and check again. " * 60)])) == "repetition_loop"
    assert why(run([U("Write an educational piece related to the following text snippet:"), A("Let us begin by exploring the concept of angular momentum in physics.")])) == "prompt_missing_payload"
    assert ok(run([U("Write an educational piece related to the following text snippet:\n\nAngular momentum is conserved."), A("Let us begin by exploring the concept of angular momentum in physics.")]))
    r = run([U("[{'type': 'text', 'content': 'What is two plus two in maths?'}]"), A("[{'type': 'text', 'content': 'It is four.'}]")])
    assert ok(r) and r[0]["messages"][0]["content"] == "What is two plus two in maths?"
    assert why(run([U("[{'text': 'click here', 'type': 'text'}, {'text': None, 'type': 'image'}]"), A("clicked the button")])) == "multimodal_image_part"
    nested = "[{'content': 'Namaste, tell me about Tunisia please', 'role': 'user'}, {'content': 'Tunisia is a country in North Africa.', 'role': 'assistant'}]"
    r = run([U("Namaste, tell me about Tunisia please"), A(nested)])
    assert ok(r) and len(r[0]["messages"]) == 2 and r[0]["messages"][1]["content"] == "Tunisia is a country in North Africa."
    sysm = S("You are a function calling AI model. <tools>[{\"name\": \"get_weather\", \"parameters\": {}}]</tools>")
    call = A("<think>Need the weather.</think>\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>")
    resp = {"role": "tool", "content": "<tool_response>\n{\"temp\": 20}\n</tool_response>"}
    r = run([sysm, U("What is the weather in Paris today?"), call, resp, A("It is 20 degrees in Paris.")])
    assert ok(r), why(r)
    m = r[0]["messages"]
    assert m[2]["tool_calls"][0]["function"]["name"] == "get_weather" and m[3]["tool_call_id"] == m[2]["tool_calls"][0]["id"] and m[3]["content"] == '{"temp": 20}' and r[0]["tools"][0]["name"] == "get_weather"
    assert why(run([U("What is the weather in Paris today?"), A("I checked."), resp, A("It is 20 degrees.")])) in ("tool_result_without_call", "seq_invalid")
    assert why(run([S("You are a function calling AI model with <functions></functions> XML tags."), U("What is the weather in Paris today?"), A("To get the weather I need to call the 'get_weather' function. By calling this function I will know.")])) == "tool_call_dropped"
    assert why(run([U("How do small cap stocks behave when markets fall hard for many months?"), A("Equity options actually expire on the Saturday after the third Friday of every month, according to exchange rules published online by exchanges. Their published documentation explains settlement procedures clearly.")])) == "qa_no_word_overlap"
    codecall = A("<tool_call>\nx = 2 + 2\nprint(x)\n</tool_call>")
    r = run([U("Compute two plus two with code please."), codecall, {"role": "tool", "content": "<tool_response>\n4\n</tool_response>"}, A("The answer is 4.")])
    assert ok(r), why(r)
    assert r[0]["messages"][1]["tool_calls"][0]["function"]["name"] == "python" and json.loads(r[0]["messages"][1]["tool_calls"][0]["function"]["arguments"])["code"].startswith("x = 2")
    assert why(run([U("Compute two plus two with code please."), codecall, {"role": "tool", "content": "4"}, A("It is 4.")], code_tool="")) == "tool_call_unparseable"
    r = run([U("Compute two plus two with code please."), A("Let me call it.\n<tool_call>\n{\"name\": \"calc\", \"arguments\": {\"q\": \"2+2\"}}\n</tool_call>")]); assert ok(r) and r[0]["messages"][-1]["tool_calls"]
    sys2 = S("You may call functions inside <tools> </tools> XML tags.\n<tools>\n[{\"type\": \"function\", \"function\": {\"name\": \"get_weather\"}}]\n</tools>")
    r = run([sys2, U("What is the weather in Paris today?"), call, resp, A("It is 20 degrees.")]); assert ok(r) and r[0]["tools"][0]["function"]["name"] == "get_weather"
    assert why(run([U("Compute two plus two with code please."), A("Sure.\n<tool_call>assistant\nThe answer is four and here is why it works well.")])) == "tool_call_unbalanced"
    r = run([U("Say hello to me please."), A("Hello there, my friend!<|eot_id|>")]); assert ok(r) and r[0]["messages"][1]["content"] == "Hello there, my friend!"
    r = run([U("Say hello to me please."), A("Hello there.</s>")]); assert ok(r) and r[0]["messages"][1]["content"] == "Hello there."
    assert why(run([U("Say hello to me please."), A("Hello <|im_start|>assistant hi again there")])) == "special_token_in_text"
    assert why(run([U("<|im_start|>user Say hello to me please."), A("Hello there friend.")])) == "special_token_in_text"
    assert ok(run([U("How do I render struck text?"), A("Use the <s>strike</s> tag in HTML, like <s>old price</s> then the new price.")]))
    assert ok(run([U("Say hello to me please."), A("Hello<|eot_id|>")], special="keep"))
    r = run([U("Please draft a short note to Jane."), A("Write to jane.roe@gmail.com or call +1 415-555-2671. Key AKIAIOSFODNN7EXAMPLE is public docs; real AKIAABCDEFGHIJKLMNOP is not. Mail user@example.com is fake.")], pii="redact")
    assert ok(r); c = r[0]["messages"][1]["content"]; assert "<EMAIL>" in c and "<PHONE>" in c and "<SECRET>" in c and "AKIAIOSFODNN7EXAMPLE" in c and "user@example.com" in c and "jane.roe" not in c and "415-555" not in c, c
    r = run([U("What is the SSN on file?"), A("The social security number: 219-09-9999 belongs to the applicant.")], pii="redact"); assert ok(r) and "<SSN>" in r[0]["messages"][1]["content"]
    r = run([U("Which card did the customer use to pay?"), A("The customer paid with card 4532 0151 1283 0366 yesterday afternoon at the store.")], pii="redact"); assert ok(r) and "<CARD_NUMBER>" in r[0]["messages"][1]["content"]
    r = run([U("Which card was charged for the order?"), A("Card number 4532015112830366 was charged for that order yesterday afternoon.")], pii="redact"); assert ok(r) and "<CARD_NUMBER>" in r[0]["messages"][1]["content"]
    numlist = "values 319 4028 19922 12132 4532 0151 1283 0366 2192 17 13 in order"
    r = run([U("List the values from the run please."), A(numlist)], pii="redact"); assert ok(r) and r[0]["messages"][1]["content"] == numlist, "number lists are not cards"
    keep = "The score was 0.4532015112830366 and the sum is +463477172 +740174259 with the post id 455629028514400 ok and \\n@app.route handler."
    r = run([U("What did the run print out at the end?"), A(keep)], pii="redact"); assert ok(r) and r[0]["messages"][1]["content"] == keep, "false-positive guards"
    r = run([U("Which card is the well known Visa test card?"), A("The standard sandbox card number is 4111 1111 1111 1111 in all payment docs.")], pii="redact"); assert ok(r) and "4111 1111 1111 1111" in r[0]["messages"][1]["content"], "famous test numbers stay"
    assert "jane.roe" in run([U("Please draft a short note to Jane."), A("Write to jane.roe@gmail.com soon.")])[0]["messages"][1]["content"], "pii is off by default"
    st = new_state()
    assert process_row(json.dumps({"messages": good}), dict(base), st)[0] and process_row(json.dumps({"messages": good}), dict(base), st)[1] == "exact_duplicate"
    c2 = dict(base, max_per_prompt=1); st = new_state()
    g2 = [good[0], A("A different but fine answer about Paris, the capital of France, a city known for art.")]
    assert process_row(json.dumps({"messages": good}), c2, st)[0] and process_row(json.dumps({"messages": g2}), c2, st)[1] == "prompt_cap"
    # ---- missing-context rules
    dna_q = "Does the sequence <DNA> exhibit methylation at its middle CpG site in human embryonic stem cells?"
    assert run([U(dna_q), A("No, this CpG site is not methylated in these cells.")])[1] == "missing_context_placeholder"
    assert run([U("Describe what the picture below shows in one sentence.\n<image>"), A("A small brown dog is running across a green field.")])[1] == "missing_context_placeholder"
    assert run([U(dna_q), A("No, this CpG site is not methylated in these cells.")], context=False)[0] is not None, "context check can be switched off"
    prot = "Presented here is a sequence of amino acids in a protein: <protein> M P K G S G K V I A Q N K K A F H D Y F I D E T Y E A G L V L Q G. Describe its function."
    assert run([U(prot), A("This protein is likely an ATPase involved in ribosome rescue in bacteria.")])[0] is not None, "sequence right after the tag = delimiter, keep"
    seqp = "Presented here is a sequence of amino acids in a protein: <seq> M P K G S G K V I A Q N K K A F H D Y F I D E T Y E A G L V L Q G. Describe its function."
    assert run([U(seqp), A("This protein is likely an ATPase involved in ribosome rescue in bacteria.")])[0] is not None, "<seq> is never a placeholder"
    assert run([U("Here is a DNA fragment <DNA> ATGCGTACGTTAGCTAGCTAGGCTAAGCT. Is it a coding region?"), A("It could be, since it starts with ATG and has no early stop codon.")])[0] is not None
    assert run([U("Parse this HTML table: <table><tr><td>1</td></tr></table> and list the cells."), A("The table has one row with one cell containing the value 1.")])[0] is not None
    assert run([U("Why does the <img> tag need an alt attribute for screen readers?"), A("The alt text describes the image for people who cannot see it.")])[0] is not None
    cut_u = "What is the prevalence of beta-lactamase CTX-M-15 in"
    assert run([U(cut_u), A("<think>The user asks: \"What is the prevalence of beta-lactamase CTX-M-15 in\". The question seems incomplete. Perhaps they mean a region.</think>Your question looks cut off; which population do you mean?")])[1] == "missing_context_cutoff"
    assert run([U(cut_u), A("Beta-lactamase CTX-M-15 is widespread among E. coli isolates, with rates that differ by region and setting.")])[0] is not None, "cut-off user but the assistant did not flag it = keep (one signal only)"
    assert run([U("Why does this question seem incomplete to some readers?"), A("Some readers see it as incomplete because the sentence lacks context and the rest of the question.")])[0] is not None, "complete question = keep"
    # ---- DataTrove-style PII
    pd = dict(pii="datatrove")
    c = run([U("Who should I contact about the newsletter?"), A("Email the editor at jane.roe@gmail.com or the desk at info@bbc.co.uk, or use test@example.com.")], **pd)[0]["messages"][1]["content"]
    assert "jane.roe" not in c and "bbc.co.uk" not in c and "test@example.com" in c and ("email@example.com" in c or "firstname.lastname@example.org" in c), c
    c = run([U("Which server did the log show as the source?"), A("The log shows the client IP address 118.25.6.39 and the gateway 192.168.1.1 and the DNS server 8.8.8.8 in use.")], **pd)[0]["messages"][1]["content"]
    assert "118.25.6.39" not in c and "192.168.1.1" in c and "8.8.8.8" in c, c
    keep = "Subcase 2.2.2.2 gives 44 ways, the tiling 3.3.3.3.6 is semiregular, version 1.2.3.4 is invalid, pin numpy==1.2.3.4 and see 1.2.3.4.5 in the log, router-id 1.1.1.1, and the decorator\\n@pytest.fixture is used."
    assert run([U("What did the notes say about the run today?"), A(keep)], **pd)[0]["messages"][1]["content"] == keep, "no damage to math, versions, tilings, decorators"
    keep2 = "The address block assigned to the Asia-Pacific region is 1.128.0.0/10 and the network address 20.30.40.0 is not a host."
    assert run([U("What block is assigned to that region?"), A(keep2)], **pd)[0]["messages"][1]["content"] == keep2, "CIDR blocks / network addresses are not hosts"
    assert run([U("Which server did the log show as the source?"), A("The client IP address was 118.25.6.39 in the log.")])[0]["messages"][1]["content"].count("118.25.6.39") == 1, "pii off by default"
    # ---- python tool schema
    pc = [U("Compute 17 times 23 with python please."), {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": "print(17*23)"})}}]}, {"role": "tool", "tool_call_id": "call_1", "content": "391"}, A("17 times 23 equals 391.")]
    r = run(pc); assert r[0] is not None and r[0]["tools"] == [PYTHON_TOOL], r
    pc2 = json.loads(json.dumps(pc)); pc2[1]["tool_calls"][0]["function"]["name"] = "calculator"
    r = run(pc2); assert r[0] is not None and "tools" not in r[0], "other tools never get a python schema"
    assert needs_python_tool(pc, [{"type": "function", "function": {"name": "x"}}]) is False, "existing schema is kept"
    print("SELFTEST OK")


if __name__ == "__main__":
    main()
