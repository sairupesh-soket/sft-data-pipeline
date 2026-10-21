"""Stage 1 -- convert Agentic_Ai/OUTPUT parquet into OpenAI-format SFT JSONL.

Unlike the v1 traces scrape, this source mixes agent trajectories, ordinary
chat data, Q/A pairs, benchmarks, images and non-chat catalogs, so there is
no generic "has a messages column" pass: every dataset is listed below with
an explicit converter (or an explicit skip reason), each written only after
reading real rows of that dataset.

One job per SOURCE parquet file (multi-file datasets such as cosmopedia or
Toucan produce one output file per source file, e.g.
HuggingFaceTB__cosmopedia_7.jsonl). Output rows:
    {"messages": [...], "tools": [...]}   ("tools" only when the source has
                                           tool definitions)

Usage:
    python3 convert.py                 # every dataset
    python3 convert.py NAME [NAME...]  # only these dataset prefixes
    python3 convert.py --limit 200 NAME   # dry-ish test: first 200 rows,
                                          # written to /tmp-style scratch
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow.parquet as pq

from ..openai_format import (ROLE_MAP, convert_message, finalize, loads_loose, parse_chatml, simple_pair, to_plain)
SRC = os.environ.get("SFTPIPE_SRC", ".")          # Agentic_Ai/OUTPUT style folder of parquet files
OUT_DIR = os.environ.get("SFTPIPE_OUT", "out")
REPORT_DIR = os.environ.get("SFTPIPE_LOG", OUT_DIR.rstrip("/") + "_logs")

# --------------------------------------------------------------------------
# generic converters
# --------------------------------------------------------------------------


def std(col, tools_col=None, keep=None):
    """A column that already holds a list of chat messages (or a JSON /
    ChatML string of one). `keep(row) -> reason|None` is an optional filter."""

    def fn(row):
        if keep:
            why = keep(row)
            if why:
                return None, why
        v = row.get(col)
        from_struct = not isinstance(v, str)
        if isinstance(v, str):
            parsed = loads_loose(v)
            v = parsed if isinstance(parsed, list) else (parse_chatml(v) if "<|im_start|>" in v else None)
        if not v:
            return None, "skip_empty_or_invalid"
        msgs = [convert_message(m, from_struct) for m in to_plain(v)]
        tools = row.get(tools_col) if tools_col else None
        return finalize(msgs, tools, tools_from_struct=not isinstance(tools, str))

    fn.columns = [col] + ([tools_col] if tools_col else [])
    return fn


def pair(u, a, s=None, extra_cols=()):
    def fn(row):
        return simple_pair(row.get(u), row.get(a), row.get(s) if s else None)
    fn.columns = [c for c in (u, a, s) if c] + list(extra_cols)
    return fn


# --------------------------------------------------------------------------
# bespoke converters (one per non-standard dataset; each was written after
# reading real rows -- see CLAUDE.md for what each one found)
# --------------------------------------------------------------------------

def swe_gym_keep(row):
    # real correctness signal: only trajectories whose patch resolved the issue
    return None if row.get("resolved") is True else "skip_not_resolved"


swe_gym = std("messages", "tools", keep=swe_gym_keep)
swe_gym.columns = ["messages", "tools", "resolved"]


def tau_keep(row):
    # 4,270 of 6,014 rows are per-round PREFIXES of a conversation (round /
    # total_rounds set, reward null) -- duplicates of the full one. Keep only
    # finished conversations the environment scored as a success.
    if not row.get("messages"):
        return "skip_empty_or_invalid"
    if row.get("reward") is None:
        return "skip_partial_round_prefix"
    return None if row["reward"] == 1.0 else "skip_reward_not_1"


tau_bench = std("messages", "tools", keep=tau_keep)
tau_bench.columns = ["messages", "tools", "reward"]

QWEN_TOOL_BOILERPLATE = [
    "# Tools",
    "You may call one or more functions to assist with the user query.",
    "You are provided with function signatures within <tools></tools> XML tags:",
    "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:",
    "<tool_call>",
    '{"name": <function-name>, "arguments": <args-json-object>}',
    "</tool_call>",
]


def toucan(row):
    q = row.get("response_quality_assessment")
    score = None
    if q:
        d = loads_loose(q)
        if isinstance(d, dict):
            score = d.get("overall_score")
    if score is None:
        return None, "skip_no_quality_score"
    if score < 4.0:
        return None, "skip_quality_below_4"
    raw = loads_loose(row["messages"] or "")
    if not isinstance(raw, list):
        return None, "skip_empty_or_invalid"
    msgs = []
    for m in raw:
        role = m.get("role")
        if role == "system" and "<tools>" in (m.get("content") or ""):
            # Qwen's own tool-prompt template: the same tool list is carried in
            # the "tools" field below, and the GLM template renders tools in
            # its own format -- keeping this would show the tools twice, in
            # two conflicting call syntaxes. Keep only any non-boilerplate text.
            # boilerplate first: one of its lines itself contains the literal
            # "<tools></tools>", which the block regex would otherwise eat
            rest = m["content"]
            for b in QWEN_TOOL_BOILERPLATE:
                rest = rest.replace(b, "")
            rest = re.sub(r"<tools>.*?</tools>", "", rest, flags=re.S)
            if rest.strip():
                msgs.append({"role": "system", "content": rest.strip()})
            continue
        if role == "tool_call":
            d = loads_loose(m.get("content") or "")
            if not isinstance(d, dict) or not d.get("name"):
                return None, "skip_bad_tool_call"
            msgs.append(convert_message({"role": "assistant", "content": "",
                                         "tool_calls": [{"name": d["name"], "arguments": d.get("arguments")}]}))
            continue
        msgs.append(convert_message(m))
    return finalize(msgs, row.get("available_tools"))


toucan.columns = ["messages", "available_tools", "response_quality_assessment"]


def ultrafeedback_chosen(row):
    return std("chosen")(row)


ultrafeedback_chosen.columns = ["chosen"]


def camel_code(row):
    u = row.get("instruction") or ""
    inp = (row.get("input") or "").strip()
    if inp and inp.lower() not in ("none", "n/a"):
        u = f"{u}\n\nInput: {inp}"
    return simple_pair(u, row.get("output"))


camel_code.columns = ["instruction", "input", "output"]


def aqua(row):
    opts = "\n".join(row.get("options") or [])
    u = f"{row['question']}\n\nOptions:\n{opts}"
    a = (row.get("rationale") or "").strip()
    c = (row.get("correct") or "").strip()
    if c and c not in a[-40:]:
        a = f"{a}\n\nAnswer: {c}"
    return simple_pair(u, a)


aqua.columns = ["question", "options", "rationale", "correct"]

_MATH500 = None


def math500():
    global _MATH500
    if _MATH500 is None:
        t = pq.read_table(f"{SRC}/HuggingFaceH4__MATH-500.parquet", columns=["problem"])
        _MATH500 = {p.strip() for p in t["problem"].to_pylist()}
    return _MATH500


def hendrycks(row):
    # MATH-500 is a held-out benchmark (skipped per user decision); its
    # problems also sit inside hendrycks_math's test split -- drop them here
    # too, or the benchmark leaks back in through this dataset.
    if (row.get("problem") or "").strip() in math500():
        return None, "skip_in_math500_benchmark"
    return simple_pair(row.get("problem"), row.get("solution"))


hendrycks.columns = ["problem", "solution"]

CC_LANG = {3: "python", 2: "cpp", 4: "java"}  # 1 = Python 2, skipped


def code_contests(row):
    sol = row.get("solutions") or {}
    langs, codes = sol.get("language") or [], sol.get("solution") or []
    for want in (3, 2, 4):
        for lang, code in zip(langs, codes):
            if lang == want and code and code.strip():
                a = f"```{CC_LANG[want]}\n{code.strip()}\n```"
                return simple_pair(row.get("description"), a)
    return None, "skip_no_py3_cpp_java_solution"


code_contests.columns = ["description", "solutions"]


def duorc(row):
    ans = [a for a in (row.get("answers") or []) if a and a.strip()]
    if row.get("no_answer") or not ans:
        return None, "skip_no_answer"
    u = (f"Read the following movie plot and answer the question.\n\n"
         f"Plot:\n{row['plot'].strip()}\n\nQuestion: {row['question'].strip()}")
    return simple_pair(u, ans[0])


duorc.columns = ["plot", "question", "answers", "no_answer"]

FINGPT_RE = re.compile(r"\[INST\]\s*<<SYS>>\s*(.*?)\s*<</SYS>>\s*(.*?)\s*\[/INST\]\s*$", re.S)


def fingpt(row):
    m = FINGPT_RE.match(row.get("prompt") or "")
    if not m:
        return None, "skip_unparsed_prompt"
    return simple_pair(m.group(2), row.get("answer"), m.group(1))


fingpt.columns = ["prompt", "answer"]

GLAIVE_TURN = re.compile(r"(?m)^(USER|ASSISTANT|FUNCTION RESPONSE):\s?")
GLAIVE_CALL = re.compile(r'<functioncall>\s*(\{.*\})\s*$', re.S)


def glaive(row):
    sysmsg = (row.get("system") or "").strip()
    if sysmsg.startswith("SYSTEM:"):
        sysmsg = sysmsg[len("SYSTEM:"):].strip()
    tools = []
    if "{" in sysmsg:
        head, _, body = sysmsg.partition("{")
        body = "{" + body
        dec = json.JSONDecoder()
        i = 0
        while i < len(body):
            while i < len(body) and body[i].isspace():
                i += 1
            if i >= len(body):
                break
            try:
                obj, j = dec.raw_decode(body, i)
            except Exception:
                return None, "skip_unparsed_tool_defs"
            tools.append(obj)
            i = j
        sysmsg = None  # only the tool listing -- carried in "tools" instead
    parts = GLAIVE_TURN.split(row.get("chat") or "")
    msgs = [{"role": "system", "content": sysmsg}] if sysmsg else []
    for tag, text in zip(parts[1::2], parts[2::2]):
        text = text.replace("<|endoftext|>", "").strip()
        if tag == "USER":
            msgs.append({"role": "user", "content": text})
        elif tag == "FUNCTION RESPONSE":
            msgs.append({"role": "tool", "content": text})
        else:
            m = GLAIVE_CALL.search(text)
            if m:
                raw = m.group(1)
                # arguments are a single-quoted JSON string: '{"a": 1}'
                am = re.search(r'"arguments"\s*:\s*\'(.*)\'\s*\}\s*$', raw, re.S)
                nm = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
                if not nm:
                    return None, "skip_bad_function_call"
                args = am.group(1) if am else (loads_loose(raw) or {}).get("arguments", {})
                pre = text[:m.start()].strip()
                msgs.append(convert_message({"role": "assistant", "content": pre,
                                             "tool_calls": [{"name": nm.group(1), "arguments": args}]}))
            else:
                msgs.append({"role": "assistant", "content": text})
    return finalize(msgs, tools or None)


glaive.columns = ["system", "chat"]


def afm(row):
    inp = (row.get("input") or "").strip()
    if inp:
        return simple_pair(inp, row.get("output"), row.get("instruction"))
    return simple_pair(row.get("instruction"), row.get("output"))


afm.columns = ["instruction", "input", "output"]


def tau2_sft(row):
    msgs = [convert_message(m) for m in to_plain(row.get("prompt") or [])]
    if row.get("response"):
        msgs.append({"role": "assistant", "content": row["response"]})
    return finalize(msgs)


tau2_sft.columns = ["prompt", "response"]


def cosmopedia(row):
    return simple_pair(row.get("prompt"), (row.get("text") or "").strip())


cosmopedia.columns = ["prompt", "text"]


# ---- Anthropic content-block conversations (claude-code traces) ----------

def anthropic_msgs(raw):
    out = []
    for m in raw:
        role, c = m.get("role"), m.get("content")
        if isinstance(c, str):
            out.append({"role": role, "content": c})
            continue
        if not isinstance(c, list):
            continue
        if role == "assistant":
            text = "".join(b.get("text", "") for b in c if b.get("type") == "text")
            think = "".join(b.get("thinking", "") for b in c if b.get("type") == "thinking")
            calls = [{"id": b.get("id"), "name": b.get("name"), "arguments": b.get("input")}
                     for b in c if b.get("type") == "tool_use"]
            out.append(convert_message({"role": "assistant", "content": text,
                                        "tool_calls": calls or None,
                                        "reasoning_content": think or None}))
        else:
            for b in c:
                if b.get("type") == "tool_result":
                    rc = b.get("content")
                    if isinstance(rc, list):
                        rc = "".join(x.get("text", "") for x in rc if isinstance(x, dict))
                    out.append({"role": "tool", "content": rc if rc is not None else "",
                                "tool_call_id": b.get("tool_use_id")})
            text = "".join(b.get("text", "") for b in c if b.get("type") == "text")
            if text.strip():
                out.append({"role": "user", "content": text})
    return out


def claude_code_file(path, limit=None):
    """Rows are per-API-request logs: each session appears once per request,
    as ever-longer prefixes. Keep only the longest row per session (keyed on
    its first two messages). The final `assistant_response` column carries
    only the reply's TEXT (its tool_use is lost), so it is not used -- the
    kept row's own messages_json already ends on complete turns."""
    pf = pq.ParquetFile(path)
    best = {}
    idx = 0
    for b in pf.iter_batches(batch_size=500, columns=["messages_json"]):
        for mj in b.column(0).to_pylist():
            if mj:
                raw = json.loads(mj)
                key = hashlib.md5(json.dumps(raw[:2], sort_keys=True).encode()).hexdigest()
                if key not in best or len(mj) > best[key][1]:
                    best[key] = (idx, len(mj))
            idx += 1
    keep = {i for i, _ in best.values()}
    idx = 0
    for b in pf.iter_batches(batch_size=500, columns=["messages_json", "system_prompt", "tools_json"]):
        for r in b.to_pylist():
            if limit and idx >= limit:
                return
            if idx not in keep:
                idx += 1
                yield None, "skip_shorter_prefix_of_same_session"
                continue
            idx += 1
            raw = loads_loose(r["messages_json"] or "") or []
            msgs = anthropic_msgs(raw)
            if r.get("system_prompt"):
                msgs.insert(0, {"role": "system", "content": r["system_prompt"]})
            yield finalize(msgs, r.get("tools_json"))


# ---- OpenTelemetry gen_ai spans (Exgentic) ---------------------------------

def otel_msgs(raw):
    out = []
    for m in raw:
        role = m.get("role")
        texts, thinks, calls, results = [], [], [], []
        for p in m.get("parts") or []:
            t = p.get("type")
            if t == "text" and p.get("content") and p["content"] != "(no content)":
                texts.append(p["content"])
            elif t == "thinking" and p.get("content"):
                thinks.append(p["content"])
            elif t == "tool_call":
                calls.append({"id": p.get("id"), "name": p.get("name"), "arguments": p.get("arguments")})
            elif t == "tool_call_response":
                r = p.get("result")
                if isinstance(r, str) and r.startswith('[{"type": "text"'):
                    r = loads_loose(r) or r
                if isinstance(r, list) and all(isinstance(x, dict) and x.get("type") == "text" for x in r):
                    r = "".join(x.get("text", "") for x in r)  # MCP content-parts wrapper
                results.append({"role": "tool", "tool_call_id": p.get("id"),
                                "content": r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)})
        out.extend(results)
        if role == "tool":
            continue
        if role == "assistant":
            out.append(convert_message({"role": "assistant", "content": "\n\n".join(texts),
                                        "tool_calls": calls or None,
                                        "reasoning_content": "\n\n".join(thinks) or None}))
        elif texts:
            out.append({"role": ROLE_MAP.get(role, role), "content": "\n\n".join(texts)})
    return out


def exgentic(row):
    """One row = one agent session; each span = one LLM call carrying the
    full input history. The call with the longest input history + its output
    is the whole trajectory."""
    best = None
    for sp in row.get("spans") or []:
        a = sp.get("attributes") or {}
        im = a.get("gen_ai.input.messages")
        if im and (best is None or len(im) > len(best.get("gen_ai.input.messages"))):
            best = a
    if best is None:
        return None, "skip_empty_or_invalid"
    raw = loads_loose(best["gen_ai.input.messages"]) or []
    raw += loads_loose(best.get("gen_ai.output.messages") or "[]") or []
    return finalize(otel_msgs(raw), best.get("gen_ai.tool.definitions"))


exgentic.columns = ["spans"]

# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

CONVERT = {
    # --- already chat-shaped ---
    "AgentGym__AgentTraj-L": std("conversations"),
    "internlm__Agent-FLAN": std("conversation"),
    "Solaris99__AgentBank": std("conversations"),
    "Lite-Coder__LiteCoder-SFT-Terminal-preview": std("conversations"),
    "Lite-Coder__LiteCoder-Terminal-SFT": std("conversations"),
    "HuggingFaceH4__no_robots": std("messages"),
    "HuggingFaceH4__ultrachat_200k": std("messages"),
    "allenai__WildChat": std("conversation"),
    "allenai__WildChat-1M": std("conversation"),
    "Kwai-Klear__SWE-smith-mini_swe_agent_plus-trajectories-66k": std("messages"),
    "R2E-Gym__R2EGym-SFT-Trajectories": std("messages"),
    "TIGER-Lab__SWE-Next-SFT-Trajectories": std("messages"),
    "TIGER-Lab__SWE-QA-Pro-SFT-Trajectories": std("messages", "tools"),
    "WaltonFuture__agentic-sft-new": std("messages"),
    "Nanbeige__ToolMind": std("conversations", "tools"),
    "groundhogLLM__ACC-dataset": std("dialogs"),
    "SWE-Gym__OpenHands-Sampled-Trajectories": swe_gym,
    "fuvty__tau-bench-synthetic": tau_bench,
    "Agent-Ark__Toucan-1.5M": toucan,
    "allenai__ultrafeedback_binarized_cleaned": ultrafeedback_chosen,
    "argilla__dpo-mix-7k": ultrafeedback_chosen,
    # --- pair / bespoke ---
    "AlicanKiraz0__Agentic-Chain-of-Thought-Coding-SFT-Dataset": pair("user", "assistant", "system"),
    "allenai__SciRIFF": pair("input", "output"),
    "camel-ai__code": camel_code,
    "deepmind__aqua_rat": aqua,
    "EleutherAI__hendrycks_math": hendrycks,
    "deepmind__code_contests": code_contests,
    "ibm__duorc": duorc,
    "FinGPT__fingpt-forecaster-dow30-202305-202405": fingpt,
    "glaiveai__glaive-function-calling-v2": glaive,
    "ise-uiuc__Magicoder-Evol-Instruct-110K": pair("instruction", "response"),
    "ise-uiuc__Magicoder-OSS-Instruct-75K": pair("problem", "solution"),
    "PersonalAILab__AFM-CodeAgent-SFT-Dataset": afm,
    "PersonalAILab__AFM-MHQA-Agent-SFT-Dataset": afm,
    "Jarrodbarnes__tau2-sft-v4-dataset": tau2_sft,
    "HuggingFaceTB__cosmopedia": cosmopedia,
    "Exgentic__agent-llm-traces": exgentic,
    "agent-data__misc-merged-claude-code-traces-v1": "FILE_LEVEL",
}

FILE_LEVEL = {"agent-data__misc-merged-claude-code-traces-v1": claude_code_file}

SKIP = {
    # held-out benchmarks / eval sets (user decision: never train on these)
    "HuggingFaceH4__MATH-500": "benchmark (test set)",
    "cais__mmlu": "benchmark (test set)",
    "HuggingFaceH4__mt_bench_prompts": "benchmark (test set)",
    "google-research-datasets__mbpp": "benchmark (test set)",
    "THU-KEG__AgentIF": "benchmark (test set)",
    "Qwen__AgentWorldBench": "benchmark (test set)",
    "ScaleAI__MCP-Atlas": "benchmark (test set)",
    "Salesforce__CRMArena": "benchmark (test set)",
    "ai-safety-institute__AgentHarm": "benchmark (harmful-behaviour eval, 6 rows)",
    "Lakera__b3-agent-security-benchmark-weak": "benchmark (security eval)",
    "eth-sri__agentbench": "benchmark (PR metadata, no trajectories)",
    "hkust-nlp__agentboard": "benchmark (goals only, no trajectories)",
    "JasperHaozhe__AgentGen-Bench": "benchmark manifest (no conversations)",
    # image data -- output is text-only, rows would lose their image
    "agentsea__wave-ui-25k": "image grounding data",
    "Hcompany__WebClick": "image grounding data",
    "cua-lite__UI-Genie-Agent": "image-based GUI agent data",
    "derek-thomas__ScienceQA": "image QA (and a benchmark)",
    # not chat data
    "AgentGym__AgentGym-RL-Data-ID": "item ids only",
    "automatelab__mcp-servers-tool-catalog": "tool catalog, no conversations",
    "BlueZeros__AgentEHR-Bench": "EHR tables, no conversations",
    "BytedTsinghua-SIA__CUDA-Agent-Ops-6K": "op specs + reference code, no prompt/response",
    "DeepNLP__ai-agent-teacher": "web listing metadata",
    "DeepNLP__Coding-Agent-Github-2025-Feb": "repo listing metadata",
    "DeepNLP__mcp-servers": "web listing metadata",
    "disco-eth__AgentsNet": "graph definitions",
    "McGill-NLP__WebLINX": "split index only (1 row)",
    "PresageLabs__NewsBench": "news articles, no conversations",
    "R2E-Gym__R2E-Gym-Subset": "environment specs, no trajectories",
    "SWE-Gym__SWE-Gym-Raw": "issues/patches, no trajectories",
    "II-Vietnam__Agentic-Multi-SWE-RL": "RL environment specs, no trajectories",
    "Snowflake__AgentWorldModel-1K": "environment specs, no trajectories",
    "data-for-agents__insta-150k-v3": "task + planned steps only, no executed trajectory",
    "data-agents__jupyter-agent-dataset": "corrupt parquet (footer missing)",
    # read and rejected on quality
    "artillerywu__DeepResearch-9K": "broken: every tool observation is replaced by a copy of the question (12,596/12,974 rows)",
    "inference-net__HALO-Gemini-3-Flash-AppWorld": "57 traces in flattened OTel attributes; too few to justify a bespoke parser",
    "armand0e__gpt-5.5-agent": "2 sessions of raw Codex event logs",
    "Srijan-Chakraborty__GLM-5.2-Agent-Distilled": "broken: tool results follow the user turn with no assistant tool call before them (315/318 rows) -- the calls are missing from the source",
    "Multi-Agent-LLMs__DEBATE": "multi-agent debate logs, no user/assistant structure",
    # duplicates of a kept dataset
    "HuggingFaceH4__ultrafeedback_binarized": "same UltraFeedback prompts as allenai__ultrafeedback_binarized_cleaned (kept)",
    "argilla__ultrafeedback-binarized-preferences-cleaned": "same UltraFeedback prompts as allenai__ultrafeedback_binarized_cleaned (kept)",
}


def dataset_of(fname):
    return re.sub(r"(_\d+)?\.parquet$", "", fname)


def run_file(fname, out_dir, limit=None):
    ds = dataset_of(fname)
    path = f"{SRC}/{fname}"
    out_path = f"{out_dir}/{fname[:-len('.parquet')]}.jsonl"
    reasons = Counter()
    rows_in = rows_out = 0
    tools_rows = 0
    err = None
    try:
        tmp = out_path + ".tmp"
        with open(tmp, "w") as fo:
            if ds in FILE_LEVEL:
                it = FILE_LEVEL[ds](path, limit)
            else:
                fn = CONVERT[ds]
                def gen():
                    n = 0
                    for b in pq.ParquetFile(path).iter_batches(batch_size=500, columns=fn.columns):
                        for r in b.to_pylist():
                            if limit and n >= limit:
                                return
                            n += 1
                            try:
                                yield fn(r)
                            except Exception:
                                yield None, "skip_convert_exception"
                it = gen()
            for row, why in it:
                rows_in += 1
                reasons[why] += 1
                if row is not None:
                    try:
                        line = json.dumps(row, ensure_ascii=False)
                    except (TypeError, ValueError):
                        reasons[why] -= 1
                        reasons["skip_not_json_serializable"] += 1
                        continue
                    fo.write(line + "\n")
                    rows_out += 1
                    tools_rows += "tools" in row
        if rows_out:
            os.replace(tmp, out_path)
        else:
            os.remove(tmp)
    except Exception:
        err = traceback.format_exc(limit=5)
    return {"file": fname, "dataset": ds, "rows_in": rows_in, "rows_out": rows_out,
            "rows_with_tools": tools_rows, "reasons": dict(reasons), "error": err}


def main():
    args = sys.argv[1:]
    limit = None
    out_dir = OUT_DIR
    if args[:1] == ["--limit"]:
        limit = int(args[1])
        args = args[2:]
        out_dir = "/tmp/claude-1020/-projects-data-datasets-code-data-sai-rupesh-taxonomy/15579090-796d-430c-8625-c2eab15b0960/scratchpad/convert_test"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(SRC) if f.endswith(".parquet"))
    unknown = sorted({dataset_of(f) for f in files} - set(CONVERT) - set(SKIP))
    if unknown:
        sys.exit(f"datasets with neither a converter nor a skip reason: {unknown}")
    todo = [f for f in files if dataset_of(f) in CONVERT and (not args or dataset_of(f) in args)]
    # biggest first so the long jobs start early
    todo.sort(key=lambda f: -os.path.getsize(f"{SRC}/{f}"))
    print(f"{len(todo)} source files to convert, {len(SKIP)} datasets skipped", flush=True)

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(run_file, f, out_dir, limit): f for f in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            results.append(r)
            print(f"[{i}/{len(todo)}] {r['file']}: in={r['rows_in']:,} out={r['rows_out']:,} "
                  f"tools={r['rows_with_tools']:,} {'ERROR' if r['error'] else ''} {r['reasons']}", flush=True)
            if r["error"]:
                print(r["error"], flush=True)

    if limit is None:
        per_ds = defaultdict(lambda: {"rows_in": 0, "rows_out": 0, "rows_with_tools": 0, "reasons": Counter(), "files": 0})
        for r in results:
            d = per_ds[r["dataset"]]
            d["files"] += 1
            for k in ("rows_in", "rows_out", "rows_with_tools"):
                d[k] += r[k]
            d["reasons"].update(r["reasons"])
        rpath = f"{REPORT_DIR}/stage1_conversion_report.json"
        old = json.load(open(rpath)) if (args and os.path.exists(rpath)) else {"per_dataset": {}, "errors": {}}
        per = old["per_dataset"]
        per.update({k: {**v, "reasons": dict(v["reasons"])} for k, v in per_ds.items()})
        errors = {f: e for f, e in old.get("errors", {}).items() if dataset_of(f) not in per_ds}
        errors.update({r["file"]: r["error"] for r in results if r["error"]})
        rep = {
            "rows_in": sum(v["rows_in"] for v in per.values()),
            "rows_out": sum(v["rows_out"] for v in per.values()),
            "datasets_converted": len(per),
            "datasets_skipped": SKIP,
            "per_dataset": dict(sorted(per.items())),
            "errors": errors,
            "last_run_min": round((time.time() - t0) / 60, 1),
            "last_run_datasets": sorted(per_ds),
        }
        json.dump(rep, open(rpath, "w"), indent=1)
        print(f"\nDONE rows_in={rep['rows_in']:,} rows_out={rep['rows_out']:,} (this run {rep['last_run_min']} min)")


if __name__ == "__main__":
    main()
