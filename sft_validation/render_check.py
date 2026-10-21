"""eka template render check (eka SentencePiece tokenizer + eka chat template).
Paths can be changed with the env vars EKA_SPM and EKA_TEMPLATE."""
import json, os, re
from collections import Counter

SPM = os.environ.get("EKA_SPM", "/projects/data/teams/all/tokenizer/spm/eka_v0_1_1.model")
TEMPLATE = os.environ.get("EKA_TEMPLATE", "/projects/data/teams/all/tokenizer/hf/eka_v0_2_0/chat_template.jinja")
TIER1 = {"<|system|>", "<|user|>", "<|assistant|>", "<|observation|>", "<think>", "</think>", "<tool_call>", "</tool_call>",
         "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "<tool_response>", "</tool_response>", "<tools>", "</tools>"}
TIER1B = {"<function>", "</function>", "<name>", "</name>", "<description>", "</description>", "<parameters>", "</parameters>",
          "<parameter>", "</parameter>", "<type>", "</type>", "<enum>", "</enum>", "<required>", "</required>",
          "<IMPORTANT>", "</IMPORTANT>", "<reminder>", "</reminder>"}
_P = _SP = _TPL = None


def init():
    global _P, _SP, _TPL
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as pb
    m = pb.ModelProto()
    m.ParseFromString(open(SPM, "rb").read())
    _P = {p.piece for p in m.pieces if p.type == pb.ModelProto.SentencePiece.USER_DEFINED}
    _SP = spm.SentencePieceProcessor()
    _SP.Load(SPM)
    import jinja2
    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False, extensions=["jinja2.ext.loopcontrols"])
    env.filters["fromjson"] = json.loads
    env.filters["tojson"] = lambda x, **k: json.dumps(x, ensure_ascii=False)
    env.filters["items"] = lambda d: d.items()
    _TPL = env.from_string(open(TEMPLATE).read())


def render_check(r):
    msgs = r.get("messages") or []
    try:
        s = _TPL.render(messages=msgs, tools=r.get("tools"), enable_thinking=True, add_generation_prompt=False)
    except Exception as e:
        return "render_error"
    pieces = [_SP.IdToPiece(i) for i in _SP.EncodeAsIds(s)]
    pc = Counter(p for p in pieces if p in TIER1)
    if r.get("tools"):  # the template's own tools block also contains example tags; subtract its counts
        try:
            base = _TPL.render(messages=[], tools=r.get("tools"), enable_thinking=True, add_generation_prompt=False)
        except Exception:
            return "render_error"
        pc.subtract(Counter(p for p in (_SP.IdToPiece(i) for i in _SP.EncodeAsIds(base)) if p in TIER1))
    nu = sum(1 for m in msgs if m.get("role") == "user")
    na = sum(1 for m in msgs if m.get("role") == "assistant")
    nt = sum(1 for m in msgs if m.get("role") == "tool")
    ntc = sum(len(m.get("tool_calls") or []) for m in msgs if m.get("role") == "assistant")
    bad = []
    if pc["<|user|>"] != nu: bad.append("user")
    if pc["<|assistant|>"] != na: bad.append("assistant")
    if pc["<think>"] != na or pc["</think>"] != na: bad.append("think")
    if pc["<tool_call>"] != ntc or pc["</tool_call>"] != ntc: bad.append("tool_call")
    if pc["<tool_response>"] != nt or pc["</tool_response>"] != nt: bad.append("tool_response")
    return "render_ok" if not bad else "render_mismatch:" + "+".join(bad)
