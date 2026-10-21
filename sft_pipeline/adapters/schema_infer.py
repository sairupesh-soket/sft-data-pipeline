"""Canonical SFT normalizer v4: any schema -> conversation + dedup keys.

Same public API as sft_normalize.py (plan/to_messages/coerce_chat/dedup_keys/
canonical) so it's a drop-in replacement -- swap the import in normalize_job.py
and nothing else changes. v4 = v3's fixes, plus five more, each individually
confirmed on real content (never invented) and signed off one at a time:

6. code_contests_instruct (7.7M rows): a lone `text` column holding
   "### Prompt\n\n<real problem>\n\n### Response\n\n<real solution>" --
   both sides genuinely authored, just glued with markdown headers instead
   of separate columns or chat tags. Added parse_prompt_response().

7. open-paws tool-use-llama-format (5.0M rows): already had a `messages`
   column (chat mode already triggers) but its content is rendered in
   Meta's Llama-3 template (`<|start_header_id|>role<|end_header_id|>
   content<|eot_id|>`, with `<|eom_id|>` for tool calls and an `ipython`
   role for tool results) which the existing parsers don't recognize.
   Added parse_llama().

8. judicialmind legal-training-dataset (3.7M rows): `query` was already
   recognized; its paired column is named `positive` (retrieval-training
   terminology for "the passage that answers this query"), which reads
   like a genuine answer when checked against real rows. Added as an
   ASSIST alias.

9. TIRData math_stage2_hard_2 (2.48M rows): its only content column,
   `my_solu`, is already claimed as the ASSIST field by exact match -- but
   its actual content is a full "User: ...\n\nAssistant: ..." transcript,
   so the old code stuffed the whole thing (question included) into the
   assistant slot and then rejected the row for having no separate user
   turn. Fixed generally: whenever no user column was found for `fields`
   mode, try parsing the chosen assistant column's raw content as a
   transcript first (coerce_chat); only fall back to using it as one flat
   turn if that parse finds nothing. This is a general pattern, not
   file-specific -- it may recover other files shaped the same way too.

10. jablonkagroup__qm9 (2.4M rows): its `input` column already glues the
    task, description AND result together ("Task: ...\nDescription:
    ...\nResult: <answer>"), with the paired `output` column left empty.
    Fixed generally: whenever the matched assistant column is empty for a
    row, look for a trailing "\nResult:"/"\nAnswer:"/"\nOutput:" marker
    inside the user column's own text and split there instead of giving up.

Also relaxed the fallback-chat gate from v3 (<=3 columns) to <=10, since
code_contests_instruct has 8 columns; the fallback is still self-correcting
(a non-matching column just yields 0 valid rows, same as before) so this
only widens coverage, and 10 still excludes wide raw-corpus tables like
MathGenie's MathCode-Pile-Full (20 columns) that were deliberately left out.

v3 = v2's two fixes, plus three more found by
reading real content in the still-unrecognized files after v2:

3. Q/A single-letter columns: math-ai__StackMathQA (8.1M rows) uses columns
   literally named 'Q' and 'A'. v1/v2 already special-cased 'q' as a user
   alias but had no matching 'a' for assistant -- added.

4. Compound field layouts: Kyudan__MathBridge / aaai25withanonymous__MathBridge
   (46.4M rows combined) split one example across `context_before`,
   `equation`, `context_after` (the source text) and `spoken_english` (the
   target) -- a real task (read the equation aloud), just not shaped like a
   single user/assistant column pair. Added 'spoken_english'/'spoken' as
   assistant aliases and a compound-user rule that concatenates
   context_before + equation + context_after when 'equation' is present,
   mirroring the existing alpaca instruction+input special case.

5. Transcript-in-a-single-column: ykarout__code-reasoning-phi4-template
   (9.4M rows) has exactly one column, `text`, holding a fully rendered
   ChatML conversation (`<|im_start|>role<|im_sep|>content<|im_end|>`).
   Added a parse_chatml() parser (alongside the existing plain "User:"/
   "Assistant:" transcript parser) and a last-resort plan() rule: when
   nothing else matches AND the file has very few columns (<=3, so this
   fallback can't misfire on a wide raw-corpus table that merely happens to
   have a column named "text" among twenty unrelated metadata columns),
   try treating that lone free-text column as a full transcript. If the
   content doesn't actually contain transcript markers, parsing simply
   returns None and the row is rejected exactly as before -- this fallback
   can only add valid rows, never produce garbage from plain prose.

Deliberately NOT attempted here (flagged to the user, not silently
"fixed"): datasets that are really text corpora with no natural question at
all -- EleutherAI's synthetic bios benchmark (285M rows: prose describing
invented facts, meant to test memorization, not to answer questions),
MathGenie's MathCode-Pile, Salesforce's FinTrain, and HDFS log lines. Turning
those into Q&A would mean *inventing* the question from metadata (e.g.
templating "What university did X attend?" from a structured field), which
changes the character of the data and was explicitly declined for the bios
case -- treated consistently here rather than silently building the same
kind of synthesis into the adapter.

Two fixes carried over from v2, both found by auditing OUTPUT/:

1. BUG FIX (crash): v1's to_messages() chat-mode fast path checked only the
   FIRST turn's keys (`set(raw[0].keys())=={'role','content'}`) then indexed
   `t['role']`/`t['content']` on every turn. A list whose first turn happens
   to be canonical but whose later turns aren't raises KeyError, and because
   normalize_job.one_file() wraps the whole per-file loop in one try/except,
   that KeyError on a SINGLE row silently discards the entire file's output.
   Reproduces on Reasoning/omersaidd__ReasoningV1.parquet. Fixed by using
   .get() throughout and applying ROLE_ALIAS per-turn instead of trusting
   the first turn's shape for all of them. to_messages() is also now wrapped
   in a top-level try/except so no future edge case can take down a whole file.

2. COVERAGE FIX (missed columns): v1's plan() only matches a column if its
   name is an EXACT hit in SYSTEM/USER/ASSIST/THINK/CHAT. Auditing the
   "no adapter" files found ~123 datasets (~27M rows) with obviously
   question/answer-shaped columns that v1 misses purely on spelling --
   reference_answer, question_ko/answer_ko, responses (plural),
   output_concept, instruction_ta/answer_ta, etc. v2 adds a second-pass
   fuzzy match: split a column name into lowercase word tokens (on
   underscores/camelCase, plural 's' stripped) and match if a root word is
   ONE OF the tokens -- but only after the exact-name pass fails, and only
   for columns with none of a metadata-token blocklist (language, id, type,
   count, label, ... ) that would otherwise cause a column like
   input_language to be mistaken for a text column just because it contains
   the token "input" (this exact false positive was found and is why the
   blocklist exists). Being a little permissive here is safe: plan() only
   picks candidate COLUMNS, it never looks at their contents -- a column
   that matches the name pattern but holds junk/empty values still gets
   rejected downstream by canonicalize_turns()/validate(), same as v1.
"""
import re, json, ast, hashlib, unicodedata

SCHEMA_VERSION = "sft-v1"

SYSTEM = ['system','system_prompt','sys_prompt','persona','role_prompt']
USER   = ['question','problem','prompt','query','input','human','user','inputs',
          'user_prompt','problem_statement','instruction','original_question','q',
          # low-priority fallbacks, only used when nothing above matched
          'text','context','document','passage','sentence']
ASSIST = ['output','answer','response','solution','generated_solution','completion',
          'gpt','assistant','model_response','my_solu','chosen','target',
          'a','spoken_english','spoken','positive']
THINK  = ['<think>','think','reasoning','chain_of_thought','cot','rationale',
          'gt_cot','reasoning_content','thinking','thought']
CHAT   = ['messages','conversations','conversation','turns','dialogue','exchanges']
# instruction+input is alpaca: they CONCATENATE into one user turn, never two turns
ALPACA_TASK = 'instruction'
ALPACA_DATA = 'input'

ROLE_ALIAS = {'human':'user','usr':'user','gpt':'assistant','chatgpt':'assistant',
              'bot':'assistant','ai':'assistant','model':'assistant','sys':'system',
              'user':'user','assistant':'assistant','system':'system','tool':'tool',
              'function':'tool','observation':'tool','ipython':'tool'}

# columns matched via ASSIST_ROOTS whose own content should be checked for an
# embedded transcript before being used as one flat turn (fix #9)
_TRANSCRIPT_MARKS = re.compile(
    r'(?:User\s*:|<\|im_start\|>|<\|start_header_id\|>|###\s*Prompt\b)', re.I)
# a matched user column may itself glue the answer on the end when the
# assistant column is empty (fix #10)
_TRAILING_RESULT_RE = re.compile(r'\n\s*(?:Result|Answer|Output)\s*:\s*(.+)\Z', re.S)

# ---- v2 fuzzy fallback: single-word roots + a blocklist of metadata tokens
# that must never count as a match even though they might share a substring
# with a root (input_language contains "input" but is not a text column).
SYSTEM_ROOTS = {'system', 'persona'}
USER_ROOTS   = {'question', 'problem', 'prompt', 'query', 'input', 'human', 'user',
                'instruction', 'request', 'premise', 'q'}
USER_FALLBACK_ROOTS = {'text', 'context', 'document', 'passage', 'sentence'}
ASSIST_ROOTS = {'output', 'answer', 'response', 'solution', 'completion', 'gpt',
                'assistant', 'chosen', 'target', 'reply', 'spoken'}
# equation+context is MathBridge-shaped: concatenate context_before + equation +
# context_after into ONE user turn, same spirit as the alpaca instruction+input rule
EQUATION_KEY = 'equation'
EQUATION_CONTEXT_COLS = ['context_before', 'equation', 'context_after']
THINK_ROOTS  = {'think', 'reasoning', 'cot', 'rationale', 'thought', 'thinking'}
CHAT_ROOTS   = {'messages', 'conversations', 'conversation', 'turns', 'dialogue',
                'exchanges', 'chat'}
BLOCKED_TOKENS = {'language', 'lang', 'type', 'id', 'ids', 'count', 'score', 'label',
                   'category', 'class', 'level', 'index', 'flag', 'date', 'year',
                   'month', 'day', 'code', 'format', 'model', 'version', 'num',
                   'length', 'size', 'hash', 'key', 'url', 'path', 'source', 'name',
                   'difficulty', 'domain', 'made_by', 'multi_turn', 'embedding',
                   'embeddings', 'cluster'}

def _tokenize(name):
    s = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', name)
    toks = [t.lower() for t in re.split(r'[^a-zA-Z0-9]+', s) if t]
    out = set()
    for t in toks:
        out.add(t)
        if len(t) > 3 and t.endswith('s'):
            out.add(t[:-1])
    return out

def _fuzzy_match(col, roots):
    toks = _tokenize(col)
    if toks & BLOCKED_TOKENS:
        return False
    return bool(toks & roots)

_WS = re.compile(r'\s+')

def norm_text(s):
    """Normalization used ONLY for hashing. Never mutates stored content."""
    if not s: return ''
    s = unicodedata.normalize('NFKC', s)
    s = s.replace('​','').replace('﻿','')
    s = _WS.sub(' ', s)
    return s.strip().lower()

def h(s):
    return hashlib.blake2b(s.encode('utf-8','ignore'), digest_size=16).hexdigest()

_USER_SPLIT = re.compile(r'(?m)^\s*User:\s*')
_THINK_RE   = re.compile(r'<think>(.*?)</think>', re.S)
_PREFIX_RE  = re.compile(r'^[ \t"\'\\\[]*(user|human|assistant|gpt|system|ai|bot)\s*[::]\s*', re.I)
_ROLE_MARK  = re.compile(r'(?:^|\n)[ \t"\',\\\[]*(User|Assistant|System|Human|GPT|AI)\s*[::]\s*', re.I)

def split_prefixed(t):
    """'User: hi' -> ('user','hi'); returns (None, text) when unprefixed."""
    m=_PREFIX_RE.match(t)
    if not m: return None, t
    return ROLE_ALIAS.get(m.group(1).lower()), t[m.end():]
_ANSWER_RE  = re.compile(r'<answer>(.*?)</answer>', re.S)

def _extract_body(blk):
    """Pull <think>/<answer> out of a block, keeping think inline."""
    th=_THINK_RE.search(blk); an=_ANSWER_RE.search(blk)
    if not (th or an): return blk.strip()
    parts=[]
    if th: parts.append('<think>'+th.group(1).strip()+'</think>')
    if an: parts.append(an.group(1).strip())
    return '\n\n'.join(parts)

def parse_bracketed(s):
    """House format: 'User: q\n\n[<think>..</think> <answer>..</answer>]' (no Assistant: marker)."""
    if 'User:' not in s: return None
    out=[]
    for chunk in re.split(r'(?:^|\n)\s*User\s*:\s*', s):
        if not chunk.strip(): continue
        i=chunk.find('\n\n[')
        u, blk = (chunk, '') if i==-1 else (chunk[:i], chunk[i+3:].rstrip().rstrip(']'))
        u=u.strip()
        if u: out.append({'role':'user','content':u})
        body=_extract_body(blk)
        if body: out.append({'role':'assistant','content':body})
    if not out: return None
    if not any(m['role']=='user' for m in out): return None
    if not any(m['role']=='assistant' for m in out): return None
    return out

def parse_transcript(s):
    """Tolerant parser for transcripts stored as text, including malformed
    JSON-ish arrays whose LaTeX escapes (\\( , \\) ) break json.loads.
    Splits on role markers wherever they appear, ignoring quote/bracket noise."""
    marks=list(_ROLE_MARK.finditer(s))
    if not marks: return None
    out=[]
    for i,m in enumerate(marks):
        role=ROLE_ALIAS.get(m.group(1).lower())
        end=marks[i+1].start() if i+1<len(marks) else len(s)
        body=s[m.end():end]
        body=body.strip().strip(',').strip()
        # drop wrapper punctuation left over from the broken array / bracket block
        while body and body[-1] in '"]\'': body=body[:-1].rstrip()
        while body and body[0] in '"[\'': body=body[1:].lstrip()
        body=_extract_body(body).strip()
        if not body or not role: continue
        if out and out[-1]['role']==role:
            out[-1]['content'] += '\n\n'+body       # merge consecutive same-role
        else:
            out.append({'role':role,'content':body})
    if not out: return None
    if not any(m['role']=='user' for m in out): return None
    if not any(m['role']=='assistant' for m in out): return None
    return out

_CHATML_RE = re.compile(r'<\|im_start\|>\s*(system|user|assistant|tool)\s*(?:<\|im_sep\|>)?(.*?)<\|im_end\|>', re.S | re.I)

def parse_chatml(s):
    """ChatML/Phi-style rendered turns: <|im_start|>role<|im_sep|>content<|im_end|>."""
    matches = list(_CHATML_RE.finditer(s))
    if not matches: return None
    out = []
    for m in matches:
        role = ROLE_ALIAS.get(m.group(1).lower(), m.group(1).lower())
        body = m.group(2).strip()
        if not body: continue
        if out and out[-1]['role']==role:
            out[-1]['content'] += '\n\n'+body
        else:
            out.append({'role':role,'content':body})
    if not out: return None
    if not any(m['role']=='user' for m in out): return None
    if not any(m['role']=='assistant' for m in out): return None
    return out

_LLAMA_RE = re.compile(
    r'<\|start_header_id\|>\s*(system|user|assistant|ipython|tool)\s*<\|end_header_id\|>'
    r'(.*?)(?:<\|eot_id\|>|<\|eom_id\|>|(?=<\|start_header_id\|>)|\Z)', re.S | re.I)

def parse_llama(s):
    """Meta Llama-3 chat template: <|start_header_id|>role<|end_header_id|>
    content<|eot_id|> (tool calls end on <|eom_id|> instead)."""
    matches = list(_LLAMA_RE.finditer(s))
    if not matches: return None
    out = []
    for m in matches:
        role = ROLE_ALIAS.get(m.group(1).lower(), m.group(1).lower())
        body = m.group(2).strip()
        if not body: continue
        if out and out[-1]['role']==role:
            out[-1]['content'] += '\n\n'+body
        else:
            out.append({'role':role,'content':body})
    if not out: return None
    if not any(m['role']=='user' for m in out): return None
    if not any(m['role']=='assistant' for m in out): return None
    return out

_PROMPT_RESPONSE_RE = re.compile(
    r'#{1,4}\s*Prompt\s*\n+(.*?)\n+#{1,4}\s*Response\s*\n+(.*)\Z', re.S | re.I)

def parse_prompt_response(s):
    """House format: '### Prompt\\n\\n<task>\\n\\n### Response\\n\\n<answer>'."""
    m = _PROMPT_RESPONSE_RE.search(s)
    if not m: return None
    user, asst = m.group(1).strip(), m.group(2).strip()
    if not user or not asst: return None
    return [{'role': 'user', 'content': user}, {'role': 'assistant', 'content': asst}]

def coerce_chat(raw):
    """messages/conversations may be a real list, a JSON string, a Python repr, or house text."""
    if raw is None: return None
    if isinstance(raw,(list,tuple)): return list(raw)
    if not isinstance(raw,str): return None
    t=raw.strip()
    if not t: return None
    if t[0] in '[{':
        for fn in (json.loads, ast.literal_eval):
            try:
                v=fn(t)
                if isinstance(v,dict): v=[v]
                if isinstance(v,(list,tuple)): return list(v)
            except Exception: pass
    return (parse_transcript(raw) or parse_llama(raw) or parse_chatml(raw)
            or parse_prompt_response(raw) or parse_bracketed(raw))

def plan(colnames):
    """Compute an adapter plan ONCE per signature, not per row.

    Two passes: exact-name (identical to v1, so every file v1 already
    handles keeps the same plan) then, only for slots still unfilled, a
    fuzzy token pass that catches near-miss spellings (reference_answer,
    question_ko, responses, ...)."""
    low = {c.lower(): c for c in colnames}
    p = {'mode': None}

    for k in CHAT:
        if k in low:
            p['mode']='chat'; p['col']=low[k]; return p

    sysc  = next((low[k] for k in SYSTEM if k in low), None)
    think = next((low[k] for k in THINK  if k in low), None)
    asst  = next((low[k] for k in ASSIST if k in low), None)
    if ALPACA_TASK in low and ALPACA_DATA in low:
        usr, extra = low[ALPACA_TASK], low[ALPACA_DATA]
    else:
        usr = next((low[k] for k in USER if k in low), None); extra = None

    if not asst:
        for name in colnames:
            if _fuzzy_match(name, CHAT_ROOTS):
                p['mode']='chat'; p['col']=name; return p
        asst = next((c for c in colnames if _fuzzy_match(c, ASSIST_ROOTS)), None)
    user_join = None
    if not usr:
        task_c = next((c for c in colnames if _fuzzy_match(c, {'instruction'})), None)
        data_c = next((c for c in colnames if _fuzzy_match(c, {'input'})), None)
        if task_c and data_c and task_c != data_c:
            usr, extra = task_c, data_c
        elif EQUATION_KEY in low:
            join_cols = [low[c] for c in EQUATION_CONTEXT_COLS if c in low]
            if join_cols:
                user_join = join_cols
                usr = join_cols[0]
        else:
            usr = next((c for c in colnames if _fuzzy_match(c, USER_ROOTS)), None)
            if not usr:
                usr = next((c for c in colnames if _fuzzy_match(c, USER_FALLBACK_ROOTS)), None)
    if not sysc:
        sysc = next((c for c in colnames if _fuzzy_match(c, SYSTEM_ROOTS)), None)
    if not think:
        think = next((c for c in colnames if _fuzzy_match(c, THINK_ROOTS)), None)

    if not asst and len(colnames) <= 10:
        # last resort: a lone free-text column might itself BE a fully
        # rendered transcript (ChatML/Llama markers, "User:"/"Assistant:"
        # markers, "### Prompt"/"### Response" markers) rather than one
        # half of a field pair -- let coerce_chat's parsers try it. Gated
        # on column count (raised from 3 to 10 in v4 for code_contests_
        # instruct's 8 columns) so this can't misfire on a wide raw table
        # that merely has a "text" column among many unrelated metadata
        # ones (e.g. MathGenie's 20-column MathCode-Pile-Full).
        text_col = (next((c for c in colnames if _fuzzy_match(c, USER_FALLBACK_ROOTS)), None)
                    or next((c for c in colnames if _fuzzy_match(c, USER_ROOTS)), None))
        if text_col:
            p['mode'] = 'chat'; p['col'] = text_col; return p

    if not asst: return {'mode':None}
    p.update(mode='fields', system=sysc, user=usr, user_extra=extra,
             user_join=user_join, think=think, assistant=asst)
    return p

def _mk(role, content):
    return {'role': role, 'content': content}

def _to_messages_impl(row, p):
    if p['mode']=='chat':
        raw = coerce_chat(row.get(p['col']))
        if not raw: return None
        if raw and isinstance(raw[0], dict) and set(raw[0].keys())=={'role','content'}:
            out=[]
            for t in raw:
                if not isinstance(t, dict): continue
                cont = t.get('content')
                if not (isinstance(cont, str) and cont.strip()): continue
                role_raw = t.get('role')
                r = ROLE_ALIAS.get(str(role_raw).strip().lower()) if role_raw is not None else None
                if r is None:
                    r = 'user' if len(out)%2==0 else 'assistant'
                out.append(_mk(r, cont))
            return out if any(m['role']=='user' for m in out) else None
        out=[]
        for i, t in enumerate(raw):
            if isinstance(t, str):
                if not t.strip(): continue
                r, body = split_prefixed(t)
                if r is None: r = 'user' if len(out)%2==0 else 'assistant'
                body = body.strip()
                if not body: continue
                th=_THINK_RE.search(body); an=_ANSWER_RE.search(body)
                if th or an:
                    parts=[]
                    if th: parts.append('<think>'+th.group(1).strip()+'</think>')
                    if an: parts.append(an.group(1).strip())
                    body='\n\n'.join(parts)
                out.append(_mk(r, body)); continue
            if not isinstance(t, dict): return None
            role = t.get('role') or t.get('from') or t.get('speaker') or t.get('sender')
            cont = t.get('content') or t.get('value') or t.get('text') or t.get('message') or ''
            if not str(cont).strip(): continue
            r = ROLE_ALIAS.get(str(role).strip().lower()) if role is not None else None
            if r is None:
                r = 'user' if len(out)%2==0 else 'assistant'
            cont=str(cont)
            if r is None:
                pr,_b = split_prefixed(cont)
                if pr: r=pr
            out.append(_mk(r, cont))
        if not out or not any(m['role']=='user' for m in out): return None
        if not any(m['role']=='assistant' for m in out): return None
        return out

    if p['mode']!='fields': return None
    out=[]
    if p.get('system'):
        v=row.get(p['system'])
        if v and str(v).strip(): out.append(_mk('system', str(v)))

    if p.get('user_join'):
        parts = [str(row.get(c) or '').strip() for c in p['user_join']]
        u = '\n\n'.join(x for x in parts if x)
    else:
        u = str(row.get(p['user']) or '').strip() if p.get('user') else ''
        if p.get('user_extra'):
            ex = str(row.get(p['user_extra']) or '').strip()
            if ex: u = f"{u}\n\n{ex}" if u else ex

    a = str(row.get(p['assistant']) or '').strip()

    # fix #9: with no separate user column at all, the sole content column
    # (a flat ASSIST match by name) might itself BE a full transcript --
    # e.g. TIRData's `my_solu` holds "User: ...\n\nAssistant: ...", not a
    # flat answer -- regardless of whether it's empty. Check it whether or
    # not `a` looks filled, since a non-empty value is exactly what this
    # case looks like from the outside.
    candidate = None
    if not u and a and _TRANSCRIPT_MARKS.search(a):
        candidate = a
    elif not a and u and _TRANSCRIPT_MARKS.search(u):
        # fix #9b: symmetric case -- the matched "user"/"assistant" columns
        # are column-NAME matches only (plan() never looks at values), so a
        # row can match a real answer column that's null THIS row while the
        # actual prompt+response live together in the user column instead
        # (e.g. code_contests_instruct's `solution` is always null; the
        # real "### Prompt"/"### Response" pair lives in `text`, i.e. `u`).
        candidate = u
    if candidate:
        embedded = coerce_chat(candidate)
        if embedded:
            turns = [_mk(t.get('role'), str(t.get('content','')).strip())
                      for t in embedded if isinstance(t, dict) and str(t.get('content','')).strip()]
            if (turns and any(t['role']=='user' for t in turns)
                    and any(t['role']=='assistant' for t in turns)):
                return out + turns

    if not a and u:
        # fix #10: the matched user column may itself glue the answer onto
        # the end (e.g. qm9's `input`: "Task: ...\nDescription: ...\nResult:
        # <answer>") when the paired assistant column is left empty. Split
        # it out rather than rejecting the row for a missing answer.
        m = _TRAILING_RESULT_RE.search(u)
        if m:
            a = m.group(1).strip()
            u = u[:m.start()].strip()

    if u: out.append(_mk('user', u))
    if p.get('think'):
        th = str(row.get(p['think']) or '').strip()
        if th: a = f"<think>{th}</think>\n\n{a}" if a else f"<think>{th}</think>"
    if not a: return None
    out.append(_mk('assistant', a))
    return out if any(m['role']=='user' for m in out) else None

def to_messages(row, p):
    """Apply a precomputed plan to one row. Returns list of turns or None.

    Never raises: a bad row in an otherwise-good file should be dropped by
    canonicalize_turns()/validate() downstream, not take out the whole file
    (see module docstring, fix #1)."""
    try:
        return _to_messages_impl(row, p)
    except Exception:
        return None

def dedup_keys(msgs):
    """prompt_hash groups same-question rows; content_hash is exact-dup."""
    pre = [m for m in msgs if m['role'] in ('system','user')]
    prompt_sig  = '\n'.join(f"{m['role']}:{norm_text(m['content'])}" for m in pre)
    content_sig = '\n'.join(f"{m['role']}:{norm_text(m['content'])}" for m in msgs)
    return h(prompt_sig), h(content_sig)

def canonical(row, p, meta):
    msgs = to_messages(row, p)
    if not msgs: return None
    ph, ch = dedup_keys(msgs)
    nchars = sum(len(m['content']) for m in msgs)
    return {
        'id': ch, 'messages': msgs,
        'n_turns': len(msgs),
        'multi_turn': sum(1 for m in msgs if m['role']=='user') > 1,
        'num_chars': nchars, 'num_tokens_est': nchars//4,
        'prompt_hash': ph, 'content_hash': ch,
        'domain': meta.get('domain',''), 'source': meta.get('source',''),
        'task': meta.get('task',''), 'language': meta.get('language','English'),
        'difficulty_level': meta.get('difficulty_level','UNKNOWN'),
        'made_by': meta.get('made_by',''),
        'src_file': meta.get('src_file',''), 'src_signature': meta.get('sig',''),
        'schema_version': SCHEMA_VERSION,
    }
