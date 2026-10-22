# SFT → OpenAI JSONL Pipeline (K8s Handoff)

Collective notes from the sft-pipeline design/review work. Use this to prepare Kubernetes Jobs/pods.

---

## 1. Goal

Convert HuggingFace-style parquet under `SFT/OUTPUT` into **OpenAI chat SFT JSONL**:

```json
{"messages": [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

- Training files must **not** include `_meta`.
- Optional `tools` / `parallel_tool_calls` only when the source already has tool turns.
- Roles are always OpenAI: `system` | `user` | `assistant` | `tool` (tool rare).

---

## 2. Source data

| Item | Value |
|---|---|
| Root | `/projects/data/datasets/translation_data/SFT/OUTPUT` (or mount as `SFT_SRC`) |
| Scale | **~1.57B** parquet rows, **~1.17 TB**, ~3.3k files, many schemas |
| Mapper helper | `sft_normalize.py` in `SFT/DEDUP_PIPELINE` (mount as `SFT_DEDUP`) |

Most of the dump is **not** SFT (bios, gnomAD, PubChem, tensors). Those stay dropped.

### Usable / convertible pool (~645M)

| Bucket | Rows | Share of convertible | Notes |
|---|---|---|---|
| **Phase 1 — Chat** | **~128.5M** | ~20% | `messages` / `conversations` / … |
| **Phase 2 — Instruction pairs** | **~476.3M** | ~74% | Both prompt + reply columns (alpaca, Q–A, …) |
| **Phase 3 — Extra aliases** | **~15.1M** | ~2% | `Q`/`A`, `long_answer`, … missed by base mapper |
| **Total** | **~645M** | 100% | Schema-mappable to OpenAI SFT |

Excluded from Phase 2: **~25M** answer-only files (`my_solu`, etc. with no user column).

Still dropped overall: **~920–940M** non-SFT tables.

---

## 3. Three phases (run in parallel on K8s)

One script: **`pipeline/run_phases.py`** (self-contained; no need for `phase3_aliases.py`).

| Phase | Input shape | Convert to | Approx rows | Files (order of) |
|---|---|---|---|---|
| **1** | Chat columns (`messages`, `conversations`, …) | `messages[]` | 129M | ~514 |
| **2** | Named pairs with **both** user + assistant cols | `messages[]` | 476M | ~1.3k |
| **3** | Alias remap only (`Q`→user, `A`→assistant, …) | `messages[]` | 15M | ~35 |

Phase 3 does **not** rewrite parquet on disk. It maps column **names → roles**, then uses the same converter as Phase 2.

### Phase 3 alias pairs (explicit)

```
q → a
question → long_answer | reference_answer | correct_answer | final_answer | answer_text | final_response
prompt → reference
instruction → final_answer
user_query → enhanced_completion
query → positive          # retrieval-style; still mapped user/assistant
```

Do **not** treat `label` / `model` / `summary` as answers (false positives).

---

## 4. System prompts

- **Not stripped** by the pipeline.
- Only **~3.1% (~14.8M)** of Phase 2 rows have a system column (`system` / `system_prompt` / …).
- **~97%** of Phase 2 are user+assistant only because upstream data had no system field.
- Phase 1 chat may already include system turns inside `messages`.
- Domain “bubble” injection (if added later) = prepend a `system` turn when missing — **not implemented yet**.

---

## 5. Agentic vs non-agentic (chat pool context)

Within the **129M chat** slice:

| Class | Approx | Note |
|---|---|---|
| Non-agentic | ~92% | Plain chat — **do not** attach `tools[]` |
| Tool_Use folder without tools schema | ~4% | Tool-themed text |
| Real `tool_calls` / tools schema | ~1–3% | Keep tools only here |

Overall convertible **645M** is **~98.5% non-agentic**. Do not force Cursor/agent `tools[]` onto QA pairs.

---

## 6. Script & I/O

### Entry point

```bash
python pipeline/run_phases.py --phase 1|2|3 [options]
```

### Environment (K8s-friendly)

| Env | Meaning | Example |
|---|---|---|
| `SFT_SRC` | Parquet root | `/data/SFT/OUTPUT` |
| `SFT_OUT` | Output directory | `/data/out/phase2` |
| `SFT_DEDUP` | Dir containing `sft_normalize.py` | `/data/SFT/DEDUP_PIPELINE` |

Flags override env: `--src` `--out` `--dedup`.

### Important flags

| Flag | Meaning |
|---|---|
| `--phase 1 2 3` | Which pools (prefer **one phase per Job**) |
| `--limit N` | Sample N converted rows (**required today**; full dump not enabled) |
| `--seed` | RNG seed for sampling |
| `--with-review` | Also write JSONL **with** `_meta` for debugging |
| `--num-shards N` | Split **file list** across pods |
| `--shard-id i` | This pod’s shard in `[0, N)` |

### Outputs per run (training)

- `phase{P}_openai_sft_{limit}.jsonl` — **train** (no `_meta`)
- `phase{P}_openai_sft_{limit}.pretty.json` — readable
- `phase{P}_openai_sft_{limit}_preview.csv` — quick scan
- `phase{P}_openai_sft_{limit}_report.json` — counts / skip reasons
- optional `*_review.jsonl` — with `_meta` if `--with-review`

With sharding: filenames include `_shard{i}of{N}`.

### Train record shape

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Review `_meta` (not for training) may include: `source_file`, `domain`, `row_index`, `alias_pair`, `phase`.

---

## 7. Kubernetes layout (recommended)

### Principle

1. **Segregate by phase** → three Job groups in parallel.  
2. **Shard within phase** → many pods per group (Phase 2 gets the most).  
3. **Separate `--out` per phase** (and ideally per shard) to avoid clobbering.

### Suggested parallelism (starting point)

| Job | `--phase` | `--num-shards` | Notes |
|---|---|---|---|
| `sft-phase1` | `1` | 8–16 | ~129M chat |
| `sft-phase2` | `2` | 32–64 | ~476M pairs — largest |
| `sft-phase3` | `3` | 2–4 | ~15M aliases — small |

Ramp shards if storage IO allows; too many pods on one NFS can slow everyone.

### Pod command template

```bash
python pipeline/run_phases.py \
  --phase ${PHASE} \
  --src ${SFT_SRC} \
  --out ${SFT_OUT}/phase${PHASE} \
  --dedup ${SFT_DEDUP} \
  --num-shards ${NUM_SHARDS} \
  --shard-id ${JOB_COMPLETION_INDEX} \
  --limit 20 \
  --seed 42
```

For Indexed Jobs, set `JOB_COMPLETION_INDEX` = completion index (0 .. parallelism-1).

### What each pod needs

- Image with: Python 3, `pyarrow`, `pandas`
- Code: `pipeline/run_phases.py` (single file is enough)
- Volume mounts:
  - **Read:** `SFT_SRC` (OUTPUT parquet)
  - **Read:** `SFT_DEDUP` (`sft_normalize.py`)
  - **Write:** `SFT_OUT`
- **Do not** run full conversion on a login node; use compute/K8s only for large jobs

### Do **not** do this for production convert

```bash
# Avoid: one pod runs all phases back-to-back
python pipeline/run_phases.py --phase 1 2 3 ...
```

Use three Jobs instead so Phase 1 and 3 finish while Phase 2 is still running.

---

## 8. Current limitation (critical for K8s planning)

**`--limit N` = sample mode only.**  
Full dump (`--limit 0` / stream all rows per shard) is **not enabled yet**.

Today:

- `--num-shards` / `--shard-id` correctly **partition the file list**
- But each pod still only converts **`--limit` random rows** from its shard pool

For a real full convert, next engineering step:

1. Add streaming mode: for each file in the shard, read all (or chunked) rows → append JSONL  
2. Keep phase segregation + sharding as above  
3. Then launch the Job matrix for the full ~645M

Until then, use the script for **validation samples** and for **proving pod wiring** (env, mounts, shard indices).

---

## 9. Validation samples already produced (local)

Under `review_openai/phases/`:

| File | Meaning |
|---|---|
| `phase1_openai_sft_20.*` | Chat sample |
| `phase2_openai_sft_20.*` | Pair sample (includes some system turns) |
| `phase3_openai_sft_20.*` | Alias sample (`Q/A`, `query/positive`, …) |

Also earlier review packs:

- `review_openai/pair_validate_50/` — stratified pairs + incomplete + aliases  
- `review_openai/convertible_645m_stats.json` — domain/format census  
- `review_openai/chat_129m_stats.json` — chat domain census  

---

## 10. Domain mix (convertible ~645M, approximate)

| Domain | ~Share |
|---|---|
| Maths | ~61% |
| Biology_and_Genomics | ~7% |
| code | ~6% |
| Reasoning | ~5% |
| Finance | ~5% |
| sft_to_reasoning | ~4% |
| NL | ~3% |
| Other | rest |

Maths dominates; plan downsampling later if training mix should be balanced.

---

## 11. Checklist for Kube team

- [ ] Mount `OUTPUT` read-only as `SFT_SRC`  
- [ ] Mount `DEDUP_PIPELINE` (with `sft_normalize.py`) as `SFT_DEDUP`  
- [ ] Writable PVC/scratch as `SFT_OUT` (expect large JSONL when full dump exists)  
- [ ] Image: Python + pyarrow + pandas; copy `run_phases.py`  
- [ ] Three IndexedJobs: phase 1 / 2 / 3, different `NUM_SHARDS`  
- [ ] Pass `--shard-id` from Job completion index  
- [ ] Separate output subdirs per phase  
- [ ] Start with `--limit 20` smoke test per shard, then enable full stream when implemented  
- [ ] Do not attach `tools[]` to Phase 2/3 QA unless source has tool turns  
- [ ] Optional later: domain system-prompt injection for rows without system  

---

## 12. Quick commands

```bash
# Smoke: all three phases sequentially (dev only)
python pipeline/run_phases.py --phase 1 2 3 --limit 20 --seed 42 --with-review

# Prod-style: one phase, one shard (K8s pod)
python pipeline/run_phases.py --phase 2 \
  --src "$SFT_SRC" --out "$SFT_OUT/phase2" --dedup "$SFT_DEDUP" \
  --num-shards 32 --shard-id "$JOB_COMPLETION_INDEX" \
  --limit 20 --seed 42
```

---

## 13. Contact / code location

| Path | Role |
|---|---|
| `pipeline/run_phases.py` | **Only script needed** for phases 1–3 |
| `pipeline/to_openai_sft.py` | Older helper; optional / legacy |
| `review_openai/phases/` | Sample outputs + reports |

**Owner context:** SFT parquet → OpenAI SFT for training; K8s preferred over Slurm login/SSH for full volume.
