#!/usr/bin/env python3
"""
Redact PII / secrets from OpenAI-format SFT JSONL (agent traces).

Default strategy (code-friendly, low false positives):
  - High-precision Presidio entities: EMAIL, PHONE, CREDIT_CARD, US_SSN, IBAN
  - Regex: API keys, private keys, Bearer tokens, AWS keys, GitHub PATs
  - Path anonymization: /home/<user> and /Users/<user> -> stable placeholders
  - Skips noisy Presidio types (PERSON, LOCATION, URL, DATE_TIME) that wreck code

Optional --use-llm: call an OpenAI-compatible hosted model for residual PII
in natural-language spans (needs OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL).

Consistent replacements: same raw value -> same placeholder within a run
(and optionally via --salt for stable cross-run hashes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


# --- high-precision secret / credential patterns ---
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AWS_ACCESS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GITHUB_TOKEN", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b")),
    ("OPENAI_KEY", re.compile(r"\bsk-[A-Za-z0-9\-_]{20,}\b")),
    ("BEARER_TOKEN", re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]+=*", re.I)),
    ("PRIVATE_KEY", re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    )),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
]

HOME_PATH_RE = re.compile(r"(?P<prefix>/(?:home|Users)/)(?P<user>[A-Za-z0-9._-]+)")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Presidio entities that are reliable enough for code/agent traces
# Keep this list tight — driver-license / passport / medical fire often on code IDs.
SAFE_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "IBAN_CODE",
    "IP_ADDRESS",
]

# Cheap gate: skip spaCy/Presidio when text has no plausible PII markers.
_PII_GATE_RE = re.compile(
    r"@|"                                  # email
    r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b|"  # ssn-like
    r"\b(?:\d[ -]*?){13,19}\b|"            # card-like digit runs
    r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|"  # phone-like
    r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"    # iban-ish
)

# Metadata keys we keep structurally but can hash
ID_KEYS = {"session_id", "session_name", "trace_ids"}


def _stable_token(kind: str, value: str, salt: str) -> str:
    h = hashlib.sha256(f"{salt}|{kind}|{value}".encode("utf-8")).hexdigest()[:8]
    return f"<{kind}_{h}>"


class PiiRedactor:
    def __init__(
        self,
        *,
        salt: str = "traces_team",
        score_threshold: float = 0.6,
        redact_ip: bool = False,
        hash_ids: bool = True,
        use_presidio: bool = False,
    ) -> None:
        self.salt = salt
        self.score_threshold = score_threshold
        self.redact_ip = redact_ip
        self.hash_ids = hash_ids
        self.use_presidio = use_presidio
        self.cache: dict[tuple[str, str], str] = {}
        self.stats: Counter[str] = Counter()
        self.analyzer = None
        self.anonymizer = None
        self.entities: list[str] = []

        if use_presidio:
            entities = list(SAFE_ENTITIES)
            if not redact_ip:
                entities = [e for e in entities if e != "IP_ADDRESS"]
            try:
                from presidio_analyzer.nlp_engine import NlpEngineProvider

                provider = NlpEngineProvider(
                    nlp_configuration={
                        "nlp_engine_name": "spacy",
                        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
                    }
                )
                nlp_engine = provider.create_engine()
                self.analyzer = AnalyzerEngine(
                    nlp_engine=nlp_engine, supported_languages=["en"]
                )
            except Exception:
                self.analyzer = AnalyzerEngine()
            self.analyzer.registry.add_recognizer(
                PatternRecognizer(
                    supported_entity="EMAIL_ADDRESS",
                    patterns=[Pattern("email", EMAIL_RE.pattern, 0.9)],
                )
            )
            self.anonymizer = AnonymizerEngine()
            self.entities = entities
            self.stats["presidio_calls"] = 0
            self.stats["presidio_skipped_gate"] = 0

    def _token(self, kind: str, value: str) -> str:
        key = (kind, value)
        if key not in self.cache:
            self.cache[key] = _stable_token(kind, value, self.salt)
        self.stats[kind] += 1
        return self.cache[key]

    def _apply_regex_secrets(self, text: str) -> str:
        for kind, pat in SECRET_PATTERNS:
            def repl(m: re.Match[str], _kind: str = kind) -> str:
                return self._token(_kind, m.group(0))

            text, n = pat.subn(repl, text)
            if n:
                self.stats[f"regex_{kind}"] += n
        return text

    def _apply_home_paths(self, text: str) -> str:
        def repl(m: re.Match[str]) -> str:
            user = m.group("user")
            # keep common non-personal dirs
            if user in {"ubuntu", "root", "nobody", "www-data", "nfs", "shared"}:
                return m.group(0)
            tok = self._token("USER", user)
            # <USER_abcd1234> -> use short form in path
            short = tok.strip("<>").split("_", 1)[-1]
            return f"{m.group('prefix')}user_{short}"

        return HOME_PATH_RE.sub(repl, text)

    def _presidio_redact(self, text: str) -> str:
        if not text or not text.strip():
            return text
        if not _PII_GATE_RE.search(text):
            self.stats["presidio_skipped_gate"] += 1
            return text
        # Cap analyzer cost on huge tool dumps
        max_chunk = 40_000
        if len(text) <= max_chunk:
            return self._presidio_redact_chunk(text)

        parts: list[str] = []
        for i in range(0, len(text), max_chunk):
            chunk = text[i : i + max_chunk]
            if _PII_GATE_RE.search(chunk):
                parts.append(self._presidio_redact_chunk(chunk))
            else:
                self.stats["presidio_skipped_gate"] += 1
                parts.append(chunk)
        return "".join(parts)

    def _presidio_redact_chunk(self, text: str) -> str:
        self.stats["presidio_calls"] += 1
        results = self.analyzer.analyze(
            text=text,
            language="en",
            entities=self.entities,
            score_threshold=self.score_threshold,
        )
        if not results:
            return text

        # Stable replace by entity type + original span
        operators: dict[str, OperatorConfig] = {}
        # Build custom replacements per result via sorting reverse & manual splice
        # (Presidio default uses <ENTITY_TYPE> which collapses distinct values)
        spans = sorted(results, key=lambda r: (r.start, -r.end))
        # drop overlaps (keep highest score / longer)
        kept: list[RecognizerResult] = []
        last_end = -1
        for r in sorted(spans, key=lambda r: (r.start, -(r.end - r.start), -r.score)):
            if r.start < last_end:
                continue
            kept.append(r)
            last_end = r.end

        out: list[str] = []
        cursor = 0
        for r in sorted(kept, key=lambda r: r.start):
            if r.start < cursor:
                continue
            out.append(text[cursor : r.start])
            span = text[r.start : r.end]
            out.append(self._token(r.entity_type, span))
            cursor = r.end
            self.stats[f"presidio_{r.entity_type}"] += 1
        out.append(text[cursor:])
        return "".join(out)

    def redact_text(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        # Fast email pass (no spaCy) — catches most real emails cheaply
        def email_repl(m: re.Match[str]) -> str:
            return self._token("EMAIL_ADDRESS", m.group(0))

        text, n_email = EMAIL_RE.subn(email_repl, text)
        if n_email:
            self.stats["regex_EMAIL"] += n_email
        text = self._apply_regex_secrets(text)
        text = self._apply_home_paths(text)
        if self.use_presidio:
            text = self._presidio_redact(text)
        return text

    def redact_obj(self, obj: Any, *, key: str | None = None) -> Any:
        if isinstance(obj, str):
            if self.hash_ids and key in {"session_id", "session_name"} and obj:
                return self._token("SESSION", obj)
            return self.redact_text(obj)
        if isinstance(obj, list):
            if key == "trace_ids" and self.hash_ids:
                return [self._token("TRACE", str(x)) if x is not None else x for x in obj]
            return [self.redact_obj(x) for x in obj]
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                out[k] = self.redact_obj(v, key=k)
            return out
        return obj


def redact_file(
    inp: Path,
    out: Path,
    *,
    report: Path | None,
    limit: int | None,
    salt: str,
    score_threshold: float,
    redact_ip: bool,
    hash_ids: bool,
    use_presidio: bool,
) -> None:
    redactor = PiiRedactor(
        salt=salt,
        score_threshold=score_threshold,
        redact_ip=redact_ip,
        hash_ids=hash_ids,
        use_presidio=use_presidio,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    t0 = time.time()
    with inp.open(encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            clean = redactor.redact_obj(row)
            fout.write(json.dumps(clean, ensure_ascii=False) + "\n")
            n += 1
            if n % 100 == 0:
                elapsed = time.time() - t0
                rate = n / elapsed if elapsed else 0
                print(
                    f"  [{n}] {rate:.2f} rows/s  cache={len(redactor.cache)}  "
                    f"hits={sum(redactor.stats.values())}",
                    flush=True,
                )
            if limit is not None and n >= limit:
                break

    elapsed = time.time() - t0
    summary = {
        "input": str(inp),
        "output": str(out),
        "rows": n,
        "seconds": round(elapsed, 2),
        "rows_per_sec": round(n / elapsed, 3) if elapsed else None,
        "unique_replacements": len(redactor.cache),
        "hit_counts": dict(redactor.stats),
        "score_threshold": score_threshold,
        "redact_ip": redact_ip,
        "hash_ids": hash_ids,
    }
    print(json.dumps(summary, indent=2))
    if report:
        report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"report -> {report}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None, help="Process only first N rows (smoke)")
    p.add_argument("--salt", default="traces_team")
    p.add_argument("--score-threshold", type=float, default=0.6)
    p.add_argument("--redact-ip", action="store_true")
    p.add_argument(
        "--presidio",
        action="store_true",
        help="Also run Microsoft Presidio (slower; better phone/SSN/CC recall)",
    )
    p.add_argument("--no-hash-ids", action="store_true", help="Keep session_id / trace_ids as-is")
    args = p.parse_args()

    redact_file(
        args.input,
        args.output,
        report=args.report,
        limit=args.limit,
        salt=args.salt,
        score_threshold=args.score_threshold,
        redact_ip=args.redact_ip,
        hash_ids=not args.no_hash_ids,
        use_presidio=args.presidio,
    )


if __name__ == "__main__":
    main()
