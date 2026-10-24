"""Input and output guardrails.

These are deliberately simple, deterministic checks. They are cheap, fast and
testable, which is why production systems usually layer them *underneath* model-based
classifiers (e.g. Bedrock Guardrails, Llama Guard, NeMo Guardrails) rather than relying
on a model alone. Good exercises:
  * add a safety classifier: questions about gas, mains electrics or structural work get
    a mandatory "use a qualified professional" claim and stricter verification
  * replace the injection regexes with a classifier and compare on the eval set
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Order matters: longer / more specific patterns first.
PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("CARD_NUMBER", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("IBAN", re.compile(r"\bGB\d{2}[A-Z]{4}\d{14}\b", re.I)),
    ("SORT_CODE", re.compile(r"\b\d{2}-\d{2}-\d{2}\b")),
    ("ACCOUNT_NUMBER", re.compile(r"\b\d{8}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("UK_PHONE", re.compile(r"(?:\+44\s?7\d{3}|\b07\d{3})\s?\d{3}\s?\d{3}\b")),
]

INJECTION_PATTERNS = [
    re.compile(p, re.I)
    for p in [
        r"ignore (all |any )?(previous|prior|above|earlier) (instructions|rules|prompts)",
        r"disregard (your|the) (instructions|rules|system prompt)",
        r"(reveal|print|show|repeat) (your|the) (system prompt|instructions|hidden prompt)",
        r"you are now (a|an|in) ",
        r"developer mode|jailbreak|\bDAN\b",
        r"\.\./|/etc/passwd|~/\.ssh",  # path traversal attempts aimed at the document tools
    ]
]


@dataclass
class GuardResult:
    text: str
    blocked: bool = False
    reason: str | None = None
    events: list[str] = field(default_factory=list)


def redact_pii(text: str) -> tuple[str, list[str]]:
    events = []
    for name, pattern in PII_PATTERNS:
        text, n = pattern.subn(f"[REDACTED_{name}]", text)
        if n:
            events.append(f"redacted {n} x {name}")
    return text, events


def check_input(question: str, max_chars: int) -> GuardResult:
    if not question.strip():
        return GuardResult(question, blocked=True, reason="empty question", events=["input:blocked:empty"])
    if len(question) > max_chars:
        return GuardResult(question, blocked=True, reason=f"question longer than {max_chars} characters",
                           events=["input:blocked:too_long"])
    for p in INJECTION_PATTERNS:
        if p.search(question):
            return GuardResult(question, blocked=True, reason="possible prompt injection or attempt to access other data",
                               events=[f"input:blocked:injection:{p.pattern[:40]}"])
    text, events = redact_pii(question)
    return GuardResult(text, events=[f"input:{e}" for e in events])


def check_output_text(text: str) -> tuple[str, list[str]]:
    """Last line of defence: never return anything that looks like PII."""
    text, events = redact_pii(text)
    return text, [f"output:{e}" for e in events]
