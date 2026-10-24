"""Deterministic claim verification.

The core idea: a model's own label is a *hypothesis*. In grounded mode a claim keeps
the label "verified" only if it cites evidence that exists AND every number in the claim
appears in that evidence. An LLM judge (in nodes.py) then checks the non-numeric part.

Numbers (torques, pressures, capacities, intervals) are the easiest thing to check
mechanically and the most damaging thing to get wrong, so they get a hard rule rather
than a model's opinion.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from introspective_agent.agent.schemas import Evidence

# Things that contain digits but aren't measurements. Removed before extracting numbers.
_DATE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")
_YEAR_RANGE = re.compile(r"\b(19|20)\d{2}\s*(?:to|-|–)\s*(19|20)\d{2}\b")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_EVIDENCE_ID = re.compile(r"\bE\d+\b")
_VISCOSITY = re.compile(r"\b\d{1,2}W-?\d{1,2}\b", re.I)              # 5W-30
_CODE = re.compile(r"\b[A-Za-z]+[A-Za-z0-9]*[-_]?\d[A-Za-z0-9-]*\b")  # F22, FX-28, GD-18, C2
_NUMBER = re.compile(r"(?<![\w.])-?£?\d[\d,]*(?:\.\d+)?%?")


def extract_numbers(text: str) -> list[float]:
    for pattern in (_DATE, _YEAR_RANGE, _YEAR, _EVIDENCE_ID, _VISCOSITY, _CODE):
        text = pattern.sub(" ", text)
    out = []
    for raw in _NUMBER.findall(text):
        cleaned = raw.replace("£", "").replace(",", "").rstrip("%")
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out


def _walk_numbers(obj: Any) -> Iterable[float]:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_numbers(v)
    elif isinstance(obj, str):
        yield from extract_numbers(obj)


def evidence_numbers(evidence: Iterable[Evidence]) -> set[float]:
    nums: set[float] = set()
    for e in evidence:
        nums.update(_walk_numbers(e.content))
    return nums


def number_supported(value: float, pool: set[float]) -> bool:
    """Match exactly, or as a reasonable rounding (to 1dp, or to a whole number for values >= 10)."""
    for p in pool:
        if abs(abs(p) - abs(value)) < 0.006:
            return True
        if value == round(value) and abs(p) >= 10 and abs(round(abs(p)) - abs(value)) < 0.5:
            return True
        if abs(p) >= 1 and abs(round(abs(p), 1) - abs(value)) < 0.051:
            return True
    return False


def check_numbers(claim_text: str, cited: list[Evidence]) -> tuple[bool, list[float]]:
    """Return (all numbers supported, unsupported numbers)."""
    pool = evidence_numbers(cited)
    missing = [n for n in extract_numbers(claim_text) if not number_supported(n, pool)]
    return (not missing, missing)
