"""Data contracts between graph nodes and with the LLM.

Every LLM call returns JSON validated against one of these Pydantic models. If the
model returns something that doesn't validate, the call is retried with the
validation error (see llm.py). That is the most basic, and most useful, guardrail.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Label(str, Enum):
    """How a claim is supported, roughly strongest to weakest.

    The first two rest on retrieved evidence that can be checked. The next two rest on
    the model's own trained knowledge, which can't be checked against a source here.
    """

    VERIFIED = "verified"                      # stated in retrieved evidence, cited
    DERIVED = "derived"                        # calculated or reasoned from evidence; steps shown
    INHERENT = "inherent_knowledge"            # strong, well-established fact from trained knowledge
    DERIVED_INHERENT = "derived_inherent"      # reasoned from trained knowledge; not a strong fact
    ESTIMATED = "estimated"                    # judgement, assumption or intuition
    UNSUPPORTED = "unsupported"                # nothing backs it; may be wrong or invented


EVIDENCE_LABELS = (Label.VERIFIED, Label.DERIVED)
KNOWLEDGE_LABELS = (Label.INHERENT, Label.DERIVED_INHERENT)

Mode = Literal["self_label", "grounded"]


# ---- LLM outputs --------------------------------------------------------------------
class ToolCall(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    why: str = ""


class Plan(BaseModel):
    reasoning: str = Field(description="Brief: what information is needed to answer")
    tool_calls: list[ToolCall] = Field(default_factory=list)


class Claim(BaseModel):
    text: str = Field(description="One self-contained statement")
    label: Label
    evidence_ids: list[str] = Field(default_factory=list, description="IDs like E1, E2 that support the claim")
    rationale: str = Field(default="", description="Why this label; for derived labels, the calculation or reasoning")


class Draft(BaseModel):
    summary: str = Field(description="Two or three sentence direct answer to the question")
    claims: list[Claim]


class Challenge(BaseModel):
    claim_index: int
    issue: str
    suggested_label: Label


class Critique(BaseModel):
    """The agent questioning itself."""

    challenges: list[Challenge] = Field(default_factory=list)


class EvidenceVerdict(BaseModel):
    claim_index: int
    verdict: Literal["entailed", "follows", "not_supported"]
    reason: str


class EvidenceJudgement(BaseModel):
    verdicts: list[EvidenceVerdict]


class KnowledgeVerdict(BaseModel):
    claim_index: int
    verdict: Literal["well_established", "plausible", "doubtful"]
    reason: str


class KnowledgeJudgement(BaseModel):
    verdicts: list[KnowledgeVerdict]


# ---- Internal state -----------------------------------------------------------------
class Evidence(BaseModel):
    id: str
    tool: str
    arguments: dict[str, Any]
    source: str
    content: Any
    is_error: bool = False


class LabelChange(BaseModel):
    stage: Literal["self_critique", "verification"]
    from_label: Label
    to_label: Label
    reason: str


class FinalClaim(BaseModel):
    text: str
    label: Label
    original_label: Label
    evidence_ids: list[str]
    rationale: str
    changes: list[LabelChange] = Field(default_factory=list)


class FinalAnswer(BaseModel):
    question: str
    mode: Mode
    summary: str
    claims: list[FinalClaim]
    blocked: bool = False
    block_reason: str | None = None
    guardrail_events: list[str] = Field(default_factory=list)
    evidence_sources: dict[str, str] = Field(default_factory=dict)
    trace_id: str | None = None
