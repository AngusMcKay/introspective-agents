"""The graph's shared state.

LangGraph merges each node's returned dict into this state. Most keys are overwritten;
`guardrail_events` uses a reducer (operator.add) so every node's events accumulate.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from introspective_agent.agent.schemas import Critique, Draft, Evidence, FinalAnswer, FinalClaim, Mode, Plan


class AgentState(TypedDict, total=False):
    # inputs
    question: str
    mode: Mode
    # guardrails
    sanitised_question: str
    blocked: bool
    block_reason: str | None
    guardrail_events: Annotated[list[str], operator.add]
    # work
    plan: Plan
    evidence: list[Evidence]
    draft: Draft
    draft_problems: list[str]
    attempts: int
    critique: Critique
    claims: list[FinalClaim]
    # output
    final: FinalAnswer
