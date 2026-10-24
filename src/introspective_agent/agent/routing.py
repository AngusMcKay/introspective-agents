"""Conditional-edge functions, kept separate so the control flow is unit-testable."""

from __future__ import annotations

from introspective_agent.agent.state import AgentState
from introspective_agent.config import settings


def after_guard_input(state: AgentState) -> str:
    return "finalize" if state.get("blocked") else "plan"


def after_check_citations(state: AgentState) -> str:
    if state.get("draft_problems") and state.get("attempts", 0) < settings.max_draft_attempts:
        return "draft"
    return "self_critique"


def after_self_critique(state: AgentState) -> str:
    return "verify" if state["mode"] == "grounded" else "finalize"
