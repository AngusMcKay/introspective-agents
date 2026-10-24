"""The LangGraph workflow.

    START
      │
    guard_input ──(blocked)──────────────────────────────┐
      │                                                  │
    plan → gather_evidence → draft                       │
                               │                         │
             (grounded) check_citations ─(problems & retries left)─→ draft
                               │                         │
                         self_critique                   │
                               │                         │
              (grounded) verify │ (self_label) skip      │
                               ▼                         ▼
                            finalize ─────────────────→ END

Two modes, so you can test whether self-assessment alone is reliable:
  * self_label: the model labels its claims, then questions itself (self_critique).
  * grounded:   as above, plus mechanical citation/number checks, an evidence judge,
                and a sceptical knowledge judge for claims made from memory.
Run evals/run_evals.py to compare them on the same questions.
"""

from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable

from langgraph.graph import END, START, StateGraph

from introspective_agent.agent.llm import LLM
from introspective_agent.agent.nodes import Nodes
from introspective_agent.agent.routing import after_check_citations, after_guard_input, after_self_critique
from introspective_agent.agent.schemas import FinalAnswer, Label, Mode
from introspective_agent.agent.state import AgentState
from introspective_agent.agent.tools import MCPTools
from introspective_agent.agent.tracing import Tracer, current_tracer, use_tracer

NodeFn = Callable[[AgentState], Awaitable[dict[str, Any]]]


def _traced(name: str, fn: NodeFn) -> NodeFn:
    @functools.wraps(fn)
    async def wrapper(state: AgentState) -> dict[str, Any]:
        with current_tracer().span(f"node:{name}"):
            return await fn(state)

    return wrapper


def build_graph(llm: LLM, tools: MCPTools):
    n = Nodes(llm, tools)
    g = StateGraph(AgentState)
    for name in ["guard_input", "plan", "gather_evidence", "draft", "check_citations",
                 "self_critique", "verify", "finalize"]:
        g.add_node(name, _traced(name, getattr(n, name)))

    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", after_guard_input, {"plan": "plan", "finalize": "finalize"})
    g.add_edge("plan", "gather_evidence")
    g.add_edge("gather_evidence", "draft")
    g.add_conditional_edges("draft", lambda s: "check_citations" if s["mode"] == "grounded" else "self_critique",
                            {"check_citations": "check_citations", "self_critique": "self_critique"})
    g.add_conditional_edges("check_citations", after_check_citations, {"draft": "draft", "self_critique": "self_critique"})
    g.add_conditional_edges("self_critique", after_self_critique, {"verify": "verify", "finalize": "finalize"})
    g.add_edge("verify", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


async def run_graph(
    question: str,
    *,
    llm: LLM,
    tools: MCPTools,
    mode: Mode = "grounded",
    tracer: Tracer | None = None,
) -> AgentState:
    """Run the graph and return the full final state (evals need the evidence too)."""
    tracer = tracer or Tracer()
    app = build_graph(llm, tools)
    with use_tracer(tracer):
        tracer.event("run_start", mode=mode)
        state = await app.ainvoke({"question": question, "mode": mode, "guardrail_events": [], "attempts": 0})
        final: FinalAnswer = state["final"]
        tracer.event("run_end", blocked=final.blocked,
                     labels={lbl.value: sum(c.label == lbl for c in final.claims) for lbl in Label})
    return state


async def answer(question: str, **kwargs: Any) -> FinalAnswer:
    """Convenience entry point used by the API and CLI."""
    return (await run_graph(question, **kwargs))["final"]
