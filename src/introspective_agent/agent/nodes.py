"""Graph nodes. Each node is an async function: state in, partial state update out.

Nodes are methods on `Nodes` so they share the LLM and tool clients, and so tests can
call any node directly without building the graph.
"""

from __future__ import annotations

import json
from typing import Any

from introspective_agent.agent import prompts
from introspective_agent.agent.guardrails import check_input, check_output_text
from introspective_agent.agent.llm import LLM
from introspective_agent.agent.schemas import (
    EVIDENCE_LABELS, KNOWLEDGE_LABELS, Critique, Draft, Evidence, EvidenceJudgement, FinalAnswer, FinalClaim,
    KnowledgeJudgement, Label, LabelChange, Plan,
)
from introspective_agent.agent.state import AgentState
from introspective_agent.agent.tools import MCPTools
from introspective_agent.agent.tracing import current_tracer
from introspective_agent.agent.verification import check_numbers
from introspective_agent.config import settings


def _evidence_block(evidence: list[Evidence]) -> str:
    if not evidence:
        return "(no evidence was retrieved)"
    return "\n\n".join(
        f"[{e.id}] source={e.source}{' (TOOL ERROR)' if e.is_error else ''}\n"
        f"<evidence>\n{json.dumps(e.content, default=str)}\n</evidence>"
        for e in evidence
    )


def _claims_block(claims: list[Any]) -> str:
    return "\n".join(
        f"{i}. [{c.label.value}] {c.text}  (evidence: {', '.join(c.evidence_ids) or 'none'}; rationale: {c.rationale})"
        for i, c in enumerate(claims)
    )


class Nodes:
    def __init__(self, llm: LLM, tools: MCPTools):
        self.llm = llm
        self.tools = tools

    # -- 1. Input guardrail --------------------------------------------------------------
    async def guard_input(self, state: AgentState) -> dict[str, Any]:
        result = check_input(state["question"], settings.max_question_chars)
        return {
            "sanitised_question": result.text,
            "blocked": result.blocked,
            "block_reason": result.reason,
            "guardrail_events": result.events,
        }

    # -- 2. Plan: which tools to call ------------------------------------------------------
    async def plan(self, state: AgentState) -> dict[str, Any]:
        tools = await self.tools.list_tools()
        plan = await self.llm.complete_json(
            task="plan",
            system=prompts.PLAN.format(max_calls=settings.max_tool_calls),
            user=f"Available tools:\n{json.dumps(tools, indent=1)}\n\nQuestion: {state['sanitised_question']}",
            schema=Plan,
        )
        return {"plan": plan}

    # -- 3. Gather evidence via MCP -------------------------------------------------------
    async def gather_evidence(self, state: AgentState) -> dict[str, Any]:
        """Execute the plan. Only tools the server actually offers are called.

        In a multi-user system this is also where access control belongs: e.g. injecting
        the authenticated user's ID so the model can never choose whose data is read.
        """
        allowed = {t["name"] for t in await self.tools.list_tools()}
        events: list[str] = []
        evidence: list[Evidence] = []
        for call in state["plan"].tool_calls[: settings.max_tool_calls]:
            if call.tool not in allowed:
                events.append(f"tool:dropped_unknown:{call.tool}")
                continue
            content, is_error = await self.tools.call(call.tool, dict(call.arguments))
            if is_error:
                events.append(f"tool:error:{call.tool}")
            evidence.append(Evidence(
                id=f"E{len(evidence) + 1}",
                tool=call.tool,
                arguments=dict(call.arguments),
                source=content.get("source", call.tool) if isinstance(content, dict) else call.tool,
                content=content,
                is_error=is_error,
            ))
        return {"evidence": evidence, "guardrail_events": events}

    # -- 4. Draft an answer as labelled claims ------------------------------------------
    async def draft(self, state: AgentState) -> dict[str, Any]:
        system = prompts.DRAFT
        problems = state.get("draft_problems") or []
        if problems:
            system += "\n\n" + prompts.DRAFT_RETRY.format(
                problems="\n".join(f"- {p}" for p in problems),
                ids=", ".join(e.id for e in state.get("evidence", [])) or "none",
            )
        draft = await self.llm.complete_json(
            task="draft",
            system=system,
            user=f"Question: {state['sanitised_question']}\n\nEvidence:\n{_evidence_block(state.get('evidence', []))}",
            schema=Draft,
        )
        return {"draft": draft, "attempts": state.get("attempts", 0) + 1}

    # -- 4b. Mechanical check on the draft (grounded mode only) --------------------------
    async def check_citations(self, state: AgentState) -> dict[str, Any]:
        valid = {e.id for e in state.get("evidence", []) if not e.is_error}
        problems = []
        if not state["draft"].claims:
            problems.append("The draft contained no claims.")
        for i, c in enumerate(state["draft"].claims):
            bad = [x for x in c.evidence_ids if x not in valid]
            if bad:
                problems.append(f"Claim {i} cites evidence that does not exist: {bad}")
            if c.label in EVIDENCE_LABELS and not c.evidence_ids:
                problems.append(f"Claim {i} is labelled {c.label.value} but cites no evidence. "
                                f"If it comes from your own knowledge, use inherent_knowledge or derived_inherent.")
        events = [f"draft:citation_problems:{len(problems)}"] if problems else []
        return {"draft_problems": problems, "guardrail_events": events}

    # -- 5. Self-critique: the agent questions itself ------------------------------------
    async def self_critique(self, state: AgentState) -> dict[str, Any]:
        draft = state["draft"]
        critique = await self.llm.complete_json(
            task="critique",
            system=prompts.CRITIQUE,
            user=(
                f"Question: {state['sanitised_question']}\n\nEvidence:\n{_evidence_block(state.get('evidence', []))}"
                f"\n\nDraft claims:\n{_claims_block(draft.claims)}"
            ),
            schema=Critique,
        )
        claims = [
            FinalClaim(text=c.text, label=c.label, original_label=c.label,
                       evidence_ids=list(c.evidence_ids), rationale=c.rationale)
            for c in draft.claims
        ]
        for ch in critique.challenges:
            if 0 <= ch.claim_index < len(claims) and ch.suggested_label != claims[ch.claim_index].label:
                fc = claims[ch.claim_index]
                fc.changes.append(LabelChange(stage="self_critique", from_label=fc.label,
                                              to_label=ch.suggested_label, reason=ch.issue))
                fc.label = ch.suggested_label
        return {"critique": critique, "claims": claims}

    # -- 6. Verify (grounded mode only) --------------------------------------------------
    async def verify(self, state: AgentState) -> dict[str, Any]:
        """Check each claim by the standard its label demands.

          verified / derived        -> deterministic checks, then an evidence judge
          inherent / derived_inherent -> a sceptical knowledge judge (ideally a different model)
          estimated / unsupported   -> left alone; they already claim nothing

        This is the place to add a strictness setting: e.g. at higher strictness, any
        inherent_knowledge claim containing a specific figure drops to derived_inherent.
        """
        evidence = {e.id: e for e in state.get("evidence", []) if not e.is_error}
        claims = [c.model_copy(deep=True) for c in state["claims"]]
        tracer = current_tracer()

        def change(c: FinalClaim, to: Label, reason: str) -> None:
            if c.label != to:
                c.changes.append(LabelChange(stage="verification", from_label=c.label, to_label=to, reason=reason))
                tracer.event("label_change", stage="verification", from_label=c.label.value,
                             to_label=to.value, reason=reason)
                c.label = to

        # 6a. Deterministic checks on evidence-based claims.
        for c in claims:
            if c.label not in EVIDENCE_LABELS:
                continue
            cited = [evidence[i] for i in c.evidence_ids if i in evidence]
            if not cited:
                change(c, Label.UNSUPPORTED, "cites no valid evidence")
                continue
            if c.label == Label.VERIFIED:
                ok, missing = check_numbers(c.text, cited)
                if not ok:
                    change(c, Label.UNSUPPORTED, f"numbers not found in cited evidence: {missing}")

        # 6b. Evidence judge for whatever still claims evidential support.
        to_judge = [i for i, c in enumerate(claims) if c.label in EVIDENCE_LABELS]
        if to_judge:
            lines = []
            for j, i in enumerate(to_judge):
                c = claims[i]
                ev = "\n".join(f"  [{x}] {json.dumps(evidence[x].content, default=str)}"
                               for x in c.evidence_ids if x in evidence)
                lines.append(f"Claim {j}: {c.text}\n  Rationale: {c.rationale}\n  Cited evidence:\n{ev}")
            result = await self.llm.complete_json(
                task="judge_evidence", system=prompts.EVIDENCE_JUDGE, user="\n\n".join(lines),
                schema=EvidenceJudgement, model=settings.judge_model,
            )
            for v in result.verdicts:
                if not 0 <= v.claim_index < len(to_judge):
                    continue
                c = claims[to_judge[v.claim_index]]
                if v.verdict == "not_supported":
                    change(c, Label.UNSUPPORTED, f"evidence judge: {v.reason}")
                elif v.verdict == "follows" and c.label == Label.VERIFIED:
                    change(c, Label.DERIVED, f"evidence judge: follows from evidence but not stated: {v.reason}")

        # 6c. Knowledge judge for claims made from memory.
        to_judge = [i for i, c in enumerate(claims) if c.label in KNOWLEDGE_LABELS]
        if to_judge:
            lines = [f"Claim {j}: [{claims[i].label.value}] {claims[i].text}\n  Rationale: {claims[i].rationale}"
                     for j, i in enumerate(to_judge)]
            result = await self.llm.complete_json(
                task="judge_knowledge", system=prompts.KNOWLEDGE_JUDGE,
                user=f"Question being answered: {state['sanitised_question']}\n\n" + "\n\n".join(lines),
                schema=KnowledgeJudgement, model=settings.judge_model,
            )
            for v in result.verdicts:
                if not 0 <= v.claim_index < len(to_judge):
                    continue
                c = claims[to_judge[v.claim_index]]
                if v.verdict == "doubtful":
                    change(c, Label.UNSUPPORTED, f"knowledge judge: {v.reason}")
                elif v.verdict == "plausible" and c.label == Label.INHERENT:
                    change(c, Label.DERIVED_INHERENT, f"knowledge judge: plausible, not established: {v.reason}")
        return {"claims": claims}

    # -- 7. Output guardrail and final answer -----------------------------------------
    async def finalize(self, state: AgentState) -> dict[str, Any]:
        tracer = current_tracer()
        if state.get("blocked"):
            final = FinalAnswer(
                question=state["question"], mode=state["mode"],
                summary="I can't help with that request.", claims=[], blocked=True,
                block_reason=state.get("block_reason"), guardrail_events=state.get("guardrail_events", []),
                trace_id=tracer.trace_id,
            )
            return {"final": final}

        events: list[str] = []
        summary, ev = check_output_text(state["draft"].summary)
        events += ev
        claims = []
        for c in state.get("claims", []):
            text, ev = check_output_text(c.text)
            events += ev
            claims.append(c.model_copy(update={"text": text}))
        final = FinalAnswer(
            question=state["question"], mode=state["mode"], summary=summary, claims=claims,
            guardrail_events=state.get("guardrail_events", []) + events,
            evidence_sources={e.id: e.source for e in state.get("evidence", [])},
            trace_id=tracer.trace_id,
        )
        return {"final": final, "guardrail_events": events}
