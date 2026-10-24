"""Turn a FinalAnswer into readable text grouped by label."""

from __future__ import annotations

from introspective_agent.agent.schemas import FinalAnswer, Label

HEADINGS = {
    Label.VERIFIED: "Verified (stated in a source)",
    Label.DERIVED: "Derived from sources (calculated or reasoned)",
    Label.INHERENT: "Inherent knowledge (well-established, no source retrieved)",
    Label.DERIVED_INHERENT: "Derived inherent knowledge (reasoned from memory; check before relying on it)",
    Label.ESTIMATED: "Estimates, assumptions and judgement",
    Label.UNSUPPORTED: "Unsupported (nothing backs this up; treat as possibly wrong)",
}
SHOW_RATIONALE = {Label.DERIVED, Label.DERIVED_INHERENT, Label.ESTIMATED}


def render(answer: FinalAnswer, show_changes: bool = True) -> str:
    if answer.blocked:
        return f"Blocked: {answer.block_reason}"
    lines = [answer.summary, ""]
    for label in Label:
        claims = [c for c in answer.claims if c.label == label]
        if not claims:
            continue
        lines.append(f"## {HEADINGS[label]}")
        for c in claims:
            cite = f" [{', '.join(c.evidence_ids)}]" if c.evidence_ids else ""
            lines.append(f"- {c.text}{cite}")
            if label in SHOW_RATIONALE and c.rationale:
                lines.append(f"    why: {c.rationale}")
            if show_changes and c.changes:
                for ch in c.changes:
                    lines.append(f"    ({ch.stage}: {ch.from_label.value} → {ch.to_label.value}: {ch.reason})")
        lines.append("")
    if answer.evidence_sources:
        lines.append("Sources: " + "; ".join(f"{k}={v}" for k, v in answer.evidence_sources.items()))
    return "\n".join(lines).rstrip()
