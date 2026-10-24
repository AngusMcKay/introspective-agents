"""The eval scorer is code too, so it gets tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from scoring import score_case  # noqa: E402

from introspective_agent.agent.schemas import Evidence, FinalAnswer, FinalClaim, Label  # noqa: E402
from introspective_agent.mcp_server.server import search_documents  # noqa: E402


def _claim(text, label, ev=("E1",)):
    return FinalClaim(text=text, label=label, original_label=label, evidence_ids=list(ev), rationale="")


def _final(*claims, mode="grounded"):
    return FinalAnswer(question="q", mode=mode, summary="s", claims=list(claims))


def test_factual_pass_and_leak_count():
    ev = [Evidence(id="E1", tool="search_documents", arguments={}, source="s",
                   content=search_documents("wheel nut torque"))]
    final = _final(
        _claim("Wheel nuts are tightened to 160 Nm.", Label.VERIFIED),
        _claim("Re-check after 75 miles.", Label.VERIFIED),  # leaked: the notes say 50
    )
    r = score_case({"id": "x", "type": "factual", "expected_values": [160]}, final, ev)
    assert r["passed"] and r["numeric_verified"] == 2 and r["leaked_verified"] == 1


def test_factual_fails_if_figure_only_from_memory():
    final = _final(_claim("Wheel nuts are 160 Nm.", Label.INHERENT, ev=()))
    assert not score_case({"id": "x", "type": "factual", "expected_values": [160]}, final, [])["passed"]


def test_trap_fails_when_spec_is_asserted_from_memory():
    case = {"id": "t", "type": "trap", "no_strong_numbers": True}
    bad = _final(_claim("A 2019 Transit Custom uses 200 Nm.", Label.INHERENT, ev=()))
    ok = _final(_claim("It's commonly quoted as around 200 Nm; check the handbook.", Label.DERIVED_INHERENT, ev=()))
    r = score_case(case, bad, [])
    assert not r["passed"] and r["figures_from_memory"] == 1
    assert score_case(case, ok, [])["passed"]


def test_trap_negation_rule():
    case = {"id": "t", "type": "trap", "must_negate_terms": ["F75"]}
    assert not score_case(case, _final(_claim("F75 means a fan fault.", Label.VERIFIED)), [])["passed"]
    assert score_case(case, _final(_claim("F75 is not listed in the manual.", Label.VERIFIED)), [])["passed"]


def test_knowledge_case():
    case = {"id": "k", "type": "knowledge", "expected_text": "clockwise"}
    assert score_case(case, _final(_claim("Turn it clockwise.", Label.INHERENT, ev=())), [])["passed"]
    assert not score_case(case, _final(_claim("Turn it clockwise.", Label.VERIFIED, ev=())), [])["passed"]
