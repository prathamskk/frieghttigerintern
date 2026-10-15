import pytest
import pandas as pd
from src.auditor import AnomalyAuditor


@pytest.mark.parametrize("route,week_of,expected_note_ids,case_description", [
    ("Chennai-Bangalore", "2025-02-24", ["N001"], "Week 1 of flood (Feb 24 - Mar 2): active INCREASE note overlaps"),
    ("Chennai-Bangalore", "2025-03-03", ["N001"], "Week 2 of flood (Mar 3 - Mar 9): active INCREASE note overlaps"),
    ("Chennai-Bangalore", "2025-03-10", [], "Week 3 after flood repairs: outside disruption window"),
    ("Chennai-Bangalore", "2025-03-17", [], "N009 matches route and date, but discarded because cost_direction == NEUTRAL"),
    ("Ahmedabad-Mumbai", "2025-01-20", ["N002"], "Festival week: matches active INCREASE note"),
    ("Delhi-Jaipur", "2025-02-24", [], "Geographic isolation: Delhi-Jaipur does not receive Chennai or Ahmedabad notes"),
    ("Delhi-Jaipur", "2025-05-12", ["N003"], "Nationwide diesel price hike: All Routes wildcard applies"),
    ("Mumbai-Delhi", "2024-07-29", [], "N005 matches route and date, but discarded because cost_direction == NEUTRAL"),
])
def test_prefilter_candidate_rules(sample_extracted_notes, fake_engine, route, week_of, expected_note_ids, case_description):
    auditor = AnomalyAuditor(extracted_notes=sample_extracted_notes, engine=fake_engine)
    candidates = auditor.prefilter_candidates(route, week_of)
    actual_ids = [c["note_id"] for c in candidates]
    assert actual_ids == expected_note_ids, f"Failed on case: {case_description}"


def test_zero_waste_fallback_no_llm_call(sample_extracted_notes, fake_engine):
    auditor = AnomalyAuditor(extracted_notes=sample_extracted_notes, engine=fake_engine)

    # Delhi-Jaipur has 0 candidates on this date
    res = auditor.audit_spike(
        route="Delhi-Jaipur",
        week_of="2024-11-11",
        cost_per_tonne_km=4.17,
        vs_own_history="+22.5% vs this route's past average",
        vs_similar_routes="+18.0% vs similar-length routes this week"
    )

    assert res["flagged"] == "Yes"
    assert res["matched_note_id"] == ""
    assert "No matching note found" in res["reason"]
    assert res["surviving_candidates"] == 0
    # Engine should never have been invoked
    assert len(fake_engine.calls) == 0


def test_audit_spike_executes_through_inference_seam(sample_extracted_notes, fake_engine):
    auditor = AnomalyAuditor(extracted_notes=sample_extracted_notes, engine=fake_engine)

    # Chennai-Bangalore during flooding week has 1 candidate (N001)
    res = auditor.audit_spike(
        route="Chennai-Bangalore",
        week_of="2025-02-24",
        cost_per_tonne_km=5.20,
        vs_own_history="+30.0% vs this route's past average",
        vs_similar_routes="+25.0% vs similar-length routes this week"
    )

    assert res["flagged"] == "No (justified)"
    assert res["matched_note_id"] == "N001"
    assert "flooding" in res["reason"].lower()
    assert res["surviving_candidates"] == 1
    # Verify seam call
    assert len(fake_engine.calls) == 1
    assert fake_engine.calls[0]["call_context"]["route"] == "Chennai-Bangalore"


def test_post_assertion_guardrail_blocks_hallucination(sample_extracted_notes, fake_engine):
    auditor = AnomalyAuditor(extracted_notes=sample_extracted_notes, engine=fake_engine)
    candidates = [{"note_id": "N001", "applies_to": "Chennai-Bangalore"}]

    # Simulated LLM hallucinating N999
    fake_response = '{"verdict": "justified", "matched_note_id": "N999", "explanation": "Fake note."}'
    guarded = auditor._enforce_guardrail(fake_response, candidates)

    assert guarded["flagged"] == "Yes"
    assert guarded["matched_note_id"] == ""
    assert "not a validated candidate" in guarded["reason"]
