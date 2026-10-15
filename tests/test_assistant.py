import json
import pytest
import pandas as pd
from src.inference import FakeInferenceEngine
from src.tracker import TokenCostTracker
from src.assistant import ConversationalAssistant
from src.config import FreightConfig, DEFAULT_CONFIG


@pytest.fixture
def sample_dataframes():
    output_df = pd.DataFrame([
        {
            "route": "Chennai-Bangalore",
            "week_of": "2025-02-24",
            "cost_per_tonne_km": 3.58,
            "vs_own_history": "+31.5% vs this route's past average",
            "vs_similar_routes": "+38.7% vs similar-length routes this week",
            "flagged": "No (justified)",
            "matched_note_id": "N001",
            "reason": "The observed cost spike aligns closely with highway flooding from Feb 24 to Mar 8."
        },
        {
            "route": "Delhi-Jaipur",
            "week_of": "2024-11-11",
            "cost_per_tonne_km": 4.17,
            "vs_own_history": "+35.5% vs this route's past average",
            "vs_similar_routes": "+21.0% vs similar-length routes this week",
            "flagged": "Yes",
            "matched_note_id": "",
            "reason": "No matching note found for this route or date range. Cost rise looks unexplained."
        }
    ])

    all_metrics_df = pd.DataFrame([
        {
            "route": "Mumbai-Delhi",
            "week_of": "2024-07-29",
            "cost_per_tonne_km": 2.06,
            "diff_hist_pct": 0.42,
            "vs_own_history": "+0.4% vs this route's past average",
            "diff_peer_pct": 3.36,
            "vs_similar_routes": "+3.4% vs similar-length routes this week"
        }
    ])

    return output_df, all_metrics_df


def test_incomplete_query_returns_guidance(sample_dataframes):
    output_df, all_metrics_df = sample_dataframes
    tracker = TokenCostTracker(model_name="test-engine")

    # Incomplete extraction response from LLM
    fake_engine = FakeInferenceEngine(
        responses={
            "extract_params": json.dumps({
                "corridor": None,
                "date_query": None,
                "is_complete": False
            })
        }
    )

    assistant = ConversationalAssistant(
        engine=fake_engine,
        output_df=output_df,
        all_metrics_df=all_metrics_df,
        tracker=tracker,
        config=DEFAULT_CONFIG
    )

    res = assistant.answer("Why did costs spike?")

    assert res["status"] == "INCOMPLETE_INPUT"
    expected_guidance = (
        "Please specify both the freight corridor (e.g., Ahmedabad-Mumbai) "
        "and the shipment date or week (e.g., 2025-01-20) to check anomaly records."
    )
    assert res["answer"] == expected_guidance
    # Ensure it short-circuited without invoking response synthesis
    assert len(fake_engine.calls) == 1


def test_justified_spike_query_returns_grounded_note_and_metrics(sample_dataframes):
    output_df, all_metrics_df = sample_dataframes
    tracker = TokenCostTracker(model_name="test-engine")

    def mock_llm_handler(messages):
        full_text = " ".join([m.get("content", "") for m in messages])
        if "freight parameter extraction" in full_text:
            return json.dumps({
                "corridor": "Chennai-Bangalore",
                "date_query": "2025-02-24",
                "is_complete": True
            })
        elif "Grounded Freight Anomaly Explainer" in full_text:
            return (
                "On the week of 2025-02-24, Chennai-Bangalore experienced an anomalous cost spike "
                "to ₹3.58 per tonne-km (+31.5% vs past average). This spike is justified by Note N001: "
                "heavy flooding on the highway forced detours and increased trip costs."
            )
        return "{}"

    fake_engine = FakeInferenceEngine(handler=mock_llm_handler)

    assistant = ConversationalAssistant(
        engine=fake_engine,
        output_df=output_df,
        all_metrics_df=all_metrics_df,
        tracker=tracker,
        config=DEFAULT_CONFIG
    )

    res = assistant.answer("Why did Chennai-Bangalore spike on 2025-02-24?")

    assert res["status"] == "SPIKE_JUSTIFIED"
    assert "3.58" in res["answer"]
    assert "N001" in res["answer"]
    assert "flooding" in res["answer"].lower()
    assert res["record"]["flagged"] == "No (justified)"
    assert res["record"]["matched_note_id"] == "N001"
    # Parameter extraction + Response synthesis = 2 calls
    assert len(fake_engine.calls) == 2


def test_unjustified_spike_query_reports_flagged_for_human_review(sample_dataframes):
    output_df, all_metrics_df = sample_dataframes
    tracker = TokenCostTracker(model_name="test-engine")

    def mock_llm_handler(messages):
        full_text = " ".join([m.get("content", "") for m in messages])
        if "freight parameter extraction" in full_text:
            return json.dumps({
                "corridor": "Delhi-Jaipur",
                "date_query": "2024-11-11",
                "is_complete": True
            })
        elif "Grounded Freight Anomaly Explainer" in full_text:
            return (
                "On the week of 2024-11-11, Delhi-Jaipur saw an anomalous cost spike to ₹4.17 per tonne-km "
                "(+35.5% vs past average). No operational context note explains this cost surge, so it has "
                "been flagged for human review."
            )
        return "{}"

    fake_engine = FakeInferenceEngine(handler=mock_llm_handler)

    assistant = ConversationalAssistant(
        engine=fake_engine,
        output_df=output_df,
        all_metrics_df=all_metrics_df,
        tracker=tracker,
        config=DEFAULT_CONFIG
    )

    res = assistant.answer("Why was Delhi-Jaipur high on 2024-11-11?")

    assert res["status"] == "SPIKE_UNJUSTIFIED"
    assert "4.17" in res["answer"]
    assert "human review" in res["answer"].lower()
    assert res["record"]["flagged"] == "Yes"
    assert res["record"]["matched_note_id"] == ""


def test_nominal_baseline_query_reports_normal_operation_and_threshold(sample_dataframes):
    output_df, all_metrics_df = sample_dataframes
    tracker = TokenCostTracker(model_name="test-engine")

    def mock_llm_handler(messages):
        full_text = " ".join([m.get("content", "") for m in messages])
        if "freight parameter extraction" in full_text:
            return json.dumps({
                "corridor": "Mumbai-Delhi",
                "date_query": "2024-07-29",
                "is_complete": True
            })
        elif "Grounded Freight Anomaly Explainer" in full_text:
            return (
                "On the week of 2024-07-29, Mumbai-Delhi operated normally at baseline at ₹2.06 per tonne-km "
                "(+0.4% vs history). No cost anomaly occurred on this week because variance remained well below "
                "the 20.0% candidate spike threshold."
            )
        return "{}"

    fake_engine = FakeInferenceEngine(handler=mock_llm_handler)

    assistant = ConversationalAssistant(
        engine=fake_engine,
        output_df=output_df,
        all_metrics_df=all_metrics_df,
        tracker=tracker,
        config=DEFAULT_CONFIG
    )

    res = assistant.answer("Why did Mumbai-Delhi cost 2.06 on 2024-07-29?")

    assert res["status"] == "NOMINAL_BASELINE"
    assert "2.06" in res["answer"]
    assert "20.0%" in res["answer"] or "20%" in res["answer"]
    assert "no" in res["answer"].lower() and "anomaly" in res["answer"].lower()
    assert res["record"]["cost_per_tonne_km"] == 2.06


def test_out_of_dataset_query_reports_not_found(sample_dataframes):
    output_df, all_metrics_df = sample_dataframes
    tracker = TokenCostTracker(model_name="test-engine")

    def mock_llm_handler(messages):
        return json.dumps({
            "corridor": "Kolkata-Chennai",
            "date_query": "2023-01-01",
            "is_complete": True
        })

    fake_engine = FakeInferenceEngine(handler=mock_llm_handler)

    assistant = ConversationalAssistant(
        engine=fake_engine,
        output_df=output_df,
        all_metrics_df=all_metrics_df,
        tracker=tracker,
        config=DEFAULT_CONFIG
    )

    res = assistant.answer("What happened to Kolkata-Chennai in 2023?")

    assert res["status"] == "OUT_OF_DATASET"
    assert "No shipment or baseline records were found" in res["answer"]
    assert res["record"] is None


