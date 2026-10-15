import json
import pytest
from src.inference import FakeInferenceEngine
from src.tracker import TokenCostTracker


@pytest.fixture
def fake_engine():
    """
    Standard test double for InferenceEngine.
    Deterministic, offline, records call context and token telemetry.
    """
    return FakeInferenceEngine(
        responses={
            "Chennai-Bangalore": json.dumps({
                "verdict": "justified",
                "matched_note_id": "N001",
                "reason": "Heavy flooding on highway forced detours and higher trip costs."
            })
        }
    )


@pytest.fixture
def sample_extracted_notes():
    """
    Clean, decoupled fixture representing extracted operational context notes.
    Covers active cost increases, route-specific notes, and neutral status notes.
    """
    return [
        {
            "note_id": "N001",
            "applies_to": "Chennai-Bangalore",
            "valid_from": "2025-02-24",
            "valid_to": "2025-03-08",
            "cost_direction": "INCREASE",
            "estimated_impact_pct": None,
            "original_note": "Heavy flooding on highway forced detours and higher trip costs."
        },
        {
            "note_id": "N002",
            "applies_to": "Ahmedabad-Mumbai",
            "valid_from": "2025-01-20",
            "valid_to": "2025-01-26",
            "cost_direction": "INCREASE",
            "estimated_impact_pct": None,
            "original_note": "Regional festival week saw temporary transporter surcharge."
        },
        {
            "note_id": "N003",
            "applies_to": "All Routes",
            "valid_from": "2025-05-05",
            "valid_to": "2025-06-04",
            "cost_direction": "INCREASE",
            "estimated_impact_pct": 6.0,
            "original_note": "Nationwide diesel price hike increased costs by 5-7%."
        },
        {
            "note_id": "N005",
            "applies_to": "Mumbai-Delhi",
            "valid_from": "2024-07-29",
            "valid_to": "2024-08-04",
            "cost_direction": "NEUTRAL",
            "estimated_impact_pct": None,
            "original_note": "Scheduled maintenance caused minor delays; costs not significantly affected."
        },
        {
            "note_id": "N009",
            "applies_to": "Chennai-Bangalore",
            "valid_from": "2025-03-17",
            "valid_to": "2025-03-23",
            "cost_direction": "NEUTRAL",
            "estimated_impact_pct": None,
            "original_note": "Highways returned to normal conditions after flood repairs."
        },
        {
            "note_id": "N010",
            "applies_to": "All Routes",
            "valid_from": "2025-10-27",
            "valid_to": "2025-11-02",
            "cost_direction": "NEUTRAL",
            "estimated_impact_pct": None,
            "original_note": "Fleet tracking compliance costs absorbed by transporters without rate change."
        }
    ]
