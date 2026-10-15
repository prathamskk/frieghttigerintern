import os
import json
import pytest
import pandas as pd
from src.extractor import NoteMetadataExtractor, extract_clean_metadata
from src.inference import FakeInferenceEngine


def test_extractor_single_note_with_fake_engine():
    fake_engine = FakeInferenceEngine(
        responses={
            "flooding": json.dumps({
                "cost_direction": "INCREASE",
                "end_date": "2025-03-08",
                "estimated_impact_pct": 15.0
            })
        }
    )

    extractor = NoteMetadataExtractor(engine=fake_engine)
    note_row = {
        "note_id": "N001",
        "date": "2025-02-24",
        "applies_to": "Chennai-Bangalore",
        "note": "Heavy flooding on highway forced detours."
    }

    result = extractor.extract_single_note(note_row)

    assert result["note_id"] == "N001"
    assert result["applies_to"] == "Chennai-Bangalore"
    assert result["valid_from"] == "2025-02-24"
    assert result["valid_to"] == "2025-03-08"
    assert result["cost_direction"] == "INCREASE"
    assert result["estimated_impact_pct"] == 15.0
    assert result["original_note"] == "Heavy flooding on highway forced detours."

    assert len(fake_engine.calls) == 1
    assert fake_engine.calls[0]["call_context"]["route"] == "EXTRACTOR_Chennai-Bangalore"


def test_extractor_handles_markdown_wrapped_json():
    wrapped_json = "Here is the extracted result:\n```json\n" + json.dumps({
        "cost_direction": "INCREASE",
        "end_date": "2025-02-28",
        "estimated_impact_pct": 10.0
    }) + "\n```"

    fake_engine = FakeInferenceEngine(default_response=wrapped_json)
    extractor = NoteMetadataExtractor(engine=fake_engine)
    note_row = {
        "note_id": "N002",
        "date": "2025-02-20",
        "applies_to": "Mumbai-Delhi",
        "note": "Temporary fuel surcharge."
    }

    result = extractor.extract_single_note(note_row)
    assert result["cost_direction"] == "INCREASE"
    assert result["valid_to"] == "2025-02-28"
    assert result["estimated_impact_pct"] == 10.0


def test_extractor_normalizes_unknown_direction_to_neutral():
    fake_engine = FakeInferenceEngine(
        default_response=json.dumps({
            "cost_direction": "UNKNOWN_DIRECTION",
            "end_date": "2025-04-01"
        })
    )
    extractor = NoteMetadataExtractor(engine=fake_engine)
    note_row = {
        "note_id": "N003",
        "date": "2025-03-01",
        "applies_to": "Delhi-Jaipur",
        "note": "Status report."
    }

    result = extractor.extract_single_note(note_row)
    assert result["cost_direction"] == "NEUTRAL"


def test_extractor_guards_against_inverted_end_date():
    # End date earlier than start date
    fake_engine = FakeInferenceEngine(
        default_response=json.dumps({
            "cost_direction": "INCREASE",
            "end_date": "2024-01-01"
        })
    )
    extractor = NoteMetadataExtractor(engine=fake_engine)
    note_row = {
        "note_id": "N004",
        "date": "2025-02-01",
        "applies_to": "Kolkata-Bhubaneswar",
        "note": "Short delay."
    }

    result = extractor.extract_single_note(note_row)
    # Must reset end_date to start_date
    assert result["valid_to"] == "2025-02-01"


def test_extractor_fallback_when_invalid_json():
    fake_engine = FakeInferenceEngine(default_response="Not valid JSON at all")
    extractor = NoteMetadataExtractor(engine=fake_engine)
    note_row = {
        "note_id": "N005",
        "date": "2025-01-20",
        "applies_to": "Ahmedabad-Mumbai",
        "note": "Some note text"
    }

    result = extractor.extract_single_note(note_row)

    assert result["cost_direction"] == "NEUTRAL"
    assert result["valid_to"] == "2025-01-20"
    assert result["estimated_impact_pct"] is None


def test_extract_clean_metadata_functional_interface():
    fake_engine = FakeInferenceEngine(
        default_response=json.dumps({
            "cost_direction": "NEUTRAL",
            "end_date": "2025-01-01",
            "estimated_impact_pct": None
        })
    )
    df = pd.DataFrame([
        {"note_id": "N1", "date": "2025-01-01", "applies_to": "R1", "note": "Note 1"},
        {"note_id": "N2", "date": "2025-01-02", "applies_to": "R2", "note": "Note 2"}
    ])

    results = extract_clean_metadata(df, engine=fake_engine)
    assert len(results) == 2
    assert results[0]["note_id"] == "N1"
    assert results[1]["note_id"] == "N2"


def test_extracted_notes_cache_contract():
    """
    Validates structural and typing contract of extracted context notes cache:
    - Non-empty collection and unique note_id across all records.
    - Required schema fields present.
    - Valid cost_direction domain enum and ISO dates.
    - Key disruption anchors correctly classified.
    """
    cache_path = "casestudy/extracted_notes.json"
    assert os.path.exists(cache_path), f"Extracted notes JSON file '{cache_path}' not found!"
    with open(cache_path, "r", encoding="utf-8") as f:
        notes = json.load(f)

    assert len(notes) > 0, "Extracted notes list is empty!"
    note_ids = [n["note_id"] for n in notes]
    assert len(note_ids) == len(set(note_ids)), "Duplicate note_ids found in extracted notes!"
    valid_directions = {"INCREASE", "NEUTRAL", "DECREASE"}

    for n in notes:
        assert "note_id" in n and n["note_id"].strip()
        assert "applies_to" in n and n["applies_to"].strip()
        assert "original_note" in n and n["original_note"].strip()
        assert n.get("cost_direction") in valid_directions, f"Invalid direction in note {n['note_id']}"

        start_str = n.get("valid_from") or n.get("start_date")
        end_str = n.get("valid_to") or n.get("end_date")
        assert start_str is not None and end_str is not None
        start_dt = pd.to_datetime(start_str)
        end_dt = pd.to_datetime(end_str)
        assert end_dt >= start_dt, f"End date before start date in note {n['note_id']}"

        impact = n.get("estimated_impact_pct")
        assert impact is None or isinstance(impact, (float, int))

    # Verify key operational disruption anchors
    notes_by_id = {n["note_id"]: n for n in notes}
    assert notes_by_id["N001"]["cost_direction"] == "INCREASE"
    assert notes_by_id["N001"]["applies_to"] == "Chennai-Bangalore"
    assert notes_by_id["N002"]["cost_direction"] == "INCREASE"
    assert notes_by_id["N002"]["applies_to"] == "Ahmedabad-Mumbai"
    assert notes_by_id["N003"]["cost_direction"] == "INCREASE"
    assert notes_by_id["N003"]["applies_to"] == "All Routes"

