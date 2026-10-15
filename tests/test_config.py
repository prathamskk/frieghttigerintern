import pytest
from dataclasses import FrozenInstanceError
from src.config import FreightConfig, DEFAULT_CONFIG


def test_freight_config_defaults():
    config = FreightConfig()
    assert config.spike_threshold_pct == 20.0
    assert config.trailing_window_weeks == 8
    assert config.shipments_path == "casestudy/shipment_records.csv"
    assert config.notes_path == "casestudy/context_notes.csv"
    assert config.output_path == "output.csv"
    assert config.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.device == "cpu"
    assert config.seed == 42


def test_freight_config_is_immutable():
    config = FreightConfig()
    with pytest.raises(FrozenInstanceError):
        config.spike_threshold_pct = 25.0


def test_freight_config_overrides():
    custom = FreightConfig(spike_threshold_pct=15.0, trailing_window_weeks=4)
    assert custom.spike_threshold_pct == 15.0
    assert custom.trailing_window_weeks == 4
    # Unoverridden values retain defaults
    assert custom.shipments_path == "casestudy/shipment_records.csv"


def test_freight_config_rejects_non_positive_spike_threshold():
    with pytest.raises(ValueError, match="spike_threshold_pct must be greater than 0"):
        FreightConfig(spike_threshold_pct=0.0)

    with pytest.raises(ValueError, match="spike_threshold_pct must be greater than 0"):
        FreightConfig(spike_threshold_pct=-10.0)


def test_freight_config_rejects_invalid_trailing_window_weeks():
    with pytest.raises(ValueError, match="trailing_window_weeks must be at least 1"):
        FreightConfig(trailing_window_weeks=0)

    with pytest.raises(ValueError, match="trailing_window_weeks must be at least 1"):
        FreightConfig(trailing_window_weeks=-2)


def test_freight_config_rejects_empty_paths():
    with pytest.raises(ValueError, match="Path shipments_path cannot be empty"):
        FreightConfig(shipments_path="")

    with pytest.raises(ValueError, match="Path notes_path cannot be empty"):
        FreightConfig(notes_path="   ")
