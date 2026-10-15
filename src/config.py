from dataclasses import dataclass


@dataclass(frozen=True)
class FreightConfig:
    """
    Centralized domain configuration for FreightTiger Shipping Cost Assistant.
    Encapsulates calculation baselines, file paths, and inference parameters
    at a single validated seam.
    """
    # 1. Core Analytics & Baseline Rules
    spike_threshold_pct: float = 20.0
    trailing_window_weeks: int = 8

    # 2. Canonical File Paths
    shipments_path: str = "casestudy/shipment_records.csv"
    notes_path: str = "casestudy/context_notes.csv"
    notes_cache_path: str = "casestudy/extracted_notes.json"
    output_path: str = "output.csv"
    token_log_path: str = "token_cost_log.csv"
    ground_truth_path: str = "ground_truth_eval.csv"

    # 3. Model & Inference Engine Parameters
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "cpu"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.spike_threshold_pct <= 0:
            raise ValueError(f"spike_threshold_pct must be greater than 0, got {self.spike_threshold_pct}")
        if self.trailing_window_weeks < 1:
            raise ValueError(f"trailing_window_weeks must be at least 1, got {self.trailing_window_weeks}")
        for path_field in [
            "shipments_path",
            "notes_path",
            "notes_cache_path",
            "output_path",
            "token_log_path",
            "ground_truth_path",
        ]:
            val = getattr(self, path_field)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"Path {path_field} cannot be empty")


# Canonical default singleton instance
DEFAULT_CONFIG = FreightConfig()

# Exported module-level constants for direct access
SPIKE_THRESHOLD_PCT = DEFAULT_CONFIG.spike_threshold_pct
TRAILING_WINDOW_WEEKS = DEFAULT_CONFIG.trailing_window_weeks
DEFAULT_SHIPMENTS_PATH = DEFAULT_CONFIG.shipments_path
DEFAULT_NOTES_PATH = DEFAULT_CONFIG.notes_path
DEFAULT_NOTES_CACHE_PATH = DEFAULT_CONFIG.notes_cache_path
DEFAULT_OUTPUT_PATH = DEFAULT_CONFIG.output_path
DEFAULT_TOKEN_LOG_PATH = DEFAULT_CONFIG.token_log_path
DEFAULT_GROUND_TRUTH_PATH = DEFAULT_CONFIG.ground_truth_path
DEFAULT_MODEL_NAME = DEFAULT_CONFIG.model_name
DEFAULT_DEVICE = DEFAULT_CONFIG.device
DEFAULT_SEED = DEFAULT_CONFIG.seed
