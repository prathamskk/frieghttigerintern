import time
import pandas as pd
from typing import Dict, Any, List

class TokenCostTracker:
    """
    Observability tracker for monitoring LLM calls, exact token counts, and cost.
    Supports both local self-hosted cost ($0.00) and published cloud rate equivalents
    (e.g., Qwen2.5-1.5B hosted pricing at $0.10 per 1M tokens).
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", input_rate_per_m: float = 0.10, output_rate_per_m: float = 0.10):
        self.model_name = model_name
        self.input_rate_per_m = input_rate_per_m
        self.output_rate_per_m = output_rate_per_m
        self.records: List[Dict[str, Any]] = []

    def record_call(self, route: str, week_of: str, input_tokens: int, output_tokens: int, note_candidates: str = ""):
        call_id = len(self.records) + 1
        total_tokens = input_tokens + output_tokens
        # Cloud published rate equivalent (e.g. Together/DeepInfra rate for Qwen 1.5B)
        cost_usd = (input_tokens * (self.input_rate_per_m / 1_000_000.0)) + (output_tokens * (self.output_rate_per_m / 1_000_000.0))
        
        self.records.append({
            "call_id": call_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model_name,
            "route": route,
            "week_of": week_of,
            "candidate_notes": note_candidates,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_usd": round(cost_usd, 6)
        })

    def get_summary(self) -> Dict[str, Any]:
        if not self.records:
            return {
                "model": self.model_name,
                "total_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "local_cost_usd": 0.0,
                "published_rate_cost_usd": 0.0
            }
        
        df = pd.DataFrame(self.records)
        return {
            "model": self.model_name,
            "total_calls": len(df),
            "total_input_tokens": int(df["input_tokens"].sum()),
            "total_output_tokens": int(df["output_tokens"].sum()),
            "total_tokens": int(df["total_tokens"].sum()),
            "local_cost_usd": 0.0,  # 100% free local CPU execution
            "published_rate_cost_usd": round(float(df["cost_usd"].sum()), 6)
        }

    def export_csv(self, filepath: str = "token_cost_log.csv"):
        if not self.records:
            pd.DataFrame(columns=[
                "call_id", "timestamp", "model", "route", "week_of",
                "candidate_notes", "input_tokens", "output_tokens", "total_tokens", "cost_usd"
            ]).to_csv(filepath, index=False)
        else:
            pd.DataFrame(self.records).to_csv(filepath, index=False)
