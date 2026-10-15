import json
import re
import time
import pandas as pd
from typing import Dict, Any, Optional
from src.tracker import TokenCostTracker
from src.config import FreightConfig, DEFAULT_CONFIG


EXTRACTION_PROMPT = """You are a freight parameter extraction system.
Analyze the user's natural language shipping cost query and extract:
1. "corridor": The origin-destination freight corridor name (e.g., "Ahmedabad-Mumbai", "Chennai-Bangalore", "Delhi-Jaipur", "Mumbai-Delhi", "Mumbai-Pune", "Delhi-Chennai", "Kolkata-Bhubaneswar") or null if not specified.
2. "date_query": The date, week, or month mentioned (e.g., "2025-01-20", "late Feb 2025", "Jan 2025", "November 2024") or null if not specified.
3. "is_complete": boolean, true ONLY if BOTH a corridor AND a date/week are specified, false otherwise.

Respond strictly with a valid JSON object matching this schema:
{{
  "corridor": string | null,
  "date_query": string | null,
  "is_complete": boolean
}}

User query: "{user_query}"
"""

INCOMPLETE_GUIDANCE_MESSAGE = (
    "Please specify both the freight corridor (e.g., Ahmedabad-Mumbai) "
    "and the shipment date or week (e.g., 2025-01-20) to check anomaly records."
)


class ConversationalAssistant:
    """
    Grounded 3-stage conversational assistant for freight corridor anomaly inquiries:
      1. Parameter Extractor: LLM structured JSON output parser.
      2. Deterministic Dual Lookup: output.csv (spikes) and all_metrics_df (nominal baselines).
      3. Response Generator: Factually grounded explanation with attribution guardrail.
    """

    def __init__(
        self,
        engine: Any,
        output_df: pd.DataFrame,
        all_metrics_df: pd.DataFrame,
        tracker: Optional[TokenCostTracker] = None,
        config: FreightConfig = DEFAULT_CONFIG
    ):
        self.engine = engine
        self.output_df = output_df.copy() if output_df is not None else pd.DataFrame()
        self.all_metrics_df = all_metrics_df.copy() if all_metrics_df is not None else pd.DataFrame()
        self.tracker = tracker
        self.config = config
        self.spike_threshold_pct = float(config.spike_threshold_pct)

    def _extract_parameters(self, query: str) -> Dict[str, Any]:
        """Extracts corridor and date_query from user query via InferenceEngine."""
        prompt = EXTRACTION_PROMPT.format(user_query=query)
        messages = [{"role": "user", "content": prompt}]

        result = self.engine.generate(
            messages=messages,
            max_new_tokens=100,
            call_context={"task": "assistant_param_extraction", "query": query}
        )

        try:
            cleaned = result.text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
                cleaned = re.sub(r"\n```$", "", cleaned)
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            raw_json = json_match.group(0) if json_match else cleaned
            parsed = json.loads(raw_json)
            corridor = parsed.get("corridor")
            date_query = parsed.get("date_query")
            raw_complete = parsed.get("is_complete", False)
            if isinstance(raw_complete, str):
                raw_complete = raw_complete.strip().lower() in ["true", "1", "yes"]
            else:
                raw_complete = bool(raw_complete)
            is_complete = raw_complete and bool(corridor) and bool(date_query)
            return {
                "corridor": str(corridor).strip() if corridor else None,
                "date_query": str(date_query).strip() if date_query else None,
                "is_complete": is_complete
            }
        except Exception:
            return {"corridor": None, "date_query": None, "is_complete": False}

    def _normalize_to_monday(self, date_str: Optional[str]) -> Optional[str]:
        """Normalizes date, month, or textual period to its calendar Monday date."""
        if not date_str:
            return None
        try:
            dt = pd.to_datetime(date_str)
            monday = dt - pd.to_timedelta(dt.weekday(), unit='D')
            return monday.strftime('%Y-%m-%d')
        except Exception:
            try:
                cleaned = re.sub(r'^(late|early|mid)\s+', '', str(date_str).strip(), flags=re.IGNORECASE)
                dt = pd.to_datetime(cleaned)
                monday = dt - pd.to_timedelta(dt.weekday(), unit='D')
                return monday.strftime('%Y-%m-%d')
            except Exception:
                return None

    def _lookup(self, corridor: str, week_of: Optional[str]):
        """Deterministic dual lookup against output_df (spikes) and all_metrics_df (baselines)."""
        if not corridor:
            return "OUT_OF_DATASET", None

        # 1. Audited Spikes Table (output_df)
        if not self.output_df.empty and 'route' in self.output_df.columns:
            route_matches = self.output_df[self.output_df['route'].str.lower() == corridor.lower()]
            if not route_matches.empty:
                if week_of and 'week_of' in route_matches.columns:
                    week_matches = route_matches[route_matches['week_of'] == week_of]
                    if not week_matches.empty:
                        rec = week_matches.iloc[0].to_dict()
                        status = "SPIKE_JUSTIFIED" if rec.get("flagged") == "No (justified)" else "SPIKE_UNJUSTIFIED"
                        return status, rec
                elif not week_of:
                    rec = route_matches.iloc[0].to_dict()
                    status = "SPIKE_JUSTIFIED" if rec.get("flagged") == "No (justified)" else "SPIKE_UNJUSTIFIED"
                    return status, rec

        # 2. Weekly Baseline Aggregates (all_metrics_df)
        if not self.all_metrics_df.empty and 'route' in self.all_metrics_df.columns:
            route_matches = self.all_metrics_df[self.all_metrics_df['route'].str.lower() == corridor.lower()]
            if not route_matches.empty:
                if week_of and 'week_of' in route_matches.columns:
                    week_matches = route_matches[route_matches['week_of'] == week_of]
                    if not week_matches.empty:
                        rec = week_matches.iloc[0].to_dict()
                        return "NOMINAL_BASELINE", rec
                elif not week_of:
                    rec = route_matches.iloc[0].to_dict()
                    return "NOMINAL_BASELINE", rec

        return "OUT_OF_DATASET", None

    def _generate_explanation(
        self,
        query: str,
        corridor: str,
        week_of: Optional[str],
        status: str,
        record: Optional[Dict[str, Any]]
    ) -> str:
        """Generates grounded explanation via InferenceEngine adhering to attribution guardrails."""
        if status == "OUT_OF_DATASET":
            return (
                f"No shipment or baseline records were found for {corridor} around "
                f"{week_of or 'the requested date'} in the 2024–2025 dataset."
            )

        if status == "SPIKE_JUSTIFIED":
            prompt = (
                f"You are a Grounded Freight Anomaly Explainer.\n"
                f"Explain why costs on corridor '{corridor}' spiked on the week of {week_of}.\n\n"
                f"Retrieved record:\n"
                f"- Actual cost: ₹{record.get('cost_per_tonne_km', 0.0):.2f} per tonne-km\n"
                f"- Vs history: {record.get('vs_own_history', '')}\n"
                f"- Vs peers: {record.get('vs_similar_routes', '')}\n"
                f"- Verdict: Justified by operational context note {record.get('matched_note_id', '')}\n"
                f"- Causal context: {record.get('reason', '')}\n\n"
                f"Write a concise, factual, plain-English summary stating the actual cost, percentage changes, "
                f"citing note {record.get('matched_note_id', '')}, and explaining the real-world operational disruption."
            )
        elif status == "SPIKE_UNJUSTIFIED":
            prompt = (
                f"You are a Grounded Freight Anomaly Explainer.\n"
                f"Explain why costs on corridor '{corridor}' spiked on the week of {week_of}.\n\n"
                f"Retrieved record:\n"
                f"- Actual cost: ₹{record.get('cost_per_tonne_km', 0.0):.2f} per tonne-km\n"
                f"- Vs history: {record.get('vs_own_history', '')}\n"
                f"- Vs peers: {record.get('vs_similar_routes', '')}\n"
                f"- Verdict: Flagged for human review (unexplained surge)\n"
                f"- Causal context: {record.get('reason', '')}\n\n"
                f"Write a concise, factual, plain-English summary stating the cost, percentage surge, and "
                f"explicitly noting that no operational context note explains this increase so it has been "
                f"flagged for human review."
            )
        else:  # NOMINAL_BASELINE
            cost_val = record.get('cost_per_tonne_km', record.get('cost_per_tonne_km_raw', 0.0))
            vs_hist = record.get('vs_own_history', '')
            prompt = (
                f"You are a Grounded Freight Anomaly Explainer.\n"
                f"Explain shipping costs for corridor '{corridor}' on the week of {week_of}.\n\n"
                f"Retrieved record:\n"
                f"- Actual cost: ₹{cost_val:.2f} per tonne-km\n"
                f"- Vs history: {vs_hist}\n"
                f"- Configured candidate spike threshold: {self.spike_threshold_pct:.1f}%\n\n"
                f"Write a concise, factual, plain-English explanation confirming that operations were normal "
                f"at baseline on this week and no anomaly occurred, explicitly citing that cost variance "
                f"remained below the {self.spike_threshold_pct:.1f}% candidate spike threshold."
            )

        messages = [{"role": "user", "content": prompt}]
        res = self.engine.generate(
            messages=messages,
            max_new_tokens=150,
            call_context={"task": "assistant_explanation", "route": corridor, "week_of": week_of}
        )
        return res.text.strip()

    def answer(self, user_query: str) -> Dict[str, Any]:
        """
        Public conversational interface answering user inquiries.
        Returns a dictionary with 'answer', 'status', 'record', 'parsed_params', and telemetry.
        """
        start_time = time.time()
        parsed = self._extract_parameters(user_query)

        if not parsed.get("is_complete", False):
            elapsed = time.time() - start_time
            return {
                "answer": INCOMPLETE_GUIDANCE_MESSAGE,
                "status": "INCOMPLETE_INPUT",
                "record": None,
                "parsed_params": parsed,
                "latency_seconds": round(elapsed, 3)
            }

        corridor = parsed.get("corridor")
        date_query = parsed.get("date_query")
        week_of = self._normalize_to_monday(date_query)
        parsed["resolved_week_of"] = week_of

        status, record = self._lookup(corridor, week_of)
        answer_text = self._generate_explanation(user_query, corridor, week_of, status, record)

        elapsed = time.time() - start_time
        return {
            "answer": answer_text,
            "status": status,
            "record": record,
            "parsed_params": parsed,
            "latency_seconds": round(elapsed, 3)
        }

