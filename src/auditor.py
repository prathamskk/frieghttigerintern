import os
import re
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from src.inference import InferenceEngine
from src.tracker import TokenCostTracker

# Clean, generalized system prompt with zero test-scenario leakage
CAUSAL_VERIFIER_SYSTEM_PROMPT = """You are an expert shipping cost anomaly auditor for FreightTiger.
Evaluate whether an observed cost spike on a freight corridor during a specific week is plausibly justified by candidate operational notes.

AUDIT RULES:
1. Proportionality & Causal Plausibility:
   - Acute local disruptions (severe weather closures, infrastructure failure, acute local surcharges) plausibly account for large cost spikes (+15% to +40%).
   - Systematic cost shifts (indexation, minor fuel adjustments) only account for spikes proportional to their stated impact. A 5-7% macro shift cannot justify an isolated +30% spike.
2. Single Primary Causal Driver:
   - Select the single primary note that best explains the timing, location, and magnitude of the spike.
   - Output its ID in 'matched_note_id'.
3. Unexplained Anomalies:
   - If no candidate note accounts for the magnitude or nature of the increase, set 'verdict' to 'unexplained' and 'matched_note_id' to "".

Output strictly valid JSON with this schema:
{
  "verdict": "justified" | "unexplained",
  "matched_note_id": "<string or empty>",
  "reason": "<1-2 sentence concise operational explanation>"
}"""


class AnomalyAuditor:
    """
    Evaluates observed freight cost spikes against operational context notes
    using an InferenceEngine seam for causal reasoning.
    """

    def __init__(
        self,
        extracted_notes: List[Dict[str, Any]],
        engine: InferenceEngine,
        tracker: Optional[TokenCostTracker] = None
    ):
        self.engine = engine
        self.tracker = tracker

        # Pre-parse timestamps once to avoid loop overhead
        self.notes = []
        for n in extracted_notes:
            item = dict(n)
            item["_v_from"] = pd.to_datetime(item.get("valid_from"))
            item["_v_to"] = pd.to_datetime(item.get("valid_to"))
            self.notes.append(item)

    def prefilter_candidates(self, route: str, week_of: str) -> List[Dict[str, Any]]:
        week_start = pd.to_datetime(week_of)
        week_end = week_start + pd.Timedelta(days=6)
        candidates = []

        norm_route = route.strip().lower()

        for note in self.notes:
            applies = str(note.get("applies_to", "")).strip()
            if applies != "All Routes" and applies.lower() != norm_route:
                continue

            direction = str(note.get("cost_direction", "NEUTRAL")).strip().upper()
            if direction != "INCREASE":
                continue

            # Interval overlap: not (week_end < valid_from or week_start > valid_to)
            if not (week_end < note["_v_from"] or week_start > note["_v_to"]):
                candidates.append(note)

        return candidates

    def audit_spike(
        self,
        route: str,
        week_of: str,
        cost_per_tonne_km: float,
        vs_own_history: str,
        vs_similar_routes: str
    ) -> Dict[str, Any]:
        candidates = self.prefilter_candidates(route, week_of)

        # Zero-waste fallback: 0 tokens incurred if no candidates survive
        if not candidates:
            if self.tracker:
                self.tracker.record_call(
                    route=route,
                    week_of=week_of,
                    input_tokens=0,
                    output_tokens=0,
                    note_candidates="SKIPPED_PREFILTER"
                )
            return {
                "flagged": "Yes",
                "matched_note_id": "",
                "reason": "No matching note found for this route or date range. Cost rise looks unexplained and worth a human review.",
                "surviving_candidates": 0
            }

        candidate_blocks = []
        for n in candidates:
            impact_val = n.get("estimated_impact_pct")
            impact_desc = f"{impact_val}%" if impact_val is not None else "Unspecified / Qualitative"
            note_body = n.get("original_note") or n.get("note") or ""

            candidate_blocks.append(
                f"- Note ID: {n['note_id']}\n"
                f"  Corridor: {n['applies_to']}\n"
                f"  Active Window: {n['valid_from']} to {n['valid_to']}\n"
                f"  Stated Impact: {impact_desc}\n"
                f"  Details: \"{note_body}\""
            )
        notes_text = "\n".join(candidate_blocks)

        user_content = (
            f"Corridor: {route}\n"
            f"Week Of: {week_of}\n"
            f"Observed Metric: ₹{cost_per_tonne_km:.2f} per tonne-km\n"
            f"Baseline Variance: {vs_own_history} vs own history | {vs_similar_routes} vs similar routes\n\n"
            f"Candidate Notes:\n{notes_text}\n\n"
            f"Evaluate plausibility and select the single primary causal driver in valid JSON."
        )

        messages = [
            {"role": "system", "content": CAUSAL_VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        result = self.engine.generate(
            messages=messages,
            max_new_tokens=150,
            call_context={
                "route": route,
                "week_of": week_of,
                "note_candidates": ",".join([c["note_id"] for c in candidates])
            }
        )

        return self._enforce_guardrail(result.text, candidates)

    def _enforce_guardrail(self, response_text: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid_candidate_ids = {c["note_id"].strip().upper(): c for c in candidates}
        
        # Non-greedy JSON search
        json_match = re.search(r'\{.*?\}', response_text.strip(), re.DOTALL)
        
        verdict = "unexplained"
        matched_id = ""
        reason = ""

        if json_match:
            try:
                data = json.loads(json_match.group(0))
                verdict = str(data.get("verdict", "unexplained")).strip().lower()
                matched_id = str(data.get("matched_note_id", "")).strip().upper()
                reason = str(data.get("reason") or data.get("explanation") or "").strip()
            except Exception:
                pass

        # Post-assertion check against hallucinated note IDs
        if verdict == "justified":
            if matched_id in valid_candidate_ids:
                return {
                    "flagged": "No (justified)",
                    "matched_note_id": matched_id,
                    "reason": reason or f"Matches note {matched_id}: verified operational explanation.",
                    "surviving_candidates": len(candidates)
                }
            else:
                return {
                    "flagged": "Yes",
                    "matched_note_id": "",
                    "reason": f"Flagged: Model suggested note {matched_id} which is not a validated candidate for this route/timeframe.",
                    "surviving_candidates": len(candidates)
                }

        # Fallback to flagged unexplained
        return {
            "flagged": "Yes",
            "matched_note_id": "",
            "reason": reason or "Cost rise not sufficiently explained by available context notes. Flagged for review.",
            "surviving_candidates": len(candidates)
        }
