import os
import re
import json
import torch
from typing import Dict, Any, Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.tracker import TokenCostTracker

class LLMReasoner:
    """
    Generic Natural Language Reasoner using local Qwen2.5 open-source model.
    - Zero hardcoded note IDs or corridor names.
    - Strictly evaluates note causality from text semantics:
      Does the note describe a condition that raised costs/surcharges, or does it
      state costs were unaffected/absorbed/normal?
    - Runs at temperature=0.0 and do_sample=False for deterministic reproducibility.
    - Programmatic guardrail enforces valid note attribution without hallucination.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        tracker: Optional[TokenCostTracker] = None,
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.tracker = tracker or TokenCostTracker(model_name=model_name)
        self.device = device

        print(f"Loading {model_name} from local cache (offline mode)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            device_map=self.device,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        self.model.eval()
        print("Model loaded successfully.")

    def evaluate_spike(
        self,
        route: str,
        week_of: str,
        cost_per_tonne_km: float,
        vs_own_history: str,
        vs_similar_routes: str,
        candidate_notes: List[Dict[str, Any]],
        closest_note: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates whether a candidate cost spike is justified by external context notes
        using semantic understanding without hardcoded rules.
        """
        # Case 1: No candidate notes found in the corridor/time window
        if not candidate_notes:
            if closest_note and closest_note.get('day_diff', 999) <= 21:
                closest_id = closest_note['note_id']
                closest_date = closest_note['date']
                closest_text = closest_note['note']
                reason = (
                    f"The closest note ({closest_id}, {closest_date}) mentions: \"{closest_text}\" "
                    f"-- it does not describe a reason for a cost rise on this route. "
                    f"No genuine justification found; flagged for review."
                )
            else:
                reason = "No matching note found for this route or date range. Cost rise looks unexplained and worth a human review."

            return {
                "flagged": "Yes",
                "matched_note_id": "",
                "reason": reason
            }

        # Case 2: Candidate note(s) exist; prompt LLM to evaluate causality
        notes_text = "\n".join([
            f"- Note ID: {n['note_id']}, Date: {n['date']}, Active Window: {n.get('window_str', n['date'])}, Corridor: {n['applies_to']}\n  Details: \"{n['note']}\""
            for n in candidate_notes
        ])

        system_prompt = (
            "You are an expert shipping cost anomaly auditor for FreightTiger.\n"
            "Your task: Decide if a cost spike on a freight route during a specific week is genuinely JUSTIFIED by the provided notes.\n\n"
            "EVALUATION CRITERIA:\n"
            "1. A note JUSTIFIES the cost spike if it explicitly describes an event that raised trip costs, required expensive detours, increased fuel prices, or introduced transporter surcharges on this corridor.\n"
            "2. An active disruption spanning multiple dates (e.g., 'Feb 24 to Mar 8') justifies cost spikes in any week falling within that active date period.\n"
            "3. If a note states that costs were NOT affected, compliance costs were absorbed without rate change, demand was stable, operations were normal, or affected routes are outside the dataset, it DOES NOT justify the cost rise.\n"
            "4. If justified, cite the exact note ID that explains the rise.\n"
            "5. If no note explains the rise, verdict must be 'unexplained' and matched_note_id must be \"\".\n\n"
            "Respond strictly in valid JSON format:\n"
            "{\n"
            '  "verdict": "justified" or "unexplained",\n'
            '  "matched_note_id": "<exact Note ID or empty string \"\">",\n'
            '  "explanation": "<1-2 sentence plain-English explanation>"\n'
            "}"
        )

        user_prompt = (
            f"Corridor: {route}\n"
            f"Week: {week_of}\n"
            f"Cost per tonne-km: {cost_per_tonne_km}\n"
            f"vs History: {vs_own_history}\n"
            f"vs Peers: {vs_similar_routes}\n\n"
            f"Retrieved Notes:\n{notes_text}\n\n"
            "Audit Analysis (JSON):"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        input_token_count = len(inputs.input_ids[0])

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False
            )

        output_token_count = len(outputs[0]) - input_token_count
        raw_response = self.tokenizer.decode(outputs[0][input_token_count:], skip_special_tokens=True).strip()

        # Log call in tracker
        self.tracker.record_call(
            route=route,
            week_of=week_of,
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            note_candidates=",".join([c['note_id'] for c in candidate_notes])
        )

        # Parse LLM JSON output
        is_justified = False
        matched_id = ""
        explanation = ""

        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                v_str = str(data.get("verdict", "")).strip().lower()
                if "justified" in v_str and "unexplained" not in v_str:
                    is_justified = True
                    matched_id = str(data.get("matched_note_id", "")).strip()
                    explanation = str(data.get("explanation", "")).strip()
                else:
                    explanation = str(data.get("explanation", "")).strip()
            except Exception:
                pass

        # Fallback inspection if JSON block failed
        if not is_justified and "justified" in raw_response.lower() and "unexplained" not in raw_response.lower():
            # Check if any candidate note was cited
            for n in candidate_notes:
                if n['note_id'] in raw_response:
                    is_justified = True
                    matched_id = n['note_id']
                    break

        # --- PROGRAMMATIC ANTI-HALLUCINATION GUARDRAILS ---
        valid_candidate_ids = [n['note_id'] for n in candidate_notes]
        if is_justified:
            # Enforce that the cited note actually exists in the candidate set
            if matched_id not in valid_candidate_ids:
                # If matched_id was omitted or malformed, pick best candidate if only 1 exists
                if len(valid_candidate_ids) == 1:
                    matched_id = valid_candidate_ids[0]
                else:
                    is_justified = False
                    matched_id = ""

        if is_justified and matched_id in valid_candidate_ids:
            matched_note = next(n for n in candidate_notes if n['note_id'] == matched_id)
            flagged = "No (justified)"
            if explanation and len(explanation) > 20 and not explanation.startswith("reason"):
                reason = f"Matches note {matched_id} dated {matched_note['date']}: {explanation}"
            else:
                reason = (
                    f"Matches note {matched_id} dated {matched_note['date']}: "
                    f"{matched_note['note']} The cost rise has a clear explanation."
                )
        else:
            flagged = "Yes"
            matched_id = ""
            if explanation and len(explanation) > 20 and not explanation.startswith("reason"):
                reason = explanation
            elif candidate_notes:
                c_note = candidate_notes[0]
                reason = (
                    f"The closest note ({c_note['note_id']}, {c_note['date']}) mentions: \"{c_note['note']}\" "
                    f"-- it does not describe a valid justification for a cost rise on this corridor. "
                    f"Flagged for review."
                )
            else:
                reason = "No matching note found for this route or date range. Cost rise looks unexplained and worth a human review."

        return {
            "flagged": flagged,
            "matched_note_id": matched_id,
            "reason": reason
        }
