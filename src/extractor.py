import os
import re
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from src.inference import InferenceEngine

EXTRACTION_PROMPT = """Analyze this single freight logistics context note.
Note Text: "{note_text}"
Event Start Date: {start_date} (Year: {year})

Extract these 3 independent properties in JSON format:

1. "cost_direction":
   - "INCREASE": The note reports higher costs, surcharges, rate hikes, fuel increases, or detours.
   - "NEUTRAL": The note reports costs were absorbed, traffic normalized, rates were unaffected, demand was stable, or affected areas are outside this dataset.
   - "DECREASE": The note reports rate cuts or discounts.

2. "end_date":
   - If an explicit end date is stated (e.g. "to Mar 8"), output as "YYYY-MM-DD" using {year}.
   - If ongoing or step-change ("starting this week"), output the date 30 days after {start_date}.
   - If a temporary week or event with no end date, output the date 7 days after {start_date}.
   - If neutral/informational, output {start_date}.

3. "estimated_impact_pct":
   - If a percentage or range is mentioned (e.g. "5-7%"), output the approximate number as a float (e.g. 6.0).
   - If a single percentage is stated (e.g. "10%"), output 10.0.
   - If no percentage is mentioned, output null.

Output strictly valid JSON:
{{
  "cost_direction": "INCREASE" | "NEUTRAL" | "DECREASE",
  "end_date": "YYYY-MM-DD",
  "estimated_impact_pct": float or null
}}"""


class NoteMetadataExtractor:
    """
    Offline Note Metadata Extractor (Single-Pass 3-Property Schema).
    Reads base fields (note_id, applies_to, valid_from, original_note) directly from CSV.
    Extracts strictly 3 properties per note via an InferenceEngine seam:
    1. cost_direction ("INCREASE" | "NEUTRAL" | "DECREASE")
    2. end_date ("YYYY-MM-DD")
    3. estimated_impact_pct (float or null)
    """

    def __init__(self, engine: InferenceEngine):
        self.engine = engine

    def extract_single_note(self, note_row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts exactly 3 properties for a context note using the InferenceEngine.
        All base fields are read directly from note_row without LLM hallucination.
        """
        note_id = str(note_row["note_id"]).strip()
        start_date = str(note_row["date"]).strip()
        applies_to = str(note_row["applies_to"]).strip()
        note_text = str(note_row["note"]).strip()

        start_dt = pd.to_datetime(start_date)
        year = start_dt.year

        prompt = EXTRACTION_PROMPT.format(
            note_text=note_text,
            start_date=start_date,
            year=year
        )

        messages = [
            {"role": "user", "content": prompt}
        ]

        result = self.engine.generate(
            messages=messages,
            max_new_tokens=128,
            call_context={
                "route": f"EXTRACTOR_{applies_to}",
                "week_of": start_date,
                "note_candidates": note_id
            }
        )
        response_text = result.text

        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            data = json.loads(json_match.group(0)) if json_match else json.loads(response_text)
            direction = str(data.get("cost_direction", "NEUTRAL")).strip().upper()
            if direction not in ["INCREASE", "DECREASE", "NEUTRAL"]:
                direction = "NEUTRAL"
            end_date = str(data.get("end_date", start_date)).strip()
            impact = data.get("estimated_impact_pct")
            estimated_impact = float(impact) if impact is not None else None
        except Exception:
            direction = "NEUTRAL"
            end_date = start_date
            estimated_impact = None

        # Sanity check: end_date cannot precede start_date
        try:
            if pd.to_datetime(end_date) < start_dt:
                end_date = start_date
        except Exception:
            end_date = start_date

        return {
            "note_id": note_id,
            "applies_to": applies_to,
            "valid_from": start_date,
            "valid_to": end_date,
            "cost_direction": direction,
            "estimated_impact_pct": estimated_impact,
            "original_note": note_text
        }

    def process_and_cache_notes(
        self,
        notes_csv_path: str = "casestudy/context_notes.csv",
        cache_json_path: str = "casestudy/extracted_notes.json",
        force_reextract: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Loads cached metadata if present; otherwise runs offline extraction on context_notes.csv
        and persists structured JSON metadata to disk.
        """
        if not force_reextract and os.path.exists(cache_json_path):
            print(f"[Extractor] Found cached metadata table at '{cache_json_path}'. Loading...")
            with open(cache_json_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            print(f"[Extractor] Loaded {len(cached_data)} structured context notes from cache.")
            return cached_data

        print(f"[Extractor] Running offline metadata extraction on '{notes_csv_path}'...")
        notes_df = pd.read_csv(notes_csv_path)
        extracted_notes = []

        for idx, row in notes_df.iterrows():
            row_dict = row.to_dict()
            print(f"  -> Extracting note {row_dict.get('note_id')} ({row_dict.get('applies_to')}, {row_dict.get('date')})...")
            extracted = self.extract_single_note(row_dict)
            extracted_notes.append(extracted)

        # Ensure target directory exists and save
        os.makedirs(os.path.dirname(os.path.abspath(cache_json_path)), exist_ok=True)
        with open(cache_json_path, "w", encoding="utf-8") as f:
            json.dump(extracted_notes, f, indent=2)

        print(f"[Extractor] Successfully saved {len(extracted_notes)} structured notes to '{cache_json_path}'.")
        return extracted_notes


def extract_clean_metadata(
    notes_df: pd.DataFrame,
    engine: InferenceEngine
) -> List[Dict[str, Any]]:
    """
    Direct function interface for extracting clean 3-property metadata from a context notes DataFrame
    using the provided InferenceEngine.
    """
    extractor = NoteMetadataExtractor(engine=engine)
    extracted_records = []
    for _, row in notes_df.iterrows():
        extracted_records.append(extractor.extract_single_note(row.to_dict()))
    return extracted_records
