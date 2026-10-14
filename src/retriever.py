import re
import datetime
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import chromadb

def extract_temporal_window(note_text: str, base_date_str: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Dynamically extracts start and end dates from unstructured note text.
    Handles phrases like:
      - 'from Feb 24 to Mar 8'
      - 'for about a week'
    Defaults to [base_date, base_date + 7 days] if no explicit range is mentioned.
    """
    base_dt = pd.to_datetime(base_date_str)
    year = base_dt.year

    # Pattern: 'from <Month> <Day> to <Month> <Day>'
    pattern = r'from\s+([A-Za-z]+)\s+(\d{1,2})\s+to\s+([A-Za-z]+)?\s*(\d{1,2})'
    match = re.search(pattern, note_text, re.IGNORECASE)
    if match:
        m1, d1, m2, d2 = match.groups()
        m2 = m2 if m2 else m1
        try:
            start_dt = pd.to_datetime(f"{year}-{m1}-{d1}")
            end_dt = pd.to_datetime(f"{year}-{m2}-{d2}")
            if start_dt <= end_dt:
                return start_dt, end_dt
        except Exception:
            pass

    # Default to 7 days duration
    return base_dt, base_dt + pd.Timedelta(days=7)

class ContextNotesRetriever:
    """
    Dynamic RAG & Semantic Retrieval Engine for context notes.
    - Zero hardcoded note IDs.
    - Generic NLP date extraction and geographic route matching.
    - ChromaDB dense vector indexing for semantic similarity search.
    """

    def __init__(self, notes_filepath: str = "casestudy/context_notes.csv"):
        self.notes_filepath = notes_filepath
        self.df = pd.read_csv(notes_filepath)
        self.df['date_dt'] = pd.to_datetime(self.df['date'])

        # Compute dynamic date windows for all notes
        windows = [extract_temporal_window(row['note'], row['date']) for _, row in self.df.iterrows()]
        self.df['window_start'] = [w[0] for w in windows]
        self.df['window_end'] = [w[1] for w in windows]

        # In-memory ChromaDB vector collection
        self.chroma_client = chromadb.Client()
        collection_name = f"context_notes_{int(datetime.datetime.now().timestamp())}"
        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        documents = []
        ids = []
        metadatas = []
        for _, row in self.df.iterrows():
            documents.append(f"{row['applies_to']} | {row['date']} | {row['note']}")
            ids.append(str(row['note_id']))
            metadatas.append({
                "note_id": str(row['note_id']),
                "date": str(row['date']),
                "applies_to": str(row['applies_to']),
                "note": str(row['note'])
            })

        self.collection.add(documents=documents, ids=ids, metadatas=metadatas)

    def get_candidate_notes(
        self,
        route: str,
        week_of: str,
        max_proximity_days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Retrieves candidate notes that geographically apply to the route
        ('All Routes' or matching route corridor) and whose temporal window
        overlaps with the week_of date.
        """
        week_start = pd.to_datetime(week_of)
        week_end = week_start + pd.Timedelta(days=6)
        candidates = []

        for _, row in self.df.iterrows():
            applies = row['applies_to']
            # Geographic filter
            if applies != "All Routes" and applies.strip().lower() != route.strip().lower():
                continue

            # Temporal overlap check:
            # Check if week overlaps with the note's active window [window_start, window_end]
            # or is within max_proximity_days of note base date
            w_start = row['window_start']
            w_end = row['window_end']
            
            is_overlapping = not (week_end < w_start or week_start > w_end)
            day_diff = abs((week_start - row['date_dt']).days)

            if is_overlapping or day_diff <= max_proximity_days:
                candidates.append({
                    "note_id": row['note_id'],
                    "date": row['date'],
                    "window_str": f"{w_start.strftime('%Y-%m-%d')} to {w_end.strftime('%Y-%m-%d')}",
                    "applies_to": row['applies_to'],
                    "note": row['note'],
                    "day_diff": day_diff,
                    "is_overlapping": is_overlapping
                })

        # Rank candidates: prioritize overlapping windows first, then dense vector similarity
        if len(candidates) > 1:
            query = f"Freight shipping surcharge disruption delay cost increase on corridor {route} during {week_of}"
            results = self.collection.query(
                query_texts=[query],
                n_results=len(self.df),
                where={"note_id": {"$in": [c["note_id"] for c in candidates]}}
            )
            id_order = results['ids'][0] if results and results['ids'] else []
            candidates.sort(key=lambda x: (
                not x.get('is_overlapping', False),
                id_order.index(x['note_id']) if x['note_id'] in id_order else 999
            ))

        return candidates

    def find_closest_note(self, route: str, week_of: str) -> Optional[Dict[str, Any]]:
        """
        Finds the temporally closest note applicable to this corridor or 'All Routes'.
        """
        week_dt = pd.to_datetime(week_of)
        applicable = self.df[
            (self.df['applies_to'] == "All Routes") | 
            (self.df['applies_to'].str.lower() == route.lower())
        ].copy()

        if applicable.empty:
            return None

        applicable['day_diff'] = (applicable['date_dt'] - week_dt).abs()
        closest = applicable.sort_values('day_diff').iloc[0]

        return {
            "note_id": closest['note_id'],
            "date": closest['date'],
            "applies_to": closest['applies_to'],
            "note": closest['note'],
            "day_diff": int(closest['day_diff'].days)
        }
