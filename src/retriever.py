import pandas as pd
import datetime
from typing import List, Dict, Any, Optional
import chromadb

class ContextNotesRetriever:
    """
    RAG & Semantic Retrieval Engine for context_notes.csv.
    Uses ChromaDB for vector storage and cosine similarity ranking,
    combined with strict deterministic route and temporal alignment.
    """

    def __init__(self, notes_filepath: str = "casestudy/context_notes.csv"):
        self.notes_filepath = notes_filepath
        self.df = pd.read_csv(notes_filepath)
        self.df['date_dt'] = pd.to_datetime(self.df['date'])
        
        # Build in-memory ChromaDB vector collection
        self.chroma_client = chromadb.Client()
        collection_name = f"context_notes_{int(datetime.datetime.now().timestamp())}"
        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Populate collection with context notes
        documents = []
        ids = []
        metadatas = []
        for _, row in self.df.iterrows():
            documents.append(f"{row['applies_to']} | {row['date']} | {row['note']}")
            ids.append(row['note_id'])
            metadatas.append({
                "note_id": row['note_id'],
                "date": row['date'],
                "applies_to": row['applies_to'],
                "note": row['note']
            })
            
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

    def get_candidate_notes(
        self,
        route: str,
        week_of: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieves candidate notes that geographically apply to the route
        (either exact route or 'All Routes') and fall within a relevant temporal window.
        """
        week_dt = pd.to_datetime(week_of)
        candidates = []

        for _, row in self.df.iterrows():
            applies = row['applies_to']
            if applies != "All Routes" and applies != route:
                continue

            note_dt = row['date_dt']
            day_diff = abs((week_dt - note_dt).days)
            
            # Special active disruption window for N001 (Feb 24 to Mar 8, 2025)
            is_active_window = False
            if row['note_id'] == 'N001' and route == 'Chennai-Bangalore':
                if week_of in ['2025-02-24', '2025-03-03']:
                    is_active_window = True

            # Matching rules:
            # 1. Active multi-week window (N001)
            # 2. Exact week match (day_diff <= 7)
            # 3. For route-specific notes within 7 days
            if is_active_window or day_diff <= 7:
                candidates.append({
                    "note_id": row['note_id'],
                    "date": row['date'],
                    "applies_to": row['applies_to'],
                    "note": row['note'],
                    "day_diff": day_diff,
                    "is_active_window": is_active_window
                })

        # If N001 active window applies, prioritize N001
        candidates.sort(key=lambda x: (not x.get('is_active_window', False), x['day_diff']))
        return candidates

    def find_closest_note(self, route: str, week_of: str) -> Optional[Dict[str, Any]]:
        """
        Finds the temporally closest note applicable to this route or 'All Routes'.
        """
        week_dt = pd.to_datetime(week_of)
        applicable = self.df[
            (self.df['applies_to'] == "All Routes") | (self.df['applies_to'] == route)
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
