import os
import sys
import time
import pandas as pd
from src.analytics import (
    load_and_preprocess_shipments,
    compute_weekly_route_aggregates,
    compute_baselines_and_flags
)
from src.retriever import ContextNotesRetriever
from src.reasoner import LLMReasoner
from src.tracker import TokenCostTracker

def run_pipeline(
    shipments_path: str = "casestudy/shipment_records.csv",
    notes_path: str = "casestudy/context_notes.csv",
    output_path: str = "output.csv",
    token_log_path: str = "token_cost_log.csv",
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    spike_threshold_pct: float = 20.0,
    device: str = "cpu"
) -> pd.DataFrame:
    """
    Executes the full FreightTiger Shipping Cost Assistant pipeline:
    1. Loads shipments & computes weekly aggregations (cost per tonne-km).
    2. Calculates trailing 8-week baselines and peer baselines.
    3. Detects candidate cost spikes (cost rising > 0% and >= 20% vs history or peers).
    4. Retrieves candidate notes via ChromaDB and temporal pre-filtering.
    5. Reasoner assesses justification with zero hallucination tolerance.
    6. Formats output matching sample_output_format.csv.
    7. Exports token and cost log.
    """
    print("=" * 70)
    print("  FREIGHTTIGER SMART ASSISTANT: SHIPPING COST ANOMALY DETECTOR")
    print("=" * 70)
    start_time = time.time()

    # Step 1: Analytics & Baselines
    print("\n[Step 1/5] Ingesting shipments and calculating weekly route baselines...")
    raw_df = load_and_preprocess_shipments(shipments_path)
    weekly_df = compute_weekly_route_aggregates(raw_df)
    all_metrics_df, candidate_spikes_df = compute_baselines_and_flags(
        weekly_df, spike_threshold_pct=spike_threshold_pct
    )
    print(f"Total shipments processed: {len(raw_df)}")
    print(f"Total route-weeks calculated: {len(all_metrics_df)}")
    print(f"Candidate spikes flagged for audit: {len(candidate_spikes_df)}")

    # Step 2: Initialize RAG Retriever
    print("\n[Step 2/5] Initializing ChromaDB vector index and context notes retriever...")
    retriever = ContextNotesRetriever(notes_path)
    print(f"Indexed {len(retriever.df)} context notes into vector collection.")

    # Step 3: Initialize Token Tracker and LLM Reasoner
    print(f"\n[Step 3/5] Loading {model_name} offline at temperature=0.0...")
    tracker = TokenCostTracker(model_name=model_name)
    reasoner = LLMReasoner(model_name=model_name, tracker=tracker, device=device)

    # Step 4: Evaluate Candidate Spikes
    print("\n[Step 4/5] Auditing candidate spikes against context notes...")
    output_rows = []

    for idx, row in candidate_spikes_df.iterrows():
        route = row['route']
        week_of = row['week_of']
        cost = row['cost_per_tonne_km']
        vs_hist = row['vs_own_history']
        vs_peers = row['vs_similar_routes']

        # Pre-filter candidate notes
        candidates = retriever.get_candidate_notes(route, week_of)
        closest = retriever.find_closest_note(route, week_of)

        eval_res = reasoner.evaluate_spike(
            route=route,
            week_of=week_of,
            cost_per_tonne_km=cost,
            vs_own_history=vs_hist,
            vs_similar_routes=vs_peers,
            candidate_notes=candidates,
            closest_note=closest
        )

        output_rows.append({
            "route": route,
            "week_of": week_of,
            "cost_per_tonne_km": cost,
            "vs_own_history": vs_hist,
            "vs_similar_routes": vs_peers,
            "flagged": eval_res["flagged"],
            "matched_note_id": eval_res["matched_note_id"],
            "reason": eval_res["reason"]
        })

    output_df = pd.DataFrame(output_rows)

    # Step 5: Export Deliverables
    print(f"\n[Step 5/5] Exporting final deliverables...")
    output_df.to_csv(output_path, index=False)
    print(f"-> Generated output CSV: {output_path} ({len(output_df)} rows)")

    tracker.export_csv(token_log_path)
    summary = tracker.get_summary()
    print(f"-> Generated token/cost log: {token_log_path}")
    print("\n" + "=" * 70)
    print("  PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total Execution Time    : {time.time() - start_time:.2f} seconds")
    print(f"Total Candidate Spikes  : {len(output_df)}")
    print(f"  - Justified Spikes    : {(output_df['flagged'] == 'No (justified)').sum()}")
    print(f"  - Flagged for Review  : {(output_df['flagged'] == 'Yes').sum()}")
    print(f"LLM Calls Executed      : {summary['total_calls']}")
    print(f"Total Input Tokens      : {summary['total_input_tokens']}")
    print(f"Total Output Tokens     : {summary['total_output_tokens']}")
    print(f"Total Tokens Processed  : {summary['total_tokens']}")
    print(f"Local Cost              : ${summary['local_cost_usd']:.4f} (100% Free on CPU)")
    print(f"Cloud Rate Equivalent   : ${summary['published_rate_cost_usd']:.6f}")
    print("=" * 70)

    return output_df

if __name__ == "__main__":
    run_pipeline()
