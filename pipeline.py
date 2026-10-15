import os
import sys
import time
import argparse
import pandas as pd
from src.analytics import (
    load_and_preprocess_shipments,
    compute_weekly_route_aggregates,
    compute_baselines_and_flags
)
from src.extractor import NoteMetadataExtractor
from src.auditor import AnomalyAuditor
from src.tracker import TokenCostTracker
from src.inference import LocalQwenEngine
from src.config import FreightConfig, DEFAULT_CONFIG

def run_pipeline(
    config: FreightConfig = DEFAULT_CONFIG,
    force_reextract: bool = False
) -> pd.DataFrame:
    """
    Executes the Production-Grade FreightTiger Shipping Cost Assistant pipeline:
    1. Loads shipments & computes weekly aggregations (cost per tonne-km).
    2. Calculates trailing 8-week baselines and peer baselines.
    3. Detects candidate cost spikes (cost rising > 0% and >= 20% vs history or peers).
    4. Offline Note Metadata Extractor: extracts structured operational metadata (cached to JSON).
    5. Anomaly Auditor:
       - Deterministic Pre-Filtering (no vector search / no LLM waste).
       - Zero-Waste Fallback (0 tokens logged when 0 candidates survive).
       - Online LLM Causal Verifier (primary driver selection at temperature=0.0).
       - Strict Programmatic Post-Assertions (Zero Hallucination).
    6. Formats output matching sample_output_format.csv.
    7. Exports token and cost log.
    """
    print("=" * 70)
    print("  FREIGHTTIGER SMART ASSISTANT: SHIPPING COST ANOMALY DETECTOR")
    print("=" * 70)
    start_time = time.time()

    # Step 1: Analytics & Baselines
    print("\n[Step 1/5] Ingesting shipments and calculating weekly route baselines...")
    raw_df = load_and_preprocess_shipments(config.shipments_path)
    weekly_df = compute_weekly_route_aggregates(raw_df)
    all_metrics_df, candidate_spikes_df = compute_baselines_and_flags(
        weekly_df, config=config
    )
    print(f"Total shipments processed: {len(raw_df)}")
    print(f"Total route-weeks calculated: {len(all_metrics_df)}")
    print(f"Candidate spikes flagged for audit: {len(candidate_spikes_df)}")

    # Step 2: Offline Note Metadata Extraction & Caching
    print("\n[Step 2/5] Loading/Extracting structured context notes operational metadata...")
    tracker = TokenCostTracker(model_name=config.model_name)
    engine = LocalQwenEngine(
        config=config,
        tracker=tracker
    )
    extractor = NoteMetadataExtractor(engine=engine)
    extracted_notes = extractor.process_and_cache_notes(
        notes_csv_path=config.notes_path,
        cache_json_path=config.notes_cache_path,
        force_reextract=force_reextract
    )

    # Step 3: Initialize Anomaly Auditor
    print("\n[Step 3/5] Initializing production Anomaly Auditor...")
    auditor = AnomalyAuditor(
        extracted_notes=extracted_notes,
        engine=engine,
        tracker=tracker
    )

    # Step 4: Audit Candidate Spikes
    print("\n[Step 4/5] Auditing candidate spikes (Deterministic Pre-Filter + Online Verifier)...")
    output_rows = []
    zero_waste_skips = 0
    llm_verifications = 0

    for idx, row in candidate_spikes_df.iterrows():
        route = row['route']
        week_of = row['week_of']
        cost = row['cost_per_tonne_km']
        vs_hist = row['vs_own_history']
        vs_peers = row['vs_similar_routes']

        audit_res = auditor.audit_spike(
            route=route,
            week_of=week_of,
            cost_per_tonne_km=cost,
            vs_own_history=vs_hist,
            vs_similar_routes=vs_peers
        )

        if audit_res.get("surviving_candidates", 0) == 0:
            zero_waste_skips += 1
        else:
            llm_verifications += 1

        output_rows.append({
            "route": route,
            "week_of": week_of,
            "cost_per_tonne_km": cost,
            "vs_own_history": vs_hist,
            "vs_similar_routes": vs_peers,
            "flagged": audit_res["flagged"],
            "matched_note_id": audit_res["matched_note_id"],
            "reason": audit_res["reason"]
        })

    output_df = pd.DataFrame(output_rows)

    # Step 5: Export Deliverables
    print(f"\n[Step 5/5] Exporting final deliverables...")
    output_df.to_csv(config.output_path, index=False)
    print(f"-> Generated output CSV: {config.output_path} ({len(output_df)} rows)")

    tracker.export_csv(config.token_log_path)
    summary = tracker.get_summary()
    print(f"-> Generated token/cost log: {config.token_log_path}")
    print("\n" + "=" * 70)
    print("  PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total Execution Time    : {time.time() - start_time:.2f} seconds")
    print(f"Total Candidate Spikes  : {len(output_df)}")
    print(f"  - Zero-Waste Skips    : {zero_waste_skips} (0 tokens consumed)")
    print(f"  - LLM Verifications   : {llm_verifications}")
    print(f"  - Justified Spikes    : {(output_df['flagged'] == 'No (justified)').sum()}")
    print(f"  - Flagged for Review  : {(output_df['flagged'] == 'Yes').sum()}")
    print(f"Total LLM Calls Logged  : {summary['total_calls']}")
    print(f"Total Input Tokens      : {summary['total_input_tokens']}")
    print(f"Total Output Tokens     : {summary['total_output_tokens']}")
    print(f"Total Tokens Processed  : {summary['total_tokens']}")
    print(f"Local Cost              : ${summary['local_cost_usd']:.4f} (100% Free on CPU)")
    print(f"Cloud Rate Equivalent   : ${summary['published_rate_cost_usd']:.6f}")
    print("=" * 70)

    return output_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FreightTiger Shipping Cost Assistant Pipeline")
    parser.add_argument("--shipments", default=DEFAULT_CONFIG.shipments_path, help="Shipments CSV path")
    parser.add_argument("--notes", default=DEFAULT_CONFIG.notes_path, help="Context notes CSV path")
    parser.add_argument("--cache", default=DEFAULT_CONFIG.notes_cache_path, help="Extracted notes cache JSON path")
    parser.add_argument("--output", default=DEFAULT_CONFIG.output_path, help="Output CSV path")
    parser.add_argument("--token-log", default=DEFAULT_CONFIG.token_log_path, help="Token log CSV path")
    parser.add_argument("--model", default=DEFAULT_CONFIG.model_name, help="HuggingFace model name")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIG.spike_threshold_pct, help="Candidate spike threshold pct")
    parser.add_argument("--re-extract", action="store_true", help="Force re-extraction of context notes")
    parser.add_argument("--device", default=DEFAULT_CONFIG.device, help="Inference device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed, help="Random seed for deterministic generation")

    args = parser.parse_args()
    config = FreightConfig(
        shipments_path=args.shipments,
        notes_path=args.notes,
        notes_cache_path=args.cache,
        output_path=args.output,
        token_log_path=args.token_log,
        model_name=args.model,
        spike_threshold_pct=args.threshold,
        device=args.device,
        seed=args.seed
    )
    run_pipeline(config=config, force_reextract=args.re_extract)

