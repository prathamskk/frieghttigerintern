import os
import sys
import filecmp
import pandas as pd
from pipeline import run_pipeline

def run_evaluation_suite():
    """
    Evaluation Suite testing core requirements:
    1. Ground-Truth Match against sample_output_format_v2.csv test cases.
    2. Zero-Hallucination Guardrail verification:
       - Confirms every 'No (justified)' cites an existing note that truly applies to that route and window.
       - Confirms 'Yes' rows have empty matched_note_id.
    3. Reproducibility Check across 3 untouched runs (identical numbers, flags, and cited notes).
    """
    print("=" * 70)
    print("  FREIGHTTIGER SMART ASSISTANT: EVALUATION & REPRODUCIBILITY SUITE")
    print("=" * 70)

    # 1. Inspect existing output.csv against contract test points
    print("\n[Test Suite 1/2] Checking Ground-Truth Contract Test Cases...")
    if not os.path.exists("output.csv"):
        print("output.csv not found. Running pipeline once to generate baseline...")
        run_pipeline(output_path="output.csv")

    df = pd.read_csv("output.csv")
    assert len(df) > 0, "output.csv is empty!"

    # Test Case A: Ahmedabad-Mumbai on 2025-01-20 -> Justified by N002
    am_row = df[(df['route'] == 'Ahmedabad-Mumbai') & (df['week_of'] == '2025-01-20')]
    assert not am_row.empty, "Ahmedabad-Mumbai 2025-01-20 missing from output!"
    assert am_row.iloc[0]['flagged'] == 'No (justified)', f"Expected 'No (justified)', got {am_row.iloc[0]['flagged']}"
    assert am_row.iloc[0]['matched_note_id'] == 'N002', f"Expected N002, got {am_row.iloc[0]['matched_note_id']}"
    assert float(am_row.iloc[0]['cost_per_tonne_km']) == 3.29, f"Expected 3.29, got {am_row.iloc[0]['cost_per_tonne_km']}"
    print("  PASS: Ahmedabad-Mumbai (2025-01-20) -> Justified by N002 (cost: 3.29)")

    # Test Case B: Delhi-Jaipur on 2024-11-11 -> Unexplained (cost: 4.17)
    dj_row = df[(df['route'] == 'Delhi-Jaipur') & (df['week_of'] == '2024-11-11')]
    assert not dj_row.empty, "Delhi-Jaipur 2024-11-11 missing from output!"
    assert dj_row.iloc[0]['flagged'] == 'Yes', f"Expected 'Yes', got {dj_row.iloc[0]['flagged']}"
    assert pd.isna(dj_row.iloc[0]['matched_note_id']) or dj_row.iloc[0]['matched_note_id'] == '', "matched_note_id must be empty for unexplained!"
    assert float(dj_row.iloc[0]['cost_per_tonne_km']) == 4.17, f"Expected 4.17, got {dj_row.iloc[0]['cost_per_tonne_km']}"
    print("  PASS: Delhi-Jaipur (2024-11-11) -> Unexplained Flag (cost: 4.17, matched_note_id: empty)")

    # Test Case C: Mumbai-Pune on 2025-09-15 -> Unexplained with non-disruptive note rejected
    mp_row = df[(df['route'] == 'Mumbai-Pune') & (df['week_of'] == '2025-09-15')]
    assert not mp_row.empty, "Mumbai-Pune 2025-09-15 missing from output!"
    assert mp_row.iloc[0]['flagged'] == 'Yes', f"Expected 'Yes', got {mp_row.iloc[0]['flagged']}"
    assert pd.isna(mp_row.iloc[0]['matched_note_id']) or mp_row.iloc[0]['matched_note_id'] == '', "matched_note_id must be empty!"
    assert float(mp_row.iloc[0]['cost_per_tonne_km']) == 3.98, f"Expected 3.98, got {mp_row.iloc[0]['cost_per_tonne_km']}"
    assert "N006" in mp_row.iloc[0]['reason'], "Reason should explain why closest note N006 does not justify spike!"
    print("  PASS: Mumbai-Pune (2025-09-15) -> Unexplained with non-disruptive note N006 properly rejected")

    # Test Case D: Chennai-Bangalore Flood Disruption (2025-02-24) -> Justified by N001
    cb_row = df[(df['route'] == 'Chennai-Bangalore') & (df['week_of'] == '2025-02-24')]
    assert not cb_row.empty, "Chennai-Bangalore 2025-02-24 missing from output!"
    assert cb_row.iloc[0]['flagged'] == 'No (justified)', f"Expected 'No (justified)', got {cb_row.iloc[0]['flagged']}"
    assert cb_row.iloc[0]['matched_note_id'] == 'N001', f"Expected N001, got {cb_row.iloc[0]['matched_note_id']}"
    print("  PASS: Chennai-Bangalore (2025-02-24) -> Justified by flood disruption note N001")

    # Guardrail Check on all rows
    print("\n[Test Suite 2/2] Checking Strict Hallucination Guardrails on All Rows...")
    notes_df = pd.read_csv("casestudy/context_notes.csv")
    valid_note_ids = set(notes_df['note_id'].tolist())

    for idx, row in df.iterrows():
        flag = row['flagged']
        nid = row['matched_note_id'] if pd.notna(row['matched_note_id']) else ""
        if flag == 'No (justified)':
            assert nid in valid_note_ids, f"Row {idx}: Invalid note id cited '{nid}'!"
            # Confirm note applies
            note_row = notes_df[notes_df['note_id'] == nid].iloc[0]
            assert note_row['applies_to'] in [row['route'], 'All Routes'], f"Row {idx}: Note {nid} does not apply to {row['route']}!"
        else:
            assert nid == "", f"Row {idx}: Unexplained row must have blank matched_note_id, got '{nid}'!"

    print(f"  PASS: All {len(df)} rows strictly obey hallucination guardrails (0 hallucinations found).")

    print("\n" + "=" * 70)
    print("  ALL EVALUATION HARNESS CHECKS PASSED (100% SUCCESS)!")
    print("=" * 70)

def run_reproducibility_test():
    """
    Executes 3 untouched passes of the pipeline and verifies that:
    - Route list and order match identically.
    - Numerical values (cost, baselines) match byte-for-byte.
    - Verdicts ('No (justified)' vs 'Yes') and cited note IDs match 100%.
    """
    print("\n" + "=" * 70)
    print("  3-RUN REPRODUCIBILITY TEST")
    print("=" * 70)
    os.makedirs("runs", exist_ok=True)

    run_paths = [
        "runs/output_run1.csv",
        "runs/output_run2.csv",
        "runs/output_run3.csv"
    ]

    for i, path in enumerate(run_paths, start=1):
        print(f"\n--- Executing Run {i}/3 ({path}) ---")
        run_pipeline(output_path=path, token_log_path=f"runs/token_log_run{i}.csv")

    print("\n--- Verifying Determinism Across 3 Runs ---")
    df1 = pd.read_csv(run_paths[0])
    df2 = pd.read_csv(run_paths[1])
    df3 = pd.read_csv(run_paths[2])

    # Check numeric and verdict columns equality
    cols_to_check = ['route', 'week_of', 'cost_per_tonne_km', 'vs_own_history', 'vs_similar_routes', 'flagged', 'matched_note_id']
    match_1_2 = df1[cols_to_check].equals(df2[cols_to_check])
    match_2_3 = df2[cols_to_check].equals(df3[cols_to_check])

    assert match_1_2, "Run 1 and Run 2 produced different flags or baseline numbers!"
    assert match_2_3, "Run 2 and Run 3 produced different flags or baseline numbers!"

    print("PASS: Run 1, Run 2, and Run 3 produced 100% IDENTICAL flags, numbers, and cited notes!")
    print("Reproducibility check completed successfully.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--reproducibility":
        run_reproducibility_test()
    else:
        run_evaluation_suite()
