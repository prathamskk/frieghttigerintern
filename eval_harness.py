import os
import sys
import argparse
import pandas as pd
import numpy as np
from pipeline import run_pipeline

def evaluate_against_golden(golden_path: str = "golden_dataset.csv", output_path: str = "output.csv"):
    """
    Evaluates pipeline predictions against a user-annotated golden dataset.
    Computes:
    - Classification Accuracy, Precision, Recall, F1 (Justified vs Unexplained)
    - Note Attribution Accuracy (Did it cite the right note ID?)
    - Hallucination Rate (Did it cite an unjustified or nonexistent note?)
    - Mismatch Breakdown Table (exact differences for debugging)
    """
    print("=" * 70)
    print("  FREIGHTTIGER EVALUATION HARNESS: GOLDEN DATASET BENCHMARK")
    print("=" * 70)

    if not os.path.exists(golden_path):
        print(f"ERROR: Golden dataset file '{golden_path}' not found!")
        print("Please provide a CSV with columns: route, week_of, expected_flagged, expected_note_id")
        sys.exit(1)

    golden_df = pd.read_csv(golden_path, skipinitialspace=True)
    golden_df.columns = golden_df.columns.str.strip()
    for col in golden_df.select_dtypes(include='object').columns:
        golden_df[col] = golden_df[col].astype(str).str.strip()

    if 'expected_note_id' in golden_df.columns:
        golden_df['expected_note_id'] = golden_df['expected_note_id'].fillna("").astype(str).str.strip().replace('nan', '')
    if 'expected_flagged' in golden_df.columns:
        golden_df['expected_flagged'] = golden_df['expected_flagged'].astype(str).str.strip()

    if not os.path.exists(output_path):
        print(f"'{output_path}' not found. Running pipeline to generate predictions...")
        run_pipeline(output_path=output_path)

    pred_df = pd.read_csv(output_path, skipinitialspace=True)
    pred_df.columns = pred_df.columns.str.strip()
    for col in pred_df.select_dtypes(include='object').columns:
        pred_df[col] = pred_df[col].astype(str).str.strip()
    pred_df['matched_note_id'] = pred_df['matched_note_id'].fillna("").astype(str).str.strip().replace('nan', '')
    pred_df['flagged'] = pred_df['flagged'].astype(str).str.strip()

    # Merge on route and week_of
    merged = pd.merge(
        golden_df,
        pred_df,
        on=['route', 'week_of'],
        how='left',
        suffixes=('_golden', '_pred')
    )

    total_golden_rows = len(golden_df)
    matched_rows = merged[merged['flagged'].notna()]
    missing_rows = merged[merged['flagged'].isna()]

    # Metric counters
    correct_flags = 0
    correct_notes = 0
    hallucinations = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    mismatches = []

    for idx, row in merged.iterrows():
        route = row['route']
        week = row['week_of']
        exp_flag = row['expected_flagged']
        exp_note = row['expected_note_id']

        if pd.isna(row['flagged']):
            mismatches.append({
                "route": route,
                "week_of": week,
                "expected_flag": exp_flag,
                "predicted_flag": "NOT FLAGGED (MISSING)",
                "expected_note": exp_note,
                "predicted_note": "",
                "status": "FAIL: Candidate spike missed by detector",
                "reason": "Not present in pipeline output"
            })
            continue

        pred_flag = row['flagged']
        pred_note = row['matched_note_id']
        reason = row['reason']

        # Classification check
        flag_match = (exp_flag.lower() == pred_flag.lower())
        note_match = (exp_note.upper() == pred_note.upper())

        if flag_match:
            correct_flags += 1
            if exp_flag.lower().startswith("no"):  # Justified
                tp += 1
            else:  # Yes (unexplained)
                tn += 1
        else:
            if pred_flag.lower().startswith("no"):
                fp += 1  # Falsely marked justified
            else:
                fn += 1  # Failed to mark justified

        # Note check
        if note_match:
            correct_notes += 1
        else:
            if pred_flag.lower().startswith("no") and exp_note == "":
                hallucinations += 1

        if not flag_match or not note_match:
            mismatches.append({
                "route": route,
                "week_of": week,
                "expected_flag": exp_flag,
                "predicted_flag": pred_flag,
                "expected_note": exp_note,
                "predicted_note": pred_note,
                "status": "FAIL: Verdict or Note Mismatch",
                "reason": reason
            })

    # Calculations
    accuracy = (correct_flags / total_golden_rows) * 100.0 if total_golden_rows else 0.0
    precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    note_accuracy = (correct_notes / total_golden_rows) * 100.0 if total_golden_rows else 0.0
    hallucination_rate = (hallucinations / total_golden_rows) * 100.0 if total_golden_rows else 0.0

    print(f"\nEvaluated on Golden Dataset: '{golden_path}'")
    print(f"Total Golden Test Cases : {total_golden_rows}")
    print(f"Correct Verdicts        : {correct_flags} / {total_golden_rows} ({accuracy:.1f}%)")
    print(f"Exact Note Citations    : {correct_notes} / {total_golden_rows} ({note_accuracy:.1f}%)")
    print(f"Hallucination Rate      : {hallucination_rate:.1f}% ({hallucinations} incidents)")
    print("-" * 70)
    print(f"Classification Precision : {precision:.1f}%")
    print(f"Classification Recall    : {recall:.1f}%")
    print(f"Classification F1-Score  : {f1:.1f}%")
    print("-" * 70)

    if mismatches:
        print(f"\n[MISMATCH] FOUND {len(mismatches)} MISMATCHES AGAINST GOLDEN ANNOTATIONS:")
        mismatch_df = pd.DataFrame(mismatches)
        print(mismatch_df[['route', 'week_of', 'expected_flag', 'predicted_flag', 'expected_note', 'predicted_note']].to_string(index=False))
    else:
        print("\n[PERFECT] 100% Match on all golden dataset annotations!")

    # Generate Markdown Report Artifact
    report_content = f"""# Evaluation Report: Golden Dataset Benchmark

- **Golden Dataset**: `{golden_path}`
- **Evaluated File**: `{output_path}`
- **Test Rows**: {total_golden_rows}

## Summary Metrics

| Metric | Score | Target |
| :--- | :--- | :--- |
| **Overall Accuracy** | **{accuracy:.1f}%** | > 90% |
| **Precision (Justified)** | **{precision:.1f}%** | > 90% |
| **Recall (Justified)** | **{recall:.1f}%** | > 90% |
| **F1-Score** | **{f1:.1f}%** | > 90% |
| **Note Attribution Accuracy** | **{note_accuracy:.1f}%** | > 95% |
| **Hallucination Rate** | **{hallucination_rate:.1f}%** | 0.0% |

## Detailed Case Breakdown

| Route | Week Of | Expected Flag | Predicted Flag | Expected Note | Predicted Note | Verdict Match |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for _, row in merged.iterrows():
        vm = "✅ PASS" if (str(row['expected_flagged']).lower() == str(row['flagged']).lower() and str(row['expected_note_id']) == str(row['matched_note_id'])) else "❌ FAIL"
        report_content += f"| {row['route']} | {row['week_of']} | {row['expected_flagged']} | {row['flagged']} | {row['expected_note_id']} | {row['matched_note_id']} | {vm} |\n"

    report_path = "eval_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\n-> Exported full report to: {report_path}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "note_accuracy": note_accuracy,
        "hallucination_rate": hallucination_rate,
        "mismatches": len(mismatches)
    }

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

    cols_to_check = ['route', 'week_of', 'cost_per_tonne_km', 'vs_own_history', 'vs_similar_routes', 'flagged', 'matched_note_id']
    match_1_2 = df1[cols_to_check].equals(df2[cols_to_check])
    match_2_3 = df2[cols_to_check].equals(df3[cols_to_check])

    assert match_1_2, "Run 1 and Run 2 produced different flags or baseline numbers!"
    assert match_2_3, "Run 2 and Run 3 produced different flags or baseline numbers!"

    print("PASS: Run 1, Run 2, and Run 3 produced 100% IDENTICAL flags, numbers, and cited notes!")
    print("Reproducibility check completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FreightTiger Assistant Evaluation Suite")
    parser.add_argument("--golden", type=str, default="golden_dataset.csv", help="Path to annotated golden dataset CSV")
    parser.add_argument("--output", type=str, default="output.csv", help="Path to pipeline predictions CSV")
    parser.add_argument("--reproducibility", action="store_true", help="Run 3-pass reproducibility verification")
    args = parser.parse_args()

    if args.reproducibility:
        run_reproducibility_test()
    else:
        evaluate_against_golden(golden_path=args.golden, output_path=args.output)
