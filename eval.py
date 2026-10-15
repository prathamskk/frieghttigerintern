import sys
import argparse
import pandas as pd
from typing import Dict, Any, Union
from src.config import DEFAULT_CONFIG


def evaluate(
    output_source: Union[str, pd.DataFrame] = DEFAULT_CONFIG.output_path,
    ground_truth_source: Union[str, pd.DataFrame] = DEFAULT_CONFIG.ground_truth_path
) -> Dict[str, Any]:
    """
    Evaluates the pipeline's output against the ground truth benchmark.

    Performs a LEFT JOIN starting from ground_truth_eval.csv to output.csv on ['route', 'week_of'].
    Calculates the 6 required audit and attribution metrics:
      1. Flag Concordance (Binary Accuracy) - Target: 100%
      2. Justification Precision - Target: >= 95%
      3. Justification Recall - Target: >= 95%
      4. F1-Score - Target: >= 95%
      5. Attribution Accuracy - Target: 100%
      6. Hallucination Count - Target: 0 (Strict tolerance)
    """
    # Load DataFrames
    if isinstance(ground_truth_source, pd.DataFrame):
        gt_df = ground_truth_source.copy()
    else:
        gt_df = pd.read_csv(ground_truth_source)

    if isinstance(output_source, pd.DataFrame):
        pred_df = output_source.copy()
    else:
        pred_df = pd.read_csv(output_source)

    # Standardize join key strings
    gt_df["route"] = gt_df["route"].astype(str).str.strip()
    gt_df["week_of"] = gt_df["week_of"].astype(str).str.strip()
    pred_df["route"] = pred_df["route"].astype(str).str.strip()
    pred_df["week_of"] = pred_df["week_of"].astype(str).str.strip()

    # Left join ground truth onto predictions
    merged = pd.merge(
        gt_df,
        pred_df,
        on=["route", "week_of"],
        how="left",
        suffixes=("_expected", "_pred")
    )

    # Clean and normalize fields
    def _clean_str(val: Any) -> str:
        if pd.isna(val) or val is None:
            return ""
        s = str(val).strip()
        return "" if s.lower() == "nan" else s

    missing_spikes = []
    total_benchmark_rows = len(merged)

    # Track classification counts for "No (justified)" as positive class
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    concordance_matches = 0

    attribution_denom = 0
    attribution_correct = 0
    hallucination_count = 0

    for _, row in merged.iterrows():
        route = row["route"]
        week_of = row["week_of"]
        exp_flag = _clean_str(row.get("expected_flagged", ""))
        exp_note = _clean_str(row.get("expected_matched_note_id", ""))

        # Check if row was missing from output.csv
        raw_pred_flag = row.get("flagged")
        is_missing = pd.isna(raw_pred_flag) or raw_pred_flag is None

        if is_missing:
            print(f"Missed expected spike: [{route}] on [{week_of}]")
            missing_spikes.append((route, week_of))
            pred_flag = "MISSING"
            pred_note = ""
        else:
            pred_flag = _clean_str(raw_pred_flag)
            pred_note = _clean_str(row.get("matched_note_id", ""))

        # 1. Flag Concordance (Binary Accuracy)
        if pred_flag == exp_flag:
            concordance_matches += 1

        # Positive Class = "No (justified)"
        is_pred_pos = (pred_flag == "No (justified)")
        is_exp_pos = (exp_flag == "No (justified)")

        if is_pred_pos and is_exp_pos:
            tp += 1
        elif is_pred_pos and not is_exp_pos:
            fp += 1
        elif not is_pred_pos and is_exp_pos:
            # Predicted "Yes", "MISSING", or other when expected "No (justified)"
            fn += 1
        else:
            tn += 1

        # 5. Attribution Accuracy
        # Denominator: Count of rows where expected_matched_note_id is not empty
        if exp_note != "":
            attribution_denom += 1
            if pred_note == exp_note:
                attribution_correct += 1

        # 6. Hallucination Count
        # Expected note ID is blank, but model populated an ID
        if exp_note == "" and pred_note != "":
            hallucination_count += 1

    # Metric 1: Flag Concordance
    flag_concordance = (concordance_matches / total_benchmark_rows * 100.0) if total_benchmark_rows > 0 else 0.0

    # Metric 2: Justification Precision
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0

    # Metric 3: Justification Recall
    recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0

    # Metric 4: F1-Score
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Metric 5: Attribution Accuracy
    attr_acc = (attribution_correct / attribution_denom * 100.0) if attribution_denom > 0 else 0.0

    # Metric 6: Hallucination Count
    hallucinations = hallucination_count

    # Evaluate target compliance
    pass_concordance = (flag_concordance == 100.0)
    pass_precision = (precision >= 95.0)
    pass_recall = (recall >= 95.0)
    pass_f1 = (f1 >= 95.0)
    pass_attr = (attr_acc == 100.0)
    pass_hallucination = (hallucinations == 0)

    all_passed = all([
        pass_concordance,
        pass_precision,
        pass_recall,
        pass_f1,
        pass_attr,
        pass_hallucination
    ])

    return {
        "metrics": {
            "flag_concordance": flag_concordance,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "attribution_accuracy": attr_acc,
            "hallucination_count": hallucinations
        },
        "passed": {
            "flag_concordance": pass_concordance,
            "precision": pass_precision,
            "recall": pass_recall,
            "f1": pass_f1,
            "attribution_accuracy": pass_attr,
            "hallucination_count": pass_hallucination
        },
        "all_passed": all_passed,
        "counts": {
            "total_benchmark_rows": total_benchmark_rows,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "missing_spikes": missing_spikes,
            "attribution_denom": attribution_denom,
            "attribution_correct": attribution_correct
        }
    }


def print_scorecard(eval_results: Dict[str, Any]) -> None:
    """
    Renders the evaluation scorecard displaying the 6 metrics, formulas,
    values, target thresholds, and PASS/FAIL status.
    """
    m = eval_results["metrics"]
    p = eval_results["passed"]

    scorecard_data = [
        (
            "1. Flag Concordance",
            "sum(flagged == expected) / Total",
            f"{m['flag_concordance']:.1f}%",
            "100%",
            "PASS" if p["flag_concordance"] else "FAIL"
        ),
        (
            "2. Justification Precision",
            "TP / (TP + FP) [Pos = No (justified)]",
            f"{m['precision']:.1f}%",
            ">= 95%",
            "PASS" if p["precision"] else "FAIL"
        ),
        (
            "3. Justification Recall",
            "TP / (TP + FN) [Pos = No (justified)]",
            f"{m['recall']:.1f}%",
            ">= 95%",
            "PASS" if p["recall"] else "FAIL"
        ),
        (
            "4. F1-Score",
            "2 * (Prec * Rec) / (Prec + Rec)",
            f"{m['f1']:.1f}%",
            ">= 95%",
            "PASS" if p["f1"] else "FAIL"
        ),
        (
            "5. Attribution Accuracy",
            "matched_note_id == expected / non-empty expected",
            f"{m['attribution_accuracy']:.1f}%",
            "100%",
            "PASS" if p["attribution_accuracy"] else "FAIL"
        ),
        (
            "6. Hallucination Count",
            "count(expected == '' and matched != '')",
            f"{m['hallucination_count']}",
            "0",
            "PASS" if p["hallucination_count"] else "FAIL"
        ),
    ]

    print("\n" + "=" * 90)
    print(" " * 28 + "PIPELINE EVALUATION SCORECARD")
    print("=" * 90)
    header = f"{'Metric':<27} | {'Formula':<35} | {'Value':<8} | {'Target':<7} | {'Status'}"
    print(header)
    print("-" * 90)

    for metric, formula, val, target, status in scorecard_data:
        print(f"{metric:<27} | {formula:<35} | {val:<8} | {target:<7} | {status}")

    print("=" * 90)
    overall_status = "ALL METRICS PASSED" if eval_results["all_passed"] else "EVALUATION FAILED"
    print(f"Overall Result: {overall_status}")
    print("=" * 90 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pipeline output against ground truth benchmark")
    parser.add_argument("--output", default=DEFAULT_CONFIG.output_path, help="Pipeline audit output CSV path")
    parser.add_argument("--ground-truth", default=DEFAULT_CONFIG.ground_truth_path, help="Curated ground truth CSV path")
    args = parser.parse_args()

    results = evaluate(
        output_source=args.output,
        ground_truth_source=args.ground_truth
    )

    print_scorecard(results)

    sys.exit(0 if results["all_passed"] else 1)


if __name__ == "__main__":
    main()
