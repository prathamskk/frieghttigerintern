# Evaluation Report: Golden Dataset Benchmark

- **Golden Dataset**: `golden_dataset.csv`
- **Evaluated File**: `output.csv`
- **Test Rows**: 6

## Summary Metrics

| Metric | Score | Target |
| :--- | :--- | :--- |
| **Overall Accuracy** | **83.3%** | > 90% |
| **Precision (Justified)** | **100.0%** | > 90% |
| **Recall (Justified)** | **66.7%** | > 90% |
| **F1-Score** | **80.0%** | > 90% |
| **Note Attribution Accuracy** | **83.3%** | > 95% |
| **Hallucination Rate** | **0.0%** | 0.0% |

## Detailed Case Breakdown

| Route | Week Of | Expected Flag | Predicted Flag | Expected Note | Predicted Note | Verdict Match |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Ahmedabad-Mumbai | 2025-01-20 | No (justified) | No (justified) | N002 | N002 | ✅ PASS |
| Delhi-Jaipur | 2024-11-11 | Yes | Yes |  |  | ✅ PASS |
| Mumbai-Pune | 2025-09-15 | Yes | Yes |  |  | ✅ PASS |
| Chennai-Bangalore | 2025-02-24 | No (justified) | No (justified) | N001 | N001 | ✅ PASS |
| Chennai-Bangalore | 2025-03-03 | No (justified) | Yes | N001 |  | ❌ FAIL |
| Delhi-Jaipur | 2024-11-18 | Yes | Yes |  |  | ✅ PASS |
