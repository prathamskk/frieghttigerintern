# Spec: Grounded Conversational Assistant for Corridor Cost Anomaly Inquiries

## Problem Statement

Users of the freight analytics dashboard need to ask natural-language questions about shipping cost changes (e.g., "Why was Ahmedabad-Mumbai high in Jan 2025?" or "Did Delhi-Jaipur spike in November?"). Currently, the dashboard contains static, brittle keyword matching (`if "chennai" in query`) that fails on arbitrary queries, ignores date context, cannot confirm when a route operated under normal baseline conditions, and risks hallucinations if connected to an unconstrained LLM. Furthermore, users who provide incomplete queries receive no structured guidance on how to formulate an actionable freight inquiry.

## Solution

A robust, 3-stage conversational assistant that extracts freight corridor and temporal parameters from user queries using a lightweight structured LLM extractor, performs a deterministic dual lookup against audited spike records and weekly baseline aggregates, and formats plain-English answers grounded strictly in retrieved facts. 

When a query is incomplete, it immediately provides a clear guidance message asking for the corridor and date. When a specific date or period is provided, the assistant evaluates the corresponding Monday calendar week. If the corridor experienced an audited spike, it reports the verdict, exact percentage variances, and cited operational context note. If the corridor operated normally, it explicitly tells the user that no anomaly occurred on that specific week, citing the pipeline's configured threshold percentage (e.g., $\ge 20\%$ vs history or peers). All LLM generation is strictly grounded and monitored for token and latency overhead.

## User Stories

1. As a freight logistics manager, I want to ask why a specific corridor was expensive during a particular week, so that I understand whether the cost increase was justified by real-world disruptions or requires human investigation.
2. As a freight logistics manager, I want the assistant to evaluate any specific date against its calendar Monday cohort, so that my queries align with the weekly aggregation boundary of the shipping dataset.
3. As a dashboard user, I want to receive immediate guidance if my query is missing either the corridor name or the date reference, so that I know what parameters are required to check cost records.
4. As a supply chain analyst, I want the assistant to confirm when a corridor experienced no anomaly in a given week, so that I know operations were within normal expected cost variance.
5. As a supply chain analyst, I want the assistant to explicitly state the threshold percentage ($\ge 20\%$ vs trailing 8-week history or peer corridors) when reporting a nominal baseline week, so that I understand the quantitative bar used to rule out an anomaly.
6. As a freight operations auditor, I want every justified anomaly explanation to cite the exact context note ID and operational cause, so that I can independently verify the claim against primary carrier bulletins.
7. As a freight operations auditor, I want unexplained cost surges to be explicitly identified with their percentage increases and a recommendation for human review, so that unjustified expenses are not swept under the rug.
8. As a logistics executive, I want the assistant to notify me if a queried corridor or date range does not exist in the 2024–2025 dataset, so that I do not mistake missing coverage for normal pricing.
9. As a dashboard viewer, I want an interactive conversational interface that preserves session turn history, so that I can review recent question-and-answer pairs without losing context.
10. As a dashboard viewer, I want quick-selection sample question chips, so that I can easily test common corridor inquiries without typing full questions from scratch.
11. As an engineering stakeholder, I want to view token consumption and latency telemetry beneath each assistant response, so that I can monitor inference efficiency and ensure budget discipline.
12. As a developer, I want the conversational assistant logic decoupled from the presentation layer, so that core query extraction, baseline cross-referencing, and response generation can be verified in automated tests without running a browser.

## Implementation Decisions

- **Architecture Pattern**: Decoupled 3-stage pipeline (Parameter Extractor -> Deterministic Lookup -> Grounded Response Generator) replacing unstructured RAG or unconstrained generation.
- **Modules**:
  - `Conversational Assistant`: A new domain module encapsulating query parsing, temporal alignment, table lookup, grounded synthesis, and telemetry.
  - `Inference Engine`: Reuses the existing execution seam to run structured parameter extraction and response generation prompts at temperature 0.0.
  - `Dashboard Application`: Updates the conversational UI tab to use modern Streamlit chat message and input components backed by session state.
- **Interface Contract**:
  - The assistant module exposes a single primary method taking a user query string and returning a structured response dictionary containing the formatted answer text, parsed parameters, lookup status (`SPIKE_FOUND`, `NOMINAL_BASELINE`, `OUT_OF_DATASET`, `INCOMPLETE_INPUT`), and token/latency telemetry.
- **Incomplete Input Contract**:
  - If the extracted corridor or temporal reference is null/missing, the assistant short-circuits without generating further LLM completions and returns the exact prompt:
    `"Please specify both the freight corridor (e.g., Ahmedabad-Mumbai) and the shipment date or week (e.g., 2025-01-20) to check anomaly records."`
- **Temporal Alignment**:
  - Dates and relative temporal expressions are converted into calendar Monday dates matching the dataset's Monday-to-Sunday aggregation contract (`(date - timedelta(days=weekday)).strftime('%Y-%m-%d')`).
- **Deterministic Dual Lookup**:
  - Stage 2 queries the audited output table first using composite key `(route, week_of)`. If found, the full audited record (cost, percentage variances vs history and peers, flagged status, matched note ID, and causal explanation) is loaded.
  - If not found in the audited anomaly table, Stage 2 queries the weekly route aggregate baseline metrics. If present, it extracts the actual cost, trailing 8-week baseline average, and calculates variance to confirm baseline stability.
  - If neither table contains the route-week key, Stage 2 returns an out-of-dataset status.
- **Threshold Variable & Configuration**:
  - The candidate spike threshold percentage is treated as a configurable parameter (`spike_threshold_pct`, defaulting to the pipeline's configured 20.0%) rather than a hardcoded literal, enabling consistent threshold propagation across detection, auditing, and conversational reporting.
- **Response Framing & Guardrails**:
  - For **Justified Spikes (`No (justified)`)**: The assistant generates plain-English prose citing the exact matched note ID and its real-world disruption details without robotic boilerplate phrases.
  - For **Unjustified Anomalies (`Yes`)**: The assistant explicitly states that the cost surge has no verified operational justification in the notes and has been flagged for human review.
  - For **Nominal Baseline Weeks**: The assistant reports that the corridor operated normally at baseline and explicitly cites the configurable threshold percentage (`spike_threshold_pct`) to explain why it was not flagged.
  - The response generator enforces a strict post-assertion check: if the retrieved record has no matching note ID, the response text is forbidden from citing or inventing any note identifier.

## Testing Decisions

- **Test Quality Standard**: Tests must strictly exercise external behavior across the public assistant interface rather than asserting internal private methods. Tests verify that given a query string, the assistant returns correct factual numbers, accurate cited note IDs, proper guidance on incomplete inputs, and truthful baseline status.
- **Test Seam**: The highest available seam is the existing `Inference Engine` abstraction (`FakeInferenceEngine`), allowing all LLM calls to be mocked deterministically offline with zero GPU/CPU cost and zero latency in CI.
- **Modules Tested**: The new `Conversational Assistant` module will be tested in isolation across all query permutations:
  - Valid query for an acute justified spike (e.g., Chennai-Bangalore flood week citing N001).
  - Valid query for an unexplained spike (e.g., Delhi-Jaipur surge flagged for review with no note).
  - Valid query for a nominal baseline week (verifying that normal operation is reported and the 20% threshold is cited).
  - Incomplete queries (missing corridor, missing date) asserting the exact guidance string.
  - Out-of-dataset query (unknown corridor or date out of range).
- **Prior Art**: Follows the established patterns in the test suite where the Anomaly Auditor and Note Extractor are tested against the `FakeInferenceEngine` fixture with parameterized test cases.

## Out of Scope

- Real-time multi-agent web searching or external live weather/traffic API querying.
- Multi-month trend forecasting or predictive pricing analytics.
- Audio speech-to-text or voice input interfaces.
- Fine-tuning or updating model weights from conversational user interactions.

## Further Notes

- The assistant operates entirely locally on CPU/CUDA using `Qwen2.5-1.5B-Instruct` or offline test doubles.
- All conversational token usage will be logged into `token_cost_log.csv` alongside batch pipeline telemetry to preserve full system observability.
