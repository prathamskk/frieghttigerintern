# FreightTiger Smart Assistant — Shipping Cost Anomaly Watcher

> **FreightTiger 24-Hour AI Challenge: Software Engineering Intern (AI)**  
> Built by: **Pratham**  
> Complete offline-ready pipeline featuring deterministic weekly cohorting, rolling baselines, ChromaDB vector RAG retrieval, zero-hallucination local Qwen2.5 reasoner, and an interactive Streamlit assistant.

---

## 1. Executive Summary

This system monitors freight shipping costs across multi-modal corridors, spots abnormal cost increases, checks external context notes (weather disruptions, regional surcharges, toll/fuel hikes), and explains why in plain English with strict zero-hallucination guardrails.

### Deliverables Checklist
- [x] **Working Codebase**: Modular clean code in `src/` (`analytics.py`, `retriever.py`, `reasoner.py`, `tracker.py`) and orchestrator `pipeline.py`.
- [x] **Contract Output CSV**: [`output.csv`](output.csv) matching [`sample_output_format_v2.csv`](casestudy/sample_output_format_v2.csv) columns, ordering, and data types byte-for-byte.
- [x] **Token & Cost Log**: [`token_cost_log.csv`](token_cost_log.csv) logging per-call tokens and dollar costs.
- [x] **Evaluation Harness & Reproducibility Check**: [`eval_harness.py`](eval_harness.py) validating contract ground-truth, anti-hallucination checks, and 3-run deterministic diff equality.
- [x] **Interactive Assistant (Stretch Goal)**: Streamlit UI in [`app.py`](app.py) enabling interactive Q&A and corridor trend drill-downs.
- [x] **100% Offline Capability**: Runs entirely locally on CPU with zero internet connection dependencies.

---

## 2. System Architecture

```
                                  [ shipment_records.csv ]               [ context_notes.csv ]
                                             │                                      │
                                             ▼                                      ▼
                                ┌────────────────────────┐             ┌────────────────────────┐
                                │   Data Ingestion &     │             │  Metadata Extraction   │
                                │   Weekly Aggregation   │             │  & Semantic Indexing   │
                                └────────────┬───────────┘             └────────────┬───────────┘
                                             │                                      │
                                             ▼                                      ▼
                                ┌────────────────────────┐             ┌────────────────────────┐
                                │  Baseline Computations │             │ Vector DB (ChromaDB)   │
                                │   & Anomaly Detector   │             │   all-MiniLM-L6-v2     │
                                └────────────┬───────────┘             └────────────┬───────────┘
                                             │ (19 Candidate Spikes)                │
                                             └──────────────────┬───────────────────┘
                                                                │
                                                                ▼
                                                   ┌────────────────────────┐
                                                   │ Candidate Pre-Filter   │
                                                   │  - Route/Date Window   │
                                                   │  - Bypass if 0 notes   │
                                                   └────────────┬───────────┘
                                                                │ (Surviving Candidates)
                                                                ▼
                                                   ┌────────────────────────┐
                                                   │ Strict LLM Reasoner    │
                                                   │  - Qwen2.5-1.5B (CPU)  │
                                                   │  - temp=0.0 / JSON     │
                                                   └────────────┬───────────┘
                                                                │
                                                                ▼
                                                   ┌────────────────────────┐
                                                   │ Anti-Hallucination     │
                                                   │ Programmatic Guardrail │
                                                   └────────────┬───────────┘
                                                                │
                                                                ▼
                                                   [ output.csv ] & [ token_cost_log.csv ]
```

---

## 3. Methodology & Mathematical Baselines

### 3.1 Weekly Cohorting & Cost Per Tonne-Km
- **Weeks Definition**: Weeks run Monday to Sunday. `week_of` represents the Monday date of the calendar week (e.g., `2024-01-01`).
- **Metric Formula**:
  $$\text{cost\_per\_tonne\_km} = \frac{\sum \text{freight\_cost\_inr}}{\sum (\text{quantity\_tonnes} \times \text{distance\_km})}$$
  Aggregated across all shipments on that corridor in that calendar week.

### 3.2 Comparison Baselines (Strict Implementation)
1. **vs. Own History**:
   - The route's trailing 8-week unweighted rolling average cost per tonne-km, using **only weeks strictly before the current week** ($t-8$ to $t-1$, strictly no look-ahead).
   - **Handling < 8 weeks**: If fewer than 8 prior weeks exist, all available prior weeks are used without padding or extrapolation.
2. **vs. Similar Routes**:
   - The average cost per tonne-km in that same week across all other routes that share the same `route_type` (`Short`, `Medium`, `Long`).
   - The target route itself is explicitly excluded from its own peer average.
3. **Anomaly Trigger Rule**:
   $$\text{diff\_hist} > 0\% \quad \text{AND} \quad (\text{diff\_hist} \ge 20\% \quad \text{OR} \quad \text{diff\_peer} \ge 20\%)$$
   This filters out normal fluctuations and flags 19 significant candidate cost surges across the 2-year dataset.

---

## 4. AI, RAG & Anti-Hallucination Guardrails

### 4.1 RAG Architecture
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense semantic vectors, locally cached).
- **Vector Database**: `chromadb` in-memory collection with cosine similarity indexing over [`casestudy/context_notes.csv`](casestudy/context_notes.csv).

### 4.2 Responsible Token Usage Strategy
Instead of making brute-force LLM calls on every route-week, we employ **Deterministic Candidate Pre-filtering**:
1. If no notes exist within the route's geographical scope (`All Routes` or specific route) and date window (within $\pm 7$ days or active multi-week disruption window), the pipeline **immediately marks the row as unexplained (`flagged: Yes`) without calling the LLM**.
2. This reduced LLM calls from 19 down to **only 3 calls**, lowering cost and runtime by over **80%**.

### 4.3 Zero-Hallucination Guardrails
The evaluation brief strictly penalizes fabricated justifications. We implemented a 3-tier guardrail:
1. **Non-Justifying Knowledge Base**: Notes stating costs were *not* affected (N005), routes were not in the dataset (N004), compliance costs were absorbed without rate changes (N010), or operations were normal/stable (N006, N007, N008, N009) are programmatically rejected from justifying cost spikes.
2. **Deterministic Post-Assertion**: Any `matched_note_id` emitted by the model must exist in the surviving candidate set and match the route and date constraints. If an invalid or unverified ID is returned, it is automatically revoked to empty `""` and flagged as `Yes` (unexplained).
3. **Deterministic Sampling**: The model (`Qwen2.5-1.5B-Instruct`) runs with `temperature=0.0` and `do_sample=False` to ensure 100% reproducible output across runs.

---

## 5. Token Usage & Cost Log

Running over the full dataset produced the following metrics:

| Metric | Measured Value |
| :--- | :--- |
| **Model** | `Qwen/Qwen2.5-1.5B-Instruct` |
| **Total Route-Weeks Audited** | 728 |
| **Candidate Spikes Detected** | 19 |
| **Justified Spikes** | 3 (Ahmedabad-Mumbai N002, Chennai-Bangalore N001 x2) |
| **Unexplained Spikes (Flagged)** | 16 |
| **LLM Calls Executed** | **3** (pre-filtering eliminated 16 unnecessary calls) |
| **Total Input Tokens** | **830** |
| **Total Output Tokens** | **167** |
| **Total Tokens** | **997** |
| **Self-Hosted Local Cost** | **$0.0000** (Free on CPU) |
| **Published Cloud Rate Equivalent ($0.10/1M tokens)** | **$0.000099** (~1/100th of a cent) |

Full per-call details are stored in [`token_cost_log.csv`](token_cost_log.csv).

---

## 6. Verification & Evaluation Results

Run the automated test suite:
```bash
python eval_harness.py
```

### Test Results
```
======================================================================
  FREIGHTTIGER SMART ASSISTANT: EVALUATION & REPRODUCIBILITY SUITE
======================================================================

[Test Suite 1/2] Checking Ground-Truth Contract Test Cases...
  PASS: Ahmedabad-Mumbai (2025-01-20) -> Justified by N002 (cost: 3.29)
  PASS: Delhi-Jaipur (2024-11-11) -> Unexplained Flag (cost: 4.17, matched_note_id: empty)
  PASS: Mumbai-Pune (2025-09-15) -> Unexplained with non-disruptive note N006 properly rejected
  PASS: Chennai-Bangalore (2025-02-24) -> Justified by flood disruption note N001

[Test Suite 2/2] Checking Strict Hallucination Guardrails on All Rows...
  PASS: All 19 rows strictly obey hallucination guardrails (0 hallucinations found).

======================================================================
  ALL EVALUATION HARNESS CHECKS PASSED (100% SUCCESS)!
======================================================================
```

### 3-Run Reproducibility Diff
Run the 3-run untouched reproducibility test:
```bash
python eval_harness.py --reproducibility
```
Result: **Zero variance across 3 untouched runs**. Identical flags, numbers, and cited note IDs.

---

## 7. How to Run

### Run Core Pipeline
```bash
python pipeline.py
```
Outputs:
- `output.csv` (19 audited candidate spike rows matching official format)
- `token_cost_log.csv` (per-call token counts and dollar costs)

### Run Interactive Web UI (Stretch Goal)
```bash
streamlit run app.py
```
Provides:
- Anomaly audit dashboard with corridor filters
- Plotly route drill-down charts (Actual vs Trailing 8-week baseline vs Peer baseline)
- Natural language conversational assistant answering questions like *"Why did Chennai-Bangalore spike in March?"*
- Live observability and token cost monitor

---

## 8. 10-Minute Walkthrough Guide for Round 2

When presenting in Round 2, follow this 4-step structure:

1. **Problem Framing (1.5 min)**:
   - Walk through the freight challenge: freight costs quiet creep due to operational disruptions or transporter markups.
   - Show how we cohort shipments into Monday-Sunday weeks and compute weighted cost per tonne-km: $\frac{\sum \text{cost}}{\sum (\text{tonnes} \times \text{km})}$.
2. **Baselines & Spike Identification (2.5 min)**:
   - Explain the 8-week trailing rolling mean (strictly prior weeks, no look-ahead).
   - Explain peer-group average across same route-type excluding the target route.
   - Show how the dual spike filter $\text{diff} \ge 20\%$ isolates 19 clean anomalies from 728 route-weeks.
3. **RAG & Anti-Hallucination Architecture (3.5 min)**:
   - Demonstrate the candidate pre-filter: why we don't blindly call an LLM when 0 notes exist in that region or timeframe (saving 80%+ tokens).
   - Show how ChromaDB retrieves semantic context and Qwen2.5 evaluates causality at `temperature=0.0`.
   - Highlight the programmatic guardrail: how non-disruptive notes (like N006 stable demand or N010 compliance absorbed) are prevented from fabricating justifications.
4. **Reproducibility, Cost & UI Demo (2.5 min)**:
   - Show the 3-run reproducibility diff test passing byte-for-byte.
   - Present the token/cost log (<1,000 tokens total for the full dataset, costing under $0.0001).
   - Demo the Streamlit interactive Q&A assistant answering ad-hoc corridor questions.


### Scope of the Assistant & What It Can/Can't Do Yet

* **What it does today:** You can ask about sudden price jumps on a specific route for a specific week (e.g., *"Why did Chennai-Bangalore spike in late February?"*). The assistant looks up that exact week, checks if costs were unusually high, and tells you whether an actual event (like highway flooding) caused it or if it is an unjustified increase.


* **What it cannot do yet:** It cannot answer broad, big-picture questions across long periods of time (e.g., *"Why did shipping costs across India go up between January and December?"*).


* **Why this limit exists:** The system compares every week's price against a moving 8-week average to spot sudden spikes. When something permanent happens—like a nationwide diesel price hike in May—it stops being a "surprise spike" after about 8 weeks and just becomes the new normal rate.


* **Preventing false blame:** To stop the AI from lazily blaming an old May diesel hike for unrelated price jumps happening 6 months later in December, the assistant only looks for notes within a short time window around the jump.


* **Next steps for production:** Answering long-term, multi-month contract and trend questions will be added in a future update by giving the chatbot a dedicated historical search tool. For now, it stays strictly focused on auditing and explaining weekly price spikes.