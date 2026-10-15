# FreightTiger Shipping Cost Intelligence
  
> Built by: **Pratham**  

---

### Deliverables Checklist
- [x] **Working Codebase**
- [x] **Contract Output CSV**: [`output.csv`](output.csv)
- [x] **Token & Cost Log**: [`token_cost_log.csv`](token_cost_log.csv)
- [x] **Lightweight eval**: [`eval.py`](eval.py)
- [x] **Reproducibility Check**: [`Reproducibility_check/diff.txt`](reproducibility_check/diff.txt)
- [x] **Interactive Assistant (Stretch Goal)**: Streamlit UI in [`app.py`](app.py)

---

## 2. HOW IT WORKS

It first starts with detecting anomalies statistically. It calculates the route's 8-week rolling baseline and compares it against similar peer routes for that week. It then uses a percentage threshold to decide if a weekly cost rise is an anomaly.

For these detected anomalies, the pipeline checks if there is a note that actually justifies the cost jump. Instead of using a traditional vector database (RAG) approach, I used a two-step setup. The reason is that standard semantic retrieval often leads to hallucinations. For example, if a note mentions general congestion or disruptions on the Ahmedabad–Mumbai corridor, a vector search might accidentally pull it for a Mumbai–Pune anomaly just because the wording sounds similar.

The method I use instead has three stages:

1. **Note Extractor:** An LLM reads every operational context note once and extracts structured details: the route it applies to, valid date ranges, cost direction (increase vs neutral), and estimated impact.
2. **Code Filter:** Simple Python logic matches notes strictly by route and overlapping dates, instantly dropping irrelevant or non-impacting notes with zero LLM calls.
3. **Auditor Agent:** If valid candidate notes remain, an LLM auditor checks the note's context against the actual cost jump to confirm if it genuinely justifies the spike, assigning the note ID or flagging it for human review.


## 3. HOW I BUILT IT

I used an open-source model Qwen2.5-1.5B-Instruct to build this pipeline. The model is run locally on my machine using CPU. I used Antigravity IDE to assist me in writing the code and I used the Matt Pocock Agent Skills to make the codebase cleaner.

## 4. HOW I CONVINCED MYSELF IT WAS RIGHT BEFORE SUBMITTING

I created a golden dataset. This is the dataset that you can see in [`ground_truth_eval.csv`](ground_truth_eval.csv). This contains cases that I personally annotated and expect the pipeline to always mark correctly.It covers Edge cases and normal cases.  Once the pipeline passes the golden dataset completely, I was able to confidently say that it was correct.

## 5. Token Usage & Cost Log

Running over the full dataset produced the following metrics:

| Metric | Measured Value |
| :--- | :--- |
| **Model** | `Qwen/Qwen2.5-1.5B-Instruct` |
| **Total Route-Weeks Audited** | 728 |
| **Cost Rises Detected** | 38 |
| **Justified Verdicts (`No (justified)`)** | 4 (Ahmedabad-Mumbai N002, Chennai-Bangalore N001 × 2, Chennai-Bangalore N003) |
| **Unexplained Verdicts (`Yes`)** | 34 (flagged for human review) |
| **Note Extractor LLM Calls** | **10** (1 call per note in `context_notes.csv`, extracted once & cached to JSON) |
| **Auditor LLM Verification Calls** | **4** (causal verification when candidate notes survived pre-filtering) |
| **Total Active LLM Generation Calls** | **14** (10 extractor + 4 auditor) |
| **Deterministic Zero-Waste Skips** | **34** (0 tokens consumed, LLM bypassed) |
| **Total Pipeline Audit Rows** | **48** (38 cost rise audits + 10 note extractions) |
| **Total Input Tokens** | **6,143** (4,286 extractor + 1,857 auditor) |
| **Total Output Tokens** | **803** (433 extractor + 370 auditor) |
| **Total Tokens Processed** | **6,946** (4,719 extractor + 2,227 auditor) |
| **Self-Hosted Local Cost** | **$0.0000** (100% Free on CPU) |
| **Published Cloud Rate Equivalent ($0.10/1M tokens)** | **$0.000697** (~0.07 cents) |

Full per-call details are stored in [`token_cost_log.csv`](token_cost_log.csv).

---

## 6. REPRODUCIBILITY CHECK

This file contains the results of a reproducibility check across 3 runs. The Pipeline runs deterministically and always gives the same output. I achieved this by setting a seed and using the temperature 0 parameter.

[`reproducibility_check/diff.txt`](reproducibility_check/diff.txt)


---
## 7. HOW TO RUN

### Setup Environment (via uv)
```bash
uv venv
.venv\Scripts\activate   # Windows (source .venv/bin/activate on Unix)
uv pip install -r requirements.txt
```
### Download models locally
```bash
python download_models.py
```
Pre-downloads the open-source `Qwen/Qwen2.5-1.5B-Instruct` model to your local system.

### Configuration

Default paths and calculation parameters are defined in [`src/config.py`](src/config.py):
- **File Paths:** Input shipment records, context notes CSV, metadata cache JSON, and output CSV.
- **Spike Threshold (`spike_threshold_pct`):** Percentage rise threshold vs history or peer routes to flag a cost rise (default `13.0%`).
- **Rolling Window (`trailing_window_weeks`):** Number of prior weeks used to compute trailing historical averages (default `8` weeks).

### Run Core Pipeline
```bash
python pipeline.py
```
Outputs:
- `output.csv`
- `token_cost_log.csv` (per-call token counts and dollar costs)

## 8. Run Interactive Web UI (Stretch Goal)
```bash
streamlit run app.py
```
- Anomaly audit dashboard with corridor filters
- Plotly route drill-down charts (Actual vs Trailing 8-week baseline vs Peer baseline)
- **[BUGGY - *i ran out of time*]** Natural language conversational assistant answering questions like *"Why did Chennai-Bangalore spike in March?"*
- Live observability and token cost monitor

---



## 9. Future Scope for ASSISTANT (i ran out of time)

* **What it does today:** You can ask about sudden price jumps on a specific route for a specific week (e.g., *"Why did Chennai-Bangalore spike in late February?"*). The assistant looks up that exact week, checks if costs were unusually high, and tells you whether an actual event (like highway flooding) caused it or if it is an unjustified increase.


* **What it cannot do yet:** It cannot answer broad, big-picture questions across long periods of time (e.g., *"Why did shipping costs across India go up between January and December?"*).


* **Why this limit exists:** The system compares every week's price against a moving 8-week average to spot sudden spikes. When something permanent happens—like a nationwide diesel price hike in May—it stops being a "surprise spike" after about 8 weeks and just becomes the new normal rate.


* **Preventing false blame:** To stop the AI from lazily blaming an old May diesel hike for unrelated price jumps happening 6 months later in December, the assistant only looks for notes within a short time window around the jump.


* **Next steps for production:** Answering long-term, multi-month contract and trend questions will be added in a future update by giving the chatbot a dedicated historical search tool. For now, it stays strictly focused on auditing and explaining weekly price spikes.