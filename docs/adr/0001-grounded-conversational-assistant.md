# Grounded 3-Stage Architecture for Conversational Assistant

To answer conversational inquiries about shipping cost anomalies without hallucination, we decided against naive vector-search RAG and unconstrained LLM chat. Instead, the assistant uses a decoupled 3-stage flow: a lightweight LLM parameter extractor to parse corridor and date into the calendar Monday week format, a deterministic dual lookup querying audited anomalies (`output.csv`) and nominal baselines (`all_metrics_df`), and a constrained generator with strict attribution guardrails. This eliminates false spike claims, guarantees exact metric citing, and delivers zero-waste prompt execution.

## Considered Options

- **Naive Vector Search / Unconstrained LLM RAG**: Embedding output rows and context notes into a vector index and asking the LLM to answer. Rejected because embedding retrieval often mixes adjacent corridor notes, produces non-deterministic cost figures, and risks hallucinating note citations.
- **Hardcoded Keyword Matching**: Simple `if "chennai" in query` rules. Rejected because it fails on arbitrary phrasing, cannot handle nominal baseline comparisons, and lacks natural-language explanation capability.
- **3-Stage Grounded Pipeline (Accepted)**: LLM parameter extractor -> deterministic relational lookup (`output.csv` / `all_metrics_df`) -> grounded factual generation.

## Consequences

- Incomplete queries lacking a corridor or date immediately return a deterministic clarification prompt without invoking synthesis compute.
- Exact numbers and cited note IDs are guaranteed to match the verified analytical and auditor results.
