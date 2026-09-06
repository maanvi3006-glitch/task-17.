# Task 17 — Customer Segmentation (RFM)
Data Analyst · PlaceMux Phase 1 Industry Immersion

Segments customers by **Recency, Frequency, Monetary (RFM)** value into named,
actionable segments — with a reusable Python pipeline, an automated test
suite, a CLI runner, and a Streamlit dashboard for live demonstration.

## Contents

```
task17_rfm/
├── data/
│   ├── generate_data.py     # builds a realistic, messy sample dataset
│   ├── customers.csv        # generated: 4,150 customers (incl. 150 "ghost"/never-purchased)
│   └── transactions.csv     # generated: ~18.8k transactions, deliberately messy
├── src/
│   └── rfm_pipeline.py      # core pipeline: clean → RFM → score → segment → profile → validate
├── tests/
│   └── test_rfm_pipeline.py # 12 edge-case / failure-handling tests (pytest)
├── outputs/                 # artefacts from the last `run_pipeline.py` run (evidence)
│   ├── rfm_customer_level.csv
│   ├── segment_summary.csv
│   ├── cleaning_report.json
│   ├── segment_distinctness.csv
│   ├── segment_transition_matrix.csv
│   └── validation_report.json
├── run_pipeline.py          # CLI: run the whole pipeline, save every artefact
├── app.py                   # Streamlit dashboard (the live, demoable deliverable)
├── requirements.txt
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt

# 1. (Already done — outputs are pre-generated in data/ and outputs/.)
#    To regenerate fresh sample data:
python data/generate_data.py

# 2. Run the pipeline from the command line (prints a full summary,
#    saves all artefacts to outputs/):
python run_pipeline.py

# 3. Run the test suite (12 edge-case / failure tests):
python -m pytest tests/ -v

# 4. Launch the live dashboard:
streamlit run app.py
```

The dashboard defaults to the bundled sample data so it's demoable
immediately, but also accepts an uploaded CSV (`customer_id, order_date,
amount`, optional `order_id`) so it can be shown working on *any* real data
live, per the Definition of Done.

## How this maps to the marking rubric (100 pts)

**1) Core deliverable — 50 pts**
*"An RFM segmentation with named segments, sizes, value and recommended actions."*
- `src/rfm_pipeline.py::segment_customers` assigns every customer to one of
  10 named, standard RFM segments (Champions, Loyal Customers, Potential
  Loyalists, New Customers, Promising, Need Attention, About To Sleep, At
  Risk, Can't Lose Them, Hibernating, Lost) plus an explicit "No Purchase
  History" bucket for customers with zero clean transactions.
- `profile_segments` reports size, % of base, avg R/F/M, total & average
  value, and % of total revenue per segment.
- `SEGMENT_ACTIONS` pairs **every** segment with a concrete recommended
  action — the "Segments with no attached action" pitfall is structurally
  impossible (`.map(...).fillna(...)` guarantees no segment is left without one).
- Rendered in `app.py`'s **Segments** tab and exported to
  `outputs/segment_summary.csv`.

**2) Real-data quality & correctness — 20 pts**
*"Real inputs at realistic scale, not a toy/happy-path."*
- `data/generate_data.py` builds **~18.8k transactions across 4,150
  customers** with deliberately realistic problems: duplicate rows, missing
  customer_id/amount/date, refunds (negative amounts), zero-amount rows,
  future-dated rows (clock-skew bug), heavily right-skewed spend (a small
  number of "whale" customers), and "ghost" customers who never purchased.
- `load_and_clean_transactions` handles every one of these explicitly and
  reports what was dropped/why (`cleaning_report.json`) — nothing is
  silently discarded.
- Scoring uses **quantile-based** binning (`pd.qcut`), never equal-width, so
  the skewed-spend pitfall named in the brief is directly avoided — see
  `test_skewed_monetary_uses_quantiles_not_equal_width`.

**3) Live verification & evidence — 15 pts**
*"Demonstrated live; real output, not claims."*
- `run_pipeline.py` runs the full pipeline end-to-end on the real dataset
  and both prints a full console summary **and** writes every artefact to
  `outputs/` (already included, from a real run).
- `app.py` is a live Streamlit dashboard: KPIs, a sortable/downloadable
  segment table, revenue/size charts, a recency-vs-frequency scatter, and a
  customer-level explorer — runnable on the bundled data or on a freshly
  uploaded CSV, so it can be demoed interactively, not just described.

**4) Dependency, failure & edge-case handling — 15 pts**
*"Errors handled; hand-offs honoured."*
- Typed exceptions (`RFMDataError`, `RFMComputationError`) for bad input
  (empty data, missing columns, everything filtered out) instead of raw
  tracebacks; `app.py` catches these and shows a clear `st.error` message.
- `_quantile_score` falls back to a rank-based scheme when a column doesn't
  have enough distinct values for 5 clean quantile bins (e.g. many
  customers tied at frequency = 1) — `pd.qcut` would otherwise raise on
  duplicate bin edges.
- Ghost/never-purchased customers are included (not dropped) and explicitly
  labelled `"No Purchase History"` rather than crashing the quantile cut or
  being silently excluded.
- 12 automated tests in `tests/test_rfm_pipeline.py` cover: empty input,
  missing columns, all-rows-invalid, negative/zero amounts, future dates,
  duplicates, missing values, single-order customers, ghost customers,
  skewed monetary, and a full messy-data end-to-end run — all passing.

## Segment stability & distinctness (Definition of Done checks)

- **Distinctness** — `validate_segments` computes average R/F/M per segment
  and flags any two segments whose average value is within 2% of each other
  (i.e. answers the brief's own question: *"Do the segments actually differ
  in behaviour, or only on paper?"*).
- **Stability** — re-scores every customer as of *N* days ago (configurable
  in the dashboard sidebar) and reports the % of customers whose segment is
  unchanged, plus a full transition matrix (`outputs/segment_transition_matrix.csv`,
  and a heatmap in the dashboard's **Validation & stability** tab).

## Notes on design choices (Section 7 — alternative approaches)

- **Rule-based vs. clustering-based segmentation:** this implementation uses
  the industry-standard rule-based RFM (quantile scores → named segment
  rules), which is transparent, explainable to stakeholders, and doesn't
  require choosing a cluster count — appropriate for this use case. A
  clustering-based alternative (e.g. k-means on standardised R/F/M) is
  mentioned in the brief as a valid alternative; swapping it in would only
  require replacing `segment_customers` since RFM computation and
  everything downstream is unaffected.
- **Static vs. rolling re-scoring:** the pipeline is stateless and reference-
  date-driven (`reference_date` parameter), so it can be re-run on a
  schedule against fresh data without any code changes — directly
  addressing the "never refreshing, so segments rot" pitfall.
"# task-17." 
