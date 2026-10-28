# rag-vector-eval

A small, offline RAG evaluation demo: Chroma vector retrieval with a local hashed n-gram embedding, a simulated extractive generator, and Ragas/DeepEval-style heuristic metrics (faithfulness, answer relevancy/correctness, context precision/recall, MRR, abstention) with a regression gate. Data is synthetic (fictional "Nordwind Air"). The metrics are heuristic approximations, not the real Ragas or DeepEval; optional adapters load those libraries only if installed.

Setup: `python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`

Run the clean evaluation with:

```sh
python -m rve.cli run --out reports/clean
```

Run the hallucination evaluation and compare both runs with:

```sh
python -m rve.cli run --hallucination .3 --out reports/hallucination
python -m rve.cli compare
```

The reports show retrieval metrics for answerable questions only. Unanswerable questions are evaluated by abstention accuracy and hallucination-on-unanswerable rate.

## License

MIT, see LICENSE.

## ELT + SQL quality gate + hallucination-risk baseline (added 2026-10)

`rve/warehouse.py` loads every `reports/*/results.jsonl` into SQLite in three layers (stdlib only):

- `raw_results`: one JSON document per row, as loaded, with a SHA-256 row hash. Reloading a run replaces it in one transaction.
- `stg_results`: a SQL view that types the fields with `json_extract`. A question counts as unanswerable when it has no ground truth or no retrieval metrics.
- `mart_run_metrics`: a SQL view with per-run aggregates. Retrieval metrics are averaged over answerable questions only.

Data-quality checks run inside the load transaction: row count, metrics within [0, 1], no partial NULLs in retrieval metrics, no duplicate or empty questions. A failing load is rolled back and raises `QualityError`, so a bad batch never reaches the mart. `regressions()` compares two runs in SQL with a max-drop threshold.

`rve/risk.py` is a small pure-Python logistic regression (same features and label work with scikit-learn) that tries to predict low faithfulness from retrieval-side features only. On the bundled data it scores about AUC 0.5, which is the honest and useful result: the injected hallucinations are independent of retrieval quality, so an answer-side check (faithfulness against the retrieved context) is required and retrieval metrics alone cannot replace it.

```sh
python -m rve.elt_cli            # exit code 3 when the hallucination run regresses vs clean
python -m pytest -q tests/test_elt_risk.py   # 13 tests, no extra dependencies
```

Data is synthetic.
