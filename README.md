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
