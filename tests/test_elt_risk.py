import json, math, pytest
from rve import warehouse as W, risk as R
def row(q, f=1.0, gt="g", d=.2, **k):
    r = {"question": q, "ground_truth": gt, "answer": "a", "hits": [{"id": "x", "text": "t", "distance": d}],
         "faithfulness": f, "answer_relevancy": .5, "answer_correctness": .5, "context_precision": .5, "context_recall": 1.0, "mrr": 1.0,
         "abstention_accuracy": 1.0, "hallucination_on_unanswerable_rate": 0.0}; r.update(k); return r
def test_load_and_mart():
    c = W.connect(); W.load_run(c, "r1", [row("a"), row("b", f=.5)]); m = W.run_metrics(c)["r1"]
    assert m["n_rows"] == 2 and m["faithfulness"] == .75
def test_reload_is_idempotent():
    c = W.connect(); W.load_run(c, "r", [row("a")]); res = W.load_run(c, "r", [row("a"), row("b")])
    assert res.replaced == 1 and W.run_metrics(c)["r"]["n_rows"] == 2
def test_out_of_range_rolls_back():
    c = W.connect(); W.load_run(c, "r", [row("a")])
    with pytest.raises(W.QualityError) as e: W.load_run(c, "r", [row("a", f=1.7)])
    assert "faithfulness" in str(e.value) and W.run_metrics(c)["r"]["n_rows"] == 1
def test_duplicate_and_missing_rejected():
    c = W.connect()
    with pytest.raises(W.QualityError, match="duplicate"): W.load_run(c, "r", [row("a"), row("a")])
    with pytest.raises(W.QualityError, match="null"): W.load_run(c, "s", [row("a", mrr=None)])
    assert W.run_metrics(c) == {}
def test_unanswerable_excluded_from_retrieval_avg():
    c = W.connect(); W.load_run(c, "r", [row("a", context_precision=1.0), row("b", gt=None, context_precision=0.0)])
    m = W.run_metrics(c)["r"]; assert m["context_precision"] == 1.0 and m["n_unanswerable"] == 1
def test_regression_detected():
    c = W.connect(); W.load_run(c, "base", [row("a"), row("b")]); W.load_run(c, "cand", [row("a", f=.5), row("b")])
    assert W.regressions(c, "base", "cand") == ["faithfulness: 1.000 -> 0.750"] and W.regressions(c, "base", "base") == []
def test_unanswerable_may_have_null_retrieval_metrics():
    c = W.connect(); W.load_run(c, "r", [row("a"), row("b", gt=None, context_precision=None, context_recall=None, mrr=None)])
    with pytest.raises(W.QualityError, match="mrr"): W.load_run(c, "s", [row("a", mrr=None)])  # partial null = bad row
    W.load_run(c, "t", [row("a"), row("b", context_precision=None, context_recall=None, mrr=None)])  # all-null = unanswerable
    assert W.run_metrics(c)["t"]["n_unanswerable"] == 1
def test_min_rows():
    with pytest.raises(W.QualityError, match="row_count"): W.load_run(W.connect(), "r", [], min_rows=1)
def test_auc_edges():
    assert R.auc([.9, .1], [1, 0]) == 1 and R.auc([.1, .9], [1, 0]) == 0 and R.auc([.5, .5], [1, 0]) == .5
    assert math.isnan(R.auc([.3], [1]))
def test_fit_separable():
    X = [[i / 10] for i in range(20)]; y = [int(i >= 10) for i in range(20)]; m = R.fit(X, y)
    assert m.proba([1.9]) > .9 and m.proba([0.0]) < .1
def test_fit_validates():
    with pytest.raises(ValueError): R.fit([], [])
def test_evaluate_on_synthetic():
    rows = [row(f"q{i}", f=.4 if i % 3 == 0 else 1.0, d=.9 if i % 3 == 0 else .2, context_precision=.1 if i % 3 == 0 else .9) for i in range(30)]
    r = R.evaluate(rows); assert r["auc"] == 1.0 and r["accuracy"] >= r["majority_baseline"]
def test_cli_on_bundled_reports(capsys):
    from rve import elt_cli
    code = elt_cli.main(["--reports", "reports", "--max-drop", "0.02"]); out = capsys.readouterr().out
    assert "risk model" in out and "regressions: ['faithfulness" in out and code == 3
