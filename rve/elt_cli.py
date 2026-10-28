"""python -m rve.elt_cli [--db eval.db] -> load reports/*/results.jsonl, print run metrics, regressions, risk model."""
import argparse, glob, json, os, sys
from . import warehouse as W, risk as R
def main(argv=None):
    a = argparse.ArgumentParser(); a.add_argument("--db", default=":memory:"); a.add_argument("--reports", default="reports")
    a.add_argument("--base", default="clean"); a.add_argument("--cand", default="hallucination"); a.add_argument("--max-drop", type=float, default=0.02)
    o = a.parse_args(argv); con = W.connect(o.db)
    for p in sorted(glob.glob(os.path.join(o.reports, "*", "results.jsonl"))):
        print("loaded", W.load_jsonl(con, os.path.basename(os.path.dirname(p)), p))
    print(json.dumps(W.run_metrics(con), indent=1, default=str))
    reg = W.regressions(con, o.base, o.cand, o.max_drop) if {o.base, o.cand} <= set(W.run_metrics(con)) else []
    print("regressions:", reg or "none")
    rows = [dict(r) for r in con.execute("SELECT * FROM stg_results WHERE unanswerable=0 AND run_id IN (?,?)", (o.base, o.cand))]
    print("risk model:", json.dumps(R.evaluate(rows), default=str))
    return 3 if reg else 0
if __name__ == "__main__": sys.exit(main())
