"""ELT for evaluation runs: load raw JSONL into SQLite, transform with SQL, gate on data quality.

Stdlib only (sqlite3 + json). Layers: raw_results (one JSON document per row, as loaded) ->
stg_results (SQL view, typed columns via json_extract) -> mart_run_metrics (SQL view, per-run aggregates).
Loads are idempotent: a run is replaced, never duplicated.
"""
from __future__ import annotations
import hashlib, json, sqlite3
from dataclasses import dataclass
from pathlib import Path

RETRIEVAL = ("context_precision", "context_recall", "mrr")
METRICS = ("faithfulness", "answer_relevancy", "answer_correctness", "context_precision", "context_recall", "mrr")

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_results (
  run_id TEXT NOT NULL, row_no INTEGER NOT NULL, row_hash TEXT NOT NULL, doc TEXT NOT NULL,
  loaded_at TEXT NOT NULL DEFAULT (datetime('now')), PRIMARY KEY (run_id, row_no));
CREATE VIEW IF NOT EXISTS stg_results AS
SELECT run_id, row_no,
  json_extract(doc,'$.question') AS question,
  -- unanswerable by dataset convention: no ground truth, or all retrieval metrics absent (undefined without relevant docs)
  CASE WHEN json_extract(doc,'$.ground_truth') IS NULL
        OR (json_extract(doc,'$.context_precision') IS NULL AND json_extract(doc,'$.context_recall') IS NULL
            AND json_extract(doc,'$.mrr') IS NULL) THEN 1 ELSE 0 END AS unanswerable,
  json_array_length(doc,'$.hits') AS n_hits,
  json_extract(doc,'$.hits[0].distance') AS top_distance,
  CAST(json_extract(doc,'$.faithfulness') AS REAL) AS faithfulness,
  CAST(json_extract(doc,'$.answer_relevancy') AS REAL) AS answer_relevancy,
  CAST(json_extract(doc,'$.answer_correctness') AS REAL) AS answer_correctness,
  CAST(json_extract(doc,'$.context_precision') AS REAL) AS context_precision,
  CAST(json_extract(doc,'$.context_recall') AS REAL) AS context_recall,
  CAST(json_extract(doc,'$.mrr') AS REAL) AS mrr,
  CAST(json_extract(doc,'$.hallucination_on_unanswerable_rate') AS REAL) AS halluc_unanswerable
FROM raw_results;
CREATE VIEW IF NOT EXISTS mart_run_metrics AS
SELECT run_id, COUNT(*) AS n_rows, SUM(unanswerable) AS n_unanswerable,
  AVG(faithfulness) AS faithfulness, AVG(answer_relevancy) AS answer_relevancy,
  AVG(answer_correctness) AS answer_correctness,
  AVG(CASE WHEN unanswerable=0 THEN context_precision END) AS context_precision,
  AVG(CASE WHEN unanswerable=0 THEN context_recall END) AS context_recall,
  AVG(CASE WHEN unanswerable=0 THEN mrr END) AS mrr,
  AVG(CASE WHEN unanswerable=1 THEN halluc_unanswerable END) AS halluc_unanswerable
FROM stg_results GROUP BY run_id;
"""

class QualityError(Exception):
    """Raised when a load fails a data-quality check; the transaction is rolled back."""
    def __init__(self, run_id: str, failures: list[str]):
        super().__init__(f"{run_id}: " + "; ".join(failures)); self.run_id, self.failures = run_id, failures

@dataclass(frozen=True)
class LoadResult:
    run_id: str
    rows: int
    replaced: int

def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    con = sqlite3.connect(str(path)); con.row_factory = sqlite3.Row; con.executescript(SCHEMA); return con

def quality_checks(con: sqlite3.Connection, run_id: str, min_rows: int = 1) -> list[str]:
    """Checks on the staged view for one run. Returns failure messages (empty = pass)."""
    f: list[str] = []
    n = con.execute("SELECT COUNT(*) FROM stg_results WHERE run_id=?", (run_id,)).fetchone()[0]
    if n < min_rows: f.append(f"row_count {n} < {min_rows}")
    for m in METRICS:
        # retrieval metrics are undefined for unanswerable questions, so they may be NULL there (never out of range)
        null_ok = "unanswerable=1" if m in RETRIEVAL else "0"
        bad = con.execute(f"SELECT COUNT(*) FROM stg_results WHERE run_id=? AND (({m} IS NULL AND NOT ({null_ok})) OR {m}<0 OR {m}>1)", (run_id,)).fetchone()[0]
        if bad: f.append(f"{m}: {bad} null/out-of-range")
    dup = con.execute("SELECT COUNT(*) FROM (SELECT question FROM stg_results WHERE run_id=? GROUP BY question HAVING COUNT(*)>1)", (run_id,)).fetchone()[0]
    if dup: f.append(f"duplicate questions: {dup}")
    empty = con.execute("SELECT COUNT(*) FROM stg_results WHERE run_id=? AND (question IS NULL OR TRIM(question)='')", (run_id,)).fetchone()[0]
    if empty: f.append(f"empty question: {empty}")
    return f

def load_run(con: sqlite3.Connection, run_id: str, rows: list[dict], min_rows: int = 1) -> LoadResult:
    """Replace one run atomically; roll back and raise QualityError if checks fail."""
    with con:  # one transaction
        replaced = con.execute("DELETE FROM raw_results WHERE run_id=?", (run_id,)).rowcount
        for i, r in enumerate(rows):
            doc = json.dumps(r, sort_keys=True, ensure_ascii=False)
            con.execute("INSERT INTO raw_results(run_id,row_no,row_hash,doc) VALUES (?,?,?,?)",
                        (run_id, i, hashlib.sha256(doc.encode()).hexdigest(), doc))
        fails = quality_checks(con, run_id, min_rows)
        if fails:
            raise QualityError(run_id, fails)
    return LoadResult(run_id, len(rows), replaced)

def load_jsonl(con: sqlite3.Connection, run_id: str, path: str | Path, min_rows: int = 1) -> LoadResult:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return load_run(con, run_id, rows, min_rows)

def run_metrics(con: sqlite3.Connection) -> dict[str, dict]:
    return {r["run_id"]: dict(r) for r in con.execute("SELECT * FROM mart_run_metrics ORDER BY run_id")}

def regressions(con: sqlite3.Connection, base: str, cand: str, max_drop: float = 0.02) -> list[str]:
    """Metrics in the candidate run that dropped more than max_drop vs the base run (SQL-computed)."""
    m = run_metrics(con); b, c = m[base], m[cand]; out = []
    for k in METRICS:
        if b[k] is not None and c[k] is not None and b[k] - c[k] > max_drop:
            out.append(f"{k}: {b[k]:.3f} -> {c[k]:.3f}")
    return out
