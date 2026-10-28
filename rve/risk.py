"""Predict hallucination risk per answer from retrieval-side features (no LLM judge needed at runtime).

Pure-Python logistic regression (batch gradient descent, L2) so it runs anywhere; the same features and
label work unchanged with scikit-learn's LogisticRegression if it is installed. Data here is synthetic.
"""
from __future__ import annotations
import math, random
from dataclasses import dataclass

FEATURES = ("top_distance", "context_precision", "context_recall", "mrr", "answer_relevancy")

def features(row: dict) -> list[float]:
    return [float(row.get(k) or 0.0) for k in FEATURES]

def label(row: dict, threshold: float = 0.8) -> int:
    """1 = at-risk answer (faithfulness below threshold)."""
    return int(float(row["faithfulness"]) < threshold)

def _sig(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))

@dataclass
class Model:
    w: list[float]
    b: float
    mu: list[float]
    sd: list[float]
    def proba(self, x: list[float]) -> float:
        z = self.b + sum(wi * (xi - m) / s for wi, xi, m, s in zip(self.w, x, self.mu, self.sd)); return _sig(z)

def fit(X: list[list[float]], y: list[int], lr: float = 0.3, epochs: int = 800, l2: float = 0.01) -> Model:
    if not X or len(X) != len(y): raise ValueError("X and y must be non-empty and the same length")
    d = len(X[0]); mu = [sum(r[j] for r in X) / len(X) for j in range(d)]
    sd = [max(1e-9, math.sqrt(sum((r[j] - mu[j]) ** 2 for r in X) / len(X))) for j in range(d)]
    Z = [[(r[j] - mu[j]) / sd[j] for j in range(d)] for r in X]; w, b, n = [0.0] * d, 0.0, len(X)
    for _ in range(epochs):
        gw, gb = [l2 * wj for wj in w], 0.0
        for z, t in zip(Z, y):
            e = _sig(b + sum(wj * zj for wj, zj in zip(w, z))) - t; gb += e / n
            for j in range(d): gw[j] += e * z[j] / n
        w = [wj - lr * g for wj, g in zip(w, gw)]; b -= lr * gb
    return Model(w, b, mu, sd)

def auc(scores: list[float], y: list[int]) -> float:
    """ROC AUC by pairwise ranking (ties count half). Returns nan if one class is missing."""
    pos = [s for s, t in zip(scores, y) if t]; neg = [s for s, t in zip(scores, y) if not t]
    if not pos or not neg: return float("nan")
    return sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg) / (len(pos) * len(neg))

def evaluate(rows: list[dict], seed: int = 7, test_frac: float = 0.3, threshold: float = 0.8) -> dict:
    """Stratified train/test split, fit, report test AUC and accuracy at 0.5 vs a majority baseline."""
    rng = random.Random(seed); by = {0: [], 1: []}
    for r in rows: by[label(r, threshold)].append(r)
    train, test = [], []
    for k in by:
        g = by[k][:]; rng.shuffle(g); cut = max(1, round(len(g) * test_frac)) if len(g) > 1 else 0
        test += g[:cut]; train += g[cut:]
    m = fit([features(r) for r in train], [label(r, threshold) for r in train])
    ys = [label(r, threshold) for r in test]; ps = [m.proba(features(r)) for r in test]
    acc = sum(int(p >= .5) == t for p, t in zip(ps, ys)) / len(ys)
    maj = max(sum(ys), len(ys) - sum(ys)) / len(ys)
    return {"n_train": len(train), "n_test": len(test), "positives": sum(ys), "auc": auc(ps, ys),
            "accuracy": acc, "majority_baseline": maj, "weights": dict(zip(FEATURES, (round(w, 3) for w in m.w)))}
