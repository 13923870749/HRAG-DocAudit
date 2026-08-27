"""Statistical tests for audit evaluation."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def mcnemar(y_true: List[int], pred_a: List[int], pred_b: List[int]) -> Dict[str, float]:
    """McNemar test for paired binary classifiers (uses continuity correction)."""
    b = sum(1 for t, a, c in zip(y_true, pred_a, pred_b) if a == t and c != t)
    c = sum(1 for t, a, d in zip(y_true, pred_a, pred_b) if a != t and d == t)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "stat": 0.0, "p_value": 1.0}
    stat = (abs(b - c) - 1) ** 2 / n
    # chi-square(1) approximation
    from math import erfc, sqrt

    p = erfc(sqrt(stat / 2))
    return {"b": b, "c": c, "stat": float(stat), "p_value": float(p)}


def bootstrap_ci(
    y_true: List[int],
    y_pred: List[int],
    metric: str = "accuracy",
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    n = len(y_t)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if metric == "accuracy":
            scores.append(float(np.mean(y_t[idx] == y_p[idx])))
        elif metric == "hallucination_rate":
            scores.append(float(np.mean((y_p[idx] == 1) & (y_t[idx] == 0))))
        else:
            scores.append(float(np.mean(y_t[idx] == y_p[idx])))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(np.mean(scores)), float(lo), float(hi)


def bootstrap_ci_bool(values: List[bool], n_boot: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap CI for a boolean vector (e.g., hallucination flags)."""
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=float)
    n = len(arr)
    rng = np.random.default_rng(seed)
    scores = [float(np.mean(arr[rng.integers(0, n, n)])) for _ in range(n_boot)]
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(np.mean(scores)), float(lo), float(hi)


def output_covariance(a: List[int], b: List[int]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    return float(np.cov(np.array(a, dtype=float), np.array(b, dtype=float))[0, 1])


def rule_subspace_stats(res) -> Dict[str, float]:
    """Metrics on items where the rule engine actively routed (used_rule=True)."""
    idx = [i for i, used in enumerate(res.used_rule) if used]
    if not idx:
        return {"n_rule_routed": 0}
    rp = [res.rule_preds[i] for i in idx]
    mp = [res.y_pred[i] for i in idx]
    gp = [res.rag_preds[i] for i in idx]
    hall = [res.hallucinated[i] for i in idx] if res.hallucinated else []
    hall_s = [res.hallucinated_strict[i] for i in idx] if res.hallucinated_strict else []
    out = {
        "n_rule_routed": len(idx),
        "rule_alignment": float(np.mean(np.array(rp) == np.array(mp))),
        "output_covariance_rule_method": output_covariance(rp, mp),
        "conflict_rate_rule_rag": float(np.mean(np.array(rp) != np.array(gp))),
    }
    if hall:
        out["hallucination_rate"] = float(np.mean(hall))
        out["hallucination_rate_lenient"] = out["hallucination_rate"]
    if hall_s:
        out["hallucination_rate_strict"] = float(np.mean(hall_s))
    return out


def conflict_rate(rule_preds: List[int], rag_preds: List[int]) -> float:
    if not rule_preds:
        return 0.0
    return float(np.mean(np.array(rule_preds) != np.array(rag_preds)))
