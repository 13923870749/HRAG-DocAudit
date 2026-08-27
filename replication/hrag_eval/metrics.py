"""Evaluation metrics for compliance audit tasks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class EvalResult:
    y_true: List[int]
    y_pred: List[int]
    hallucinated: List[bool] = field(default_factory=list)  # lenient (method-specific)
    hallucinated_strict: List[bool] = field(default_factory=list)  # pred=1 & label=0
    evidence_hit: List[bool] = field(default_factory=list)
    rule_preds: Optional[List[int]] = None
    rag_preds: Optional[List[int]] = None
    used_rule: List[bool] = field(default_factory=list)

    def summary(self, label_names: Optional[List[str]] = None) -> Dict[str, float]:
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        out = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "n": int(len(y_true)),
        }
        if self.hallucinated:
            out["hallucination_rate"] = float(np.mean(self.hallucinated))
            out["hallucination_rate_lenient"] = out["hallucination_rate"]
        if self.hallucinated_strict:
            out["hallucination_rate_strict"] = float(np.mean(self.hallucinated_strict))
        elif len(self.y_pred):
            out["hallucination_rate_strict"] = float(
                np.mean((np.array(self.y_pred) == 1) & (np.array(self.y_true) == 0))
            )
        if self.evidence_hit:
            out["evidence_hit_rate"] = float(np.mean(self.evidence_hit))
        if self.rule_preds is not None and self.rag_preds is not None:
            rp = np.array(self.rule_preds, dtype=float)
            gp = np.array(self.rag_preds, dtype=float)
            if len(rp) == len(gp) and len(rp) > 1:
                out["covariance"] = float(np.cov(rp, gp)[0, 1])
        return out


def compute_metrics(result: EvalResult) -> Dict[str, float]:
    return result.summary()
