#!/usr/bin/env python3
"""Run Self-RAG / ReAct on C3PA test split (Tier-1 proxy only; no CNAS transfer)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hrag_eval.core import AuditItem, HRAGEvaluator  # noqa: E402

C3PA = ROOT / "data" / "c3pa" / "test.jsonl"
OUT = ROOT / "results" / "tier2_baselines.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def to_items(rows: list[dict]) -> list[AuditItem]:
    return [
        AuditItem(
            doc_id=r["doc_id"],
            item_id=r["item_id"],
            document=r["document"],
            query=r["query"],
            label=int(r["label"]),
            rule_covered=bool(r["rule_covered"]),
            rule_keywords=r["rule_keywords"],
            evidence_span=r.get("evidence_span", ""),
        )
        for r in rows
    ]


def pct_rate(x: float) -> float:
    return round(100.0 * x, 1)


def hitl_fraction(res, confidences: list[float]) -> float:
    review = sum(1 for used, conf in zip(res.used_rule, confidences) if not used and conf < 0.7)
    return review / len(res.y_true) if res.y_true else 0.0


def collect_confidences(ev: HRAGEvaluator, items: list[AuditItem], method: str) -> list[float]:
    confs: list[float] = []
    for item in items:
        if method == "self_rag":
            _, _, _, conf, _ = ev._self_rag_predict(item)
        elif method == "react":
            _, _, _, conf = ev._react_predict(item)
        else:
            _, _, _, conf = ev._rag_predict_with_query(item, item.query)
        confs.append(min(0.99, conf))
    return confs


def self_rag_rejection_stats(ev: HRAGEvaluator, items: list[AuditItem]) -> dict[str, float]:
    """Reflection diagnostics: rejections among initial RAG positives and over all items."""
    n = len(items)
    if n == 0:
        return {
            "reflection_reject_rate_pct": 0.0,
            "rejected_correct_pct": 0.0,
            "reject_among_initial_positives_pct": 0.0,
        }
    rejected = rejected_correct = initial_pos = 0
    for item in items:
        initial_pred, _, _, _ = ev._rag_predict_with_query(item, item.query)
        _, _, _, _, was_rejected = ev._self_rag_predict(item)
        if initial_pred == 1:
            initial_pos += 1
            if was_rejected:
                rejected += 1
                if item.label == 1:
                    rejected_correct += 1
    return {
        "reflection_reject_rate_pct": pct_rate(rejected / n),
        "rejected_correct_pct": pct_rate(rejected_correct / n),
        "reject_among_initial_positives_pct": pct_rate(rejected / initial_pos if initial_pos else 0),
    }


def main() -> None:
    if not C3PA.exists():
        raise SystemExit("Missing C3PA test split")

    items = to_items(load_jsonl(C3PA))
    ev = HRAGEvaluator(lam=0.6)

    proxy: dict[str, dict] = {}
    for method in ("rag", "self_rag", "react"):
        res = ev.evaluate(items, method=method)
        confs = collect_confidences(ev, items, method)
        summ = res.summary()
        summ["hitl_fraction"] = hitl_fraction(res, confs)
        proxy[method] = summ

    self_rag_diag = self_rag_rejection_stats(ev, items)

    out = {
        "proxy_dataset": "C3PA official test (n=456)",
        "retrieval_config": {
            "lambda": ev.lam,
            "top_M": ev.TOP_M,
            "document_scoped": True,
            "bm25_k1": 1.5,
            "bm25_b": 0.75,
            "self_rag_reflection_threshold": ev.REFLECTION_THRESHOLD,
            "react_max_steps": ev.MAX_REACT_STEPS,
        },
        "proxy": {
            k: {
                "accuracy_pct": pct_rate(v["accuracy"]),
                "hallucination_pct": pct_rate(v["hallucination_rate"]),
                "hitl_pct": pct_rate(v["hitl_fraction"]),
            }
            for k, v in proxy.items()
        },
        "self_rag_diagnostics": self_rag_diag,
        "note": "C3PA Tier-1 proxy measurements; excluded from CNAS Table 5.",
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
