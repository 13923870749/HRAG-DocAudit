#!/usr/bin/env python3
"""Run Public Track-B on ContractNLI-like data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hrag_eval.core import AuditItem, HRAGEvaluator, HybridRetriever


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


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


def main():
    data_dir = ROOT / "data"
    for name in ("contract-nli", "contractnli_synthetic"):
        test_path = data_dir / name / "test.jsonl"
        if not test_path.exists():
            test_path = data_dir / name / "validation.jsonl"
        if test_path.exists():
            break
    else:
        raise SystemExit("No ContractNLI split found.")

    rows = load_jsonl(test_path)
    items = to_items(rows)
    corpus = list({r["document"] for r in rows})
    ev = HRAGEvaluator(lam=0.6)

    retriever = HybridRetriever(corpus, lam=0.6)
    rel_idx = {doc: i for i, doc in enumerate(corpus)}
    ndcg = retriever.ndcg_at_10(
        [it.query for it in items[:200]],
        [rel_idx.get(it.document, 0) for it in items[:200]],
    )

    results = {"ndcg_at_10": ndcg}
    for m in ["rule", "rag", "ensemble", "hrag"]:
        results[m] = ev.evaluate(items, corpus, method=m).summary()

    # covariance subset
    ens = ev.evaluate(items, corpus, method="ensemble")
    hrg = ev.evaluate(items, corpus, method="hrag")
    results["covariance"] = {"ensemble": ens.summary().get("covariance", 0), "hrag": hrg.summary().get("covariance", 0)}

    out = ROOT / "results" / "contractnli_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
