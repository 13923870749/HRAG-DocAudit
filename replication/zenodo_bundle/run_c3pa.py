#!/usr/bin/env python3
"""Run Public Track-A on C3PA-like data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hrag_eval.core import AuditItem, HRAGEvaluator


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
    for name in ("c3pa", "c3pa_synthetic"):
        test_path = data_dir / name / "test.jsonl"
        if test_path.exists():
            break
    else:
        raise SystemExit("No C3PA test split found. Run generate_c3pa_synthetic.py or download_datasets.sh")

    rows = load_jsonl(test_path)
    items = to_items(rows)
    corpus = list({r["document"] for r in rows})
    ev = HRAGEvaluator(lam=0.6)

    methods = ["rule", "rag", "rag_rerank", "ensemble", "hrag"]
    results = {}
    for m in methods:
        res = ev.evaluate(items, corpus, method=m)
        results[m] = res.summary()

    out = ROOT / "results" / "c3pa_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
