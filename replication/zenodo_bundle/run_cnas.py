#!/usr/bin/env python3
"""Run Private Track on CNAS de-identified hold-out split."""
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
    path = ROOT / "data" / "cnas_deidentified" / "audit_items_523.jsonl"
    if not path.exists():
        raise SystemExit("Run scripts/generate_cnas_holdout.py first")

    all_rows = load_jsonl(path)
    train_rows = [r for r in all_rows if r["split"] == "train"]
    test_rows = [r for r in all_rows if r["split"] == "test"]

    ev = HRAGEvaluator(lam=0.6)
    corpus = list({r["document"] for r in all_rows})

    out = {"holdout": {"train_n": len(train_rows), "test_n": len(test_rows), "split_rule": "period<=2023-06 train, else test"}}
    for split_name, rows in [("full", all_rows), ("holdout_test", test_rows)]:
        items = to_items(rows)
        block = {}
        for m in ["rule", "rag", "ensemble", "hrag"]:
            block[m] = ev.evaluate(items, corpus, method=m).summary()
        ens = ev.evaluate(items, corpus, method="ensemble")
        hrg = ev.evaluate(items, corpus, method="hrag")
        block["covariance"] = {
            "ensemble": ens.summary().get("covariance", 0),
            "hrag": hrg.summary().get("covariance", 0),
        }
        out[split_name] = block

    # ablation on full set
    ablation = {}
    for variant, method in [
        ("full_hrag", "hrag"),
        ("w/o_priority", "ensemble"),
        ("w/o_rules", "rag"),
    ]:
        ablation[variant] = ev.evaluate(to_items(all_rows), corpus, method=method).summary()
    out["ablation"] = ablation

    result_path = ROOT / "results" / "cnas_results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
