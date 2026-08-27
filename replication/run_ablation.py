#!/usr/bin/env python3
"""Hybrid-retrieval lambda ablation on validation splits."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hrag_eval.core import AuditItem, HRAGEvaluator

LAMBDAS = [0.0, 0.3, 0.6, 0.9, 1.0]


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


def sweep(name: str, val_path: Path) -> dict:
    items = to_items(load_jsonl(val_path))
    rows = []
    for lam in LAMBDAS:
        ev = HRAGEvaluator(lam=lam)
        rows.append(
            {
                "lambda": lam,
                "ndcg_at_10": ev.ndcg_at_10(items),
            }
        )
    best = max(rows, key=lambda r: r["ndcg_at_10"])
    return {
        "dataset": name,
        "split": val_path.name,
        "n": len(items),
        "sweep": rows,
        "best_lambda_ndcg": best["lambda"],
        "best_ndcg_at_10": best["ndcg_at_10"],
    }


def main():
    data = ROOT / "data"
    out: dict = {"lambdas": LAMBDAS, "tracks": []}

    c3pa_val = data / "c3pa" / "val.jsonl"
    if c3pa_val.exists():
        out["tracks"].append(sweep("c3pa", c3pa_val))

    cnli_val = data / "contract-nli" / "validation.jsonl"
    if cnli_val.exists():
        out["tracks"].append(sweep("contractnli", cnli_val))

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "lambda_ablation.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
