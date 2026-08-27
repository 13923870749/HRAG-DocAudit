#!/usr/bin/env python3
"""Unified public-track experiment runner with stratified stats."""
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


def run_track(name: str, test_path: Path, methods: list[str]) -> dict:
    rows = load_jsonl(test_path)
    items = to_items(rows)
    ev = HRAGEvaluator(lam=0.6)
    report = ev.full_report(items, methods=methods)
    report["dataset"] = name
    report["split"] = test_path.name
    return report


def main():
    data = ROOT / "data"
    tracks = []

    c3pa_test = data / "c3pa" / "test.jsonl"
    if c3pa_test.exists():
        tracks.append(run_track("c3pa", c3pa_test, ["rule", "rag", "rag_rerank", "ensemble", "hrag"]))

    cnli_test = data / "contract-nli" / "test.jsonl"
    if cnli_test.exists():
        tracks.append(run_track("contractnli", cnli_test, ["rule", "rag", "ensemble", "hrag"]))

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    for rep in tracks:
        name = rep["dataset"]
        slim = {m: rep["methods"][m] for m in rep["methods"]}
        (out_dir / f"{name}_results.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
        (out_dir / f"{name}_full_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"=== {name} ===")
        print(json.dumps(slim, indent=2))


if __name__ == "__main__":
    main()
