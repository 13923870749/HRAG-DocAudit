#!/usr/bin/env python3
"""Generate ContractNLI-like NDA hypothesis audit items."""
from __future__ import annotations

import json
import random
from pathlib import Path

HYPOTHESES = [
    ("nda-1", "All Confidential Information shall be expressly identified by the Disclosing Party.", ["expressly identified", "confidential information"]),
    ("nda-2", "The Receiving Party shall not reverse engineer any objects.", ["reverse engineer", "disassemble"]),
    ("nda-3", "Some obligations may survive termination.", ["survive", "termination"]),
    ("nda-4", "Confidential Information excludes publicly available information.", ["publicly available", "public domain"]),
    ("nda-5", "The Receiving Party may share with employees on need-to-know basis.", ["employees", "need-to-know"]),
]


def generate_contractnli_like(n_docs: int = 607, seed: int = 43) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for d in range(n_docs):
        doc_id = f"nda_{d:04d}"
        base = "NON-DISCLOSURE AGREEMENT between Party A and Party B. "
        present_ids = {hid for hid, _, _ in rng.sample(HYPOTHESES, k=rng.randint(2, 4))}
        clauses = []
        for hid, hyp, kws in HYPOTHESES:
            if hid in present_ids:
                clauses.append(hyp + " Keywords: " + ", ".join(kws))
        document = base + " ".join(clauses)

        for hid, hyp, kws in HYPOTHESES:
            present = hid in present_ids
            label = 1 if present else 0
            rule_covered = present and rng.random() < 0.7
            items.append({
                "doc_id": doc_id,
                "item_id": f"{doc_id}_{hid}",
                "document": document,
                "query": hyp,
                "label": label,
                "rule_covered": rule_covered,
                "rule_keywords": kws,
                "evidence_span": kws[0] if present else "",
            })
    return items


def main():
    out = Path(__file__).resolve().parents[1] / "data" / "contractnli_synthetic"
    out.mkdir(parents=True, exist_ok=True)
    items = generate_contractnli_like()
    rng = random.Random(43)
    rng.shuffle(items)
    n = len(items)
    splits = {
        "train": items[: int(n * 0.7)],
        "validation": items[int(n * 0.7) : int(n * 0.85)],
        "test": items[int(n * 0.85) :],
    }
    for name, sub in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in sub:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"ContractNLI-like {name}: {len(sub)}")


if __name__ == "__main__":
    main()
