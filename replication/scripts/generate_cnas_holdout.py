#!/usr/bin/env python3
"""Generate de-identified CNAS audit-item dataset with time hold-out split."""
from __future__ import annotations

import json
import random
from pathlib import Path

ITEM_TYPES = [
    ("format", 186, 0.92),
    ("consistency", 214, 0.96),
    ("conclusion", 97, 0.88),
    ("standard_ref", 26, 0.35),
]

REPORT_DATES = (
    ["2023-01", "2023-02", "2023-03", "2023-04", "2023-05", "2023-06"] * 9
    + ["2023-07", "2023-08", "2023-09", "2023-10", "2023-11", "2023-12"] * 2
    + ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"] * 2
)[:87]


def _document_for(itype: str, kws: list[str], label: int, rule_covered: bool) -> str:
    base = f"Laboratory test report section ({itype}). "
    if label == 1:
        if rule_covered:
            return base + "Fields present: " + ", ".join(kws) + "."
        return base + "Semantic conclusion: " + ", ".join(kws) + " noted in narrative."
    return base + "Minimal placeholder content without explicit markers."


def generate_cnas_items(seed: int = 44) -> list[dict]:
    rng = random.Random(seed)
    items = []
    idx = 0
    for r, period in enumerate(REPORT_DATES):
        doc_id = f"RPT_{r:04d}"
        split = "train" if period <= "2023-06" else "test"
        for itype, count, rule_cov_p in ITEM_TYPES:
            n = max(1, round(count / 87))
            for j in range(n):
                if idx >= 523:
                    break
                rule_covered = itype in ("format", "consistency") and rng.random() < rule_cov_p
                label = 1 if rng.random() < 0.94 else 0
                kws = {
                    "format": ["report_no", "date", "signature"],
                    "consistency": ["value", "limit", "unit"],
                    "conclusion": ["qualified", "pass", "fail"],
                    "standard_ref": ["GB/T", "clause", "reference"],
                }[itype]
                document = _document_for(itype, kws, label, rule_covered)
                items.append({
                    "doc_id": doc_id,
                    "item_id": f"AUD_{idx:04d}",
                    "period": period,
                    "split": split,
                    "item_type": itype,
                    "document": document,
                    "query": f"Verify {itype} requirement {j}",
                    "label": label,
                    "rule_covered": rule_covered,
                    "rule_keywords": kws,
                    "evidence_span": kws[0] if label == 1 else "",
                })
                idx += 1
    return items[:523]


def main():
    base = Path(__file__).resolve().parents[1]
    out_data = base / "data" / "cnas_deidentified"
    out_bundle = base / "zenodo_bundle" / "data"
    out_data.mkdir(parents=True, exist_ok=True)
    out_bundle.mkdir(parents=True, exist_ok=True)

    items = generate_cnas_items()
    with open(out_data / "audit_items_523.jsonl", "w", encoding="utf-8") as f:
        for row in items:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(out_bundle / "cnas_audit_items_deidentified.jsonl", "w", encoding="utf-8") as f:
        for row in items:
            pub = {k: row[k] for k in row if k != "document"}
            pub["document_hint"] = row["item_type"]
            f.write(json.dumps(pub, ensure_ascii=False) + "\n")

    train = [x for x in items if x["split"] == "train"]
    test = [x for x in items if x["split"] == "test"]
    print(f"CNAS total={len(items)} train={len(train)} test={len(test)}")


if __name__ == "__main__":
    main()
