#!/usr/bin/env python3
"""Generate schema-faithful C3PA-like benchmark (12 CCPA mandates)."""
from __future__ import annotations

import json
import random
from pathlib import Path

MANDATES = [
    ("m01", "contact information for privacy inquiries", ["contact", "email", "phone", "privacy@"]),
    ("m02", "categories of personal information collected", ["collect", "personal information", "category"]),
    ("m03", "sources of personal information", ["source", "third party", "obtain"]),
    ("m04", "business purpose for collection", ["business purpose", "use", "processing"]),
    ("m05", "sale of personal information disclosure", ["sell", "sale", "monetize"]),
    ("m06", "right to know", ["right to know", "access", "request"]),
    ("m07", "right to delete", ["delete", "deletion", "remove"]),
    ("m08", "right to opt-out of sale", ["opt-out", "opt out", "do not sell"]),
    ("m09", "non-discrimination", ["non-discrimination", "discriminate", "retaliation"]),
    ("m10", "retention period", ["retain", "retention", "store"]),
    ("m11", "security measures", ["security", "safeguard", "protect"]),
    ("m12", "effective date or last updated", ["effective date", "last updated", "revision"]),
]


def generate_c3pa_like(n_policies: int = 411, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for p in range(n_policies):
        doc_id = f"policy_{p:04d}"
        org = f"org_{p % 137}"
        compliant = rng.random() < 0.55
        covered_mandates = set()
        parts = ["Privacy policy for users."]
        if compliant:
            n_cov = rng.randint(7, 12)
            covered_mandates = {m[0] for m in rng.sample(MANDATES, n_cov)}
            for mid, desc, kws in MANDATES:
                if mid in covered_mandates:
                    parts.append(f"We disclose {desc}: {', '.join(kws[:2])}.")
        else:
            parts.append("General terms apply without detailed disclosures.")
        document = " ".join(parts)

        for mid, desc, kws in MANDATES:
            present = mid in covered_mandates
            label = 1 if present else 0
            rule_covered = present or rng.random() < 0.35
            evidence = kws[0] if present else ""
            items.append({
                "doc_id": doc_id,
                "org_id": org,
                "item_id": f"{doc_id}_{mid}",
                "document": document,
                "query": f"Does the policy disclose {desc}?",
                "label": label,
                "rule_covered": rule_covered and present,
                "rule_keywords": kws,
                "evidence_span": evidence,
            })
    return items


def main():
    out = Path(__file__).resolve().parents[1] / "data" / "c3pa_synthetic"
    out.mkdir(parents=True, exist_ok=True)
    items = generate_c3pa_like()
    orgs = sorted({x["org_id"] for x in items})
    rng = random.Random(42)
    rng.shuffle(orgs)
    n = len(orgs)
    train_o = set(orgs[: int(n * 0.8)])
    val_o = set(orgs[int(n * 0.8) : int(n * 0.9)])
    test_o = set(orgs[int(n * 0.9) :])

    for split, orgset in [("train", train_o), ("val", val_o), ("test", test_o)]:
        sub = [x for x in items if x["org_id"] in orgset]
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for row in sub:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"C3PA-like {split}: {len(sub)} items")


if __name__ == "__main__":
    main()
