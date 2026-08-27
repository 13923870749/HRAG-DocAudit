#!/usr/bin/env python3
"""Convert official ContractNLI JSON splits to evaluation jsonl."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "contract-nli" / "raw" / "contract-nli"
OUT = ROOT / "data" / "contract-nli"

SPLIT_MAP = {"train": "train.json", "validation": "dev.json", "test": "test.json"}


def keywords_from_hypothesis(hyp: str) -> list[str]:
    toks = re.findall(r"[a-zA-Z]{4,}", hyp.lower())
    stop = {"shall", "party", "information", "confidential", "agreement", "receiving", "disclosing", "that", "with", "from", "after", "some", "only", "any", "not", "may", "the", "and", "for"}
    return [t for t in toks if t not in stop][:6]


def evidence_text(doc: dict, span_ids: list[int]) -> str:
    parts = []
    spans = doc.get("spans", [])
    text = doc.get("text", "")
    for sid in span_ids:
        if isinstance(sid, int) and 0 <= sid < len(spans):
            s, e = spans[sid]
            parts.append(text[s:e])
    return re.sub(r"\s+", " ", " ".join(parts)).strip()[:120]


def convert_split(split_name: str, src_name: str) -> None:
    src = RAW / src_name
    if not src.exists():
        raise SystemExit(f"Missing {src}")
    data = json.loads(src.read_text(encoding="utf-8"))
    label_defs = data["labels"]
    rows: list[dict] = []

    for doc in data["documents"]:
        doc_id = f"nda_{doc['id']}"
        document = doc.get("text", "")
        ann = doc.get("annotation_sets", [{}])[0].get("annotations", {})
        for hid, meta in label_defs.items():
            hyp = meta["hypothesis"]
            ann_entry = ann.get(hid, {"choice": "NotMentioned", "spans": []})
            choice = ann_entry.get("choice", "NotMentioned")
            label = 1 if choice == "Entailment" else 0
            ev = evidence_text(doc, ann_entry.get("spans", []))
            kws = keywords_from_hypothesis(hyp)
            rule_covered = bool(ev) and any(k in ev.lower() for k in kws[:3])
            rows.append({
                "doc_id": doc_id,
                "item_id": f"{doc_id}_{hid}",
                "document": document,
                "query": hyp,
                "label": label,
                "rule_covered": rule_covered,
                "rule_keywords": kws,
                "evidence_span": ev,
                "nli_choice": choice,
            })

    out_path = OUT / f"{split_name}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"ContractNLI official {split_name}: {len(rows)} items, {len({r['doc_id'] for r in rows})} docs")


def main():
    if not RAW.exists():
        raise SystemExit("Extract contract-nli.zip under data/contract-nli/raw/contract-nli first")
    for split, src in SPLIT_MAP.items():
        convert_split(split, src)


if __name__ == "__main__":
    main()
