#!/usr/bin/env python3
"""Convert official C3PA repo (Annotations/Htmls/Crawl) to evaluation jsonl splits."""
from __future__ import annotations

import csv
import json
import random
import re
from html import unescape
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
C3PA = ROOT / "data" / "c3pa"
OUT = C3PA  # write train/val/test.jsonl beside raw files

MANDATES = [
    ("m01", "Categories of Personal Information Collected", ["collected", "categories", "personal information"]),
    ("m02", "Categories of Personal Information Shared / Disclosed", ["shared", "disclosed", "third party"]),
    ("m03", "Categories of Personal Information Sold", ["sold", "sale", "sell"]),
    ("m04", "Methods to exercise rights", ["request", "access", "call us", "submit"]),
    ("m05", "Description of Right to Know PI Collected", ["right to know", "access", "collected"]),
    ("m06", "Description of Right to Opt-out of sale of PI", ["opt-out", "opt out", "do not sell"]),
    ("m07", "Updated Privacy Policy", ["updated", "last updated", "effective date"]),
    ("m08", "Description of Right to Delete", ["delete", "deletion", "remove"]),
    ("m09", "Description of Right to Know PI sold / shared", ["sold", "shared", "disclosed"]),
    ("m10", "Description of Right to Non-discrimination on exercising rights", ["non-discrimination", "discriminate", "discriminatory"]),
    ("m11", "Description of Right to Correct Information", ["correct", "correction", "inaccurate"]),
    ("m12", "Description of Right to Limit use of PI", ["limit", "sensitive personal information", "limit the use"]),
]

TAG_RE = re.compile(r"<[^>]+>")


def html_to_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text = unescape(TAG_RE.sub(" ", raw))
    return re.sub(r"\s+", " ", text).strip()


def org_from_link(link: str) -> str:
    host = urlparse(link).netloc.lower().replace("www.", "")
    return host or "unknown"


def load_crawl(source: str) -> dict[int, dict]:
    crawl_path = C3PA / "Crawl" / f"{source.lower()}.csv"
    rows: dict[int, dict] = {}
    with crawl_path.open(newline="", encoding="utf-8", errors="ignore") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            rows[i] = row
    return rows


def policy_labels(ann_path: Path) -> dict[str, str]:
    """Map mandate label -> evidence snippet."""
    out: dict[str, str] = {}
    with ann_path.open(newline="", encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            lab = row.get("Label", "").strip()
            if not lab or lab == "Others":
                continue
            txt = row.get("Text", "").strip()
            if lab not in out and txt:
                out[lab] = re.sub(r"\s+", " ", txt)[:120]
    return out


def keyword_hits(document: str, keywords: list[str]) -> int:
    low = document.lower()
    return sum(1 for k in keywords if k.lower() in low)


def build_items() -> list[dict]:
    items: list[dict] = []
    for source in ("WS", "DB"):
        crawl = load_crawl(source)
        ann_dir = C3PA / "Annotations" / source
        html_dir = C3PA / "Htmls" / source
        for ann_path in sorted(ann_dir.glob("*.csv"), key=lambda p: int(p.stem)):
            idx = int(ann_path.stem)
            html_path = html_dir / f"{idx}.html"
            if not html_path.exists():
                continue
            document = html_to_text(html_path)
            crawl_row = crawl.get(idx, {})
            link = crawl_row.get("Link", "")
            org_id = org_from_link(link)
            doc_id = f"{source}_{idx:04d}"
            present = policy_labels(ann_path)
            match_blob = " ".join(
                str(crawl_row.get(k, "")) for k in ("Textmatch_P", "Textmatch_S", "Textmatch_PP", "Link_Match")
            ).lower()

            for mid, mandate_label, kws in MANDATES:
                label = 1 if mandate_label in present else 0
                # Rule-amenable when crawl or document exposes keyword patterns (independent of label).
                rule_covered = keyword_hits(match_blob, kws) >= 1 or keyword_hits(document, kws) >= 1
                items.append({
                    "doc_id": doc_id,
                    "org_id": org_id,
                    "item_id": f"{doc_id}_{mid}",
                    "document": document,
                    "query": f"Does the policy satisfy: {mandate_label}?",
                    "label": label,
                    "rule_covered": bool(rule_covered),
                    "rule_keywords": kws,
                    "evidence_span": present.get(mandate_label, "")[:80],
                })
    return items


def write_splits(items: list[dict], seed: int = 42) -> None:
    orgs = sorted({x["org_id"] for x in items})
    rng = random.Random(seed)
    rng.shuffle(orgs)
    n = len(orgs)
    train_o = set(orgs[: int(n * 0.8)])
    val_o = set(orgs[int(n * 0.8) : int(n * 0.9)])
    test_o = set(orgs[int(n * 0.9) :])
    for split, orgset in [("train", train_o), ("val", val_o), ("test", test_o)]:
        sub = [x for x in items if x["org_id"] in orgset]
        path = OUT / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in sub:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"C3PA official {split}: {len(sub)} items, {len({x['doc_id'] for x in sub})} docs")


def main():
    if not (C3PA / "Annotations").exists():
        raise SystemExit("Official C3PA not found under data/c3pa")
    items = build_items()
    write_splits(items)
    print(f"Total mandate-items: {len(items)}")


if __name__ == "__main__":
    main()
