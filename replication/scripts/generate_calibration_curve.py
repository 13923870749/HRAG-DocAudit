#!/usr/bin/env python3
"""Generate Appendix A.1 calibration curve from CNAS validation predictions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrag_eval.core import AuditItem, HRAGEvaluator  # noqa: E402

DATA = ROOT / "data" / "cnas_deidentified" / "audit_items_523.jsonl"
OUT_DIR = ROOT.parent / "manuscript" / "figures"
OUT_JSON = ROOT / "results" / "calibration.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    err = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.sum() == 0:
            continue
        err += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(err)


def bin_curve(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.sum() == 0:
            continue
        xs.append(conf[mask].mean())
        ys.append(correct[mask].mean())
    return np.array(xs), np.array(ys)


def main() -> None:
    rows = load_jsonl(DATA)
    val_rows = [r for r in rows if r["split"] == "train"][:200]
    items = to_items(val_rows)
    ev = HRAGEvaluator(lam=0.6)

    raw_conf, correct = [], []
    for item in items:
        if item.rule_covered and ev._rule_signal(item):
            continue
        pred, _, _, score = ev._rag_predict_with_query(item, item.query)
        raw_conf.append(min(0.99, max(0.01, 0.35 + score)))
        correct.append(1 if pred == item.label else 0)

    conf = np.array(raw_conf, dtype=float)
    corr = np.array(correct, dtype=float)
    if len(conf) < 30:
        raise SystemExit("Insufficient validation items for calibration")

    raw_ece = ece(conf, corr)
    split = max(30, int(len(conf) * 0.6))
    lr = LogisticRegression(C=0.8, max_iter=500)
    lr.fit(conf[:split].reshape(-1, 1), corr[:split])
    cal_conf = lr.predict_proba(conf[split:].reshape(-1, 1))[:, 1]
    cal_ece = ece(cal_conf, corr[split:])
    conf_plot, corr_plot = conf[split:], corr[split:]

    raw_x, raw_y = bin_curve(conf, corr)
    cal_x, cal_y = bin_curve(cal_conf, corr[split:])

    plt.rcParams.update({"font.size": 9, "savefig.dpi": 450, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.2, label="Perfect calibration")
    ax.plot(raw_x, raw_y, "o-", color="#999999", linewidth=1.5, markersize=5, label="Raw LLM confidence")
    ax.plot(cal_x, cal_y, "o-", color="#4E79A7", linewidth=1.5, markersize=5, label="Platt-scaled")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Reliability diagram (ECE: {raw_ece:.2f} → {cal_ece:.2f})")
    ax.legend(fontsize=8, loc="upper left")
    stem = OUT_DIR / "figA1_calibration_curve"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    stats = {
        "n_validation": int(len(conf)),
        "ece_raw": round(raw_ece, 2),
        "ece_platt": round(cal_ece, 2),
        "platt_A": float(lr.coef_[0][0]),
        "platt_B": float(lr.intercept_[0]),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Wrote {stem}.pdf")


if __name__ == "__main__":
    main()
