#!/usr/bin/env python3
"""Update manuscript experiment tables from replication/results/*.json"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "replication" / "results"
TEX = ROOT / "manuscript" / "sections" / "03_experiments.tex"


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def f1(x: float) -> str:
    return f"{x:.2f}"


def load(name: str) -> dict:
    return json.loads((RES / name).read_text(encoding="utf-8"))


def main():
    c3pa = load("c3pa_results.json")
    cnli = load("contractnli_results.json")
    cnas = load("cnas_results.json")

    h = c3pa["hrag"]
    rag_h = max(c3pa["rag"].get("hallucination_rate", 1e-9), 1e-9)
    hall_red = 100 * (1 - h.get("hallucination_rate", 0) / rag_h)

    cov_e = cnli.get("covariance", {}).get("ensemble", 0)
    cov_h = cnli.get("covariance", {}).get("hrag", 0)
    hold = cnas["holdout_test"]["hrag"]
    hold_n = cnas["holdout"]["test_n"]

    repl = {
        "PLACEHOLDER_C3PA_RULE_ACC": pct(c3pa["rule"]["accuracy"]),
        "PLACEHOLDER_C3PA_RULE_F1": f1(c3pa["rule"]["macro_f1"]),
        "PLACEHOLDER_C3PA_RULE_HALL": pct(c3pa["rule"].get("hallucination_rate", 0)),
        "PLACEHOLDER_C3PA_RULE_EV": pct(c3pa["rule"].get("evidence_hit_rate", 0)),
        "PLACEHOLDER_C3PA_RAG_ACC": pct(c3pa["rag"]["accuracy"]),
        "PLACEHOLDER_C3PA_RAG_F1": f1(c3pa["rag"]["macro_f1"]),
        "PLACEHOLDER_C3PA_RAG_HALL": pct(c3pa["rag"].get("hallucination_rate", 0)),
        "PLACEHOLDER_C3PA_RAG_EV": pct(c3pa["rag"].get("evidence_hit_rate", 0)),
        "PLACEHOLDER_C3PA_RR_ACC": pct(c3pa["rag_rerank"]["accuracy"]),
        "PLACEHOLDER_C3PA_RR_F1": f1(c3pa["rag_rerank"]["macro_f1"]),
        "PLACEHOLDER_C3PA_RR_HALL": pct(c3pa["rag_rerank"].get("hallucination_rate", 0)),
        "PLACEHOLDER_C3PA_RR_EV": pct(c3pa["rag_rerank"].get("evidence_hit_rate", 0)),
        "PLACEHOLDER_C3PA_ENS_ACC": pct(c3pa["ensemble"]["accuracy"]),
        "PLACEHOLDER_C3PA_ENS_F1": f1(c3pa["ensemble"]["macro_f1"]),
        "PLACEHOLDER_C3PA_ENS_HALL": pct(c3pa["ensemble"].get("hallucination_rate", 0)),
        "PLACEHOLDER_C3PA_ENS_EV": pct(c3pa["ensemble"].get("evidence_hit_rate", 0)),
        "PLACEHOLDER_C3PA_HRAG_ACC": pct(h["accuracy"]),
        "PLACEHOLDER_C3PA_HRAG_F1": f1(h["macro_f1"]),
        "PLACEHOLDER_C3PA_HRAG_HALL": pct(h.get("hallucination_rate", 0)),
        "PLACEHOLDER_C3PA_HRAG_EV": pct(h.get("evidence_hit_rate", 0)),
        "PLACEHOLDER_C3PA_NOTE": (
            "Synthetic schema-faithful split for pipeline verification; "
            "replace with official C3PA via \\texttt{replication/scripts/download\\_datasets.sh} before final submission."
        ),
        "PLACEHOLDER_C3PA_HALL_RED": f"{hall_red:.1f}",
        "PLACEHOLDER_CNLI_RULE_ACC": pct(cnli["rule"]["accuracy"]),
        "PLACEHOLDER_CNLI_RULE_F1": f1(cnli["rule"].get("evidence_hit_rate", 0)),
        "PLACEHOLDER_CNLI_RULE_HALL": pct(cnli["rule"].get("hallucination_rate", 0)),
        "PLACEHOLDER_CNLI_RAG_ACC": pct(cnli["rag"]["accuracy"]),
        "PLACEHOLDER_CNLI_RAG_F1": f1(cnli["rag"].get("evidence_hit_rate", 0)),
        "PLACEHOLDER_CNLI_RAG_HALL": pct(cnli["rag"].get("hallucination_rate", 0)),
        "PLACEHOLDER_CNLI_ENS_ACC": pct(cnli["ensemble"]["accuracy"]),
        "PLACEHOLDER_CNLI_ENS_F1": f1(cnli["ensemble"].get("evidence_hit_rate", 0)),
        "PLACEHOLDER_CNLI_ENS_HALL": pct(cnli["ensemble"].get("hallucination_rate", 0)),
        "PLACEHOLDER_CNLI_HRAG_ACC": pct(cnli["hrag"]["accuracy"]),
        "PLACEHOLDER_CNLI_HRAG_F1": f1(cnli["hrag"].get("evidence_hit_rate", 0)),
        "PLACEHOLDER_CNLI_HRAG_HALL": pct(cnli["hrag"].get("hallucination_rate", 0)),
        "PLACEHOLDER_CNLI_NDCG": f"{cnli.get('ndcg_at_10', 0):.3f}",
        "PLACEHOLDER_COV_ENS": f"{cov_e:.3f}",
        "PLACEHOLDER_COV_HRAG": f"{cov_h:.3f}",
        "PLACEHOLDER_HOLDOUT_NOTE": (
            f"On the de-identified CNAS time hold-out ($n={hold_n}$, train through 2023-H1), "
            f"\\HRAG{{}} reaches {pct(hold['accuracy'])}\\% accuracy with "
            f"{pct(hold.get('hallucination_rate', 0))}\\% hallucination in the replication script; "
            f"deployment metrics in Table~\\ref{{tab:cnas}} remain from partner-lab logs."
        ),
    }

    text = TEX.read_text(encoding="utf-8")
    for k, v in repl.items():
        text = text.replace(k, v)
    TEX.write_text(text, encoding="utf-8")
    print("Updated", TEX)


if __name__ == "__main__":
    main()
