#!/usr/bin/env python3
"""Generate supplementary hallucination table (lenient vs strict) from full reports."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrag_eval.core import AuditItem, HRAGEvaluator

RES = ROOT / "results"
OUT = ROOT.parent / "manuscript" / "sections" / "appendix_hallucination.tex"


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def load_items(name: str) -> list[AuditItem]:
    if name == "c3pa":
        path = ROOT / "data" / "c3pa" / "test.jsonl"
    else:
        path = ROOT / "data" / "contract-nli" / "test.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
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


def rule_routed_subset(items: list[AuditItem]) -> list[AuditItem]:
    ev = HRAGEvaluator()
    return [it for it in items if ev._rule_signal(it)]


def row(dataset: str, scope: str, n: int, entries: list[tuple[str, dict]]) -> list[str]:
    lines = []
    for i, (label, m) in enumerate(entries):
        ds = dataset if i == 0 else ""
        sc = scope if i == 0 else ""
        nn = str(n) if i == 0 else ""
        lenient = pct(m.get("hallucination_rate_lenient", m.get("hallucination_rate", 0)))
        strict = pct(m.get("hallucination_rate_strict", 0))
        lines.append(f"{ds} & {sc} & {nn} & {label} & {lenient} & {strict} \\\\")
    return lines


def metrics_from_summary(s: dict) -> dict:
    return s


def main():
    c3pa = json.loads((RES / "c3pa_full_report.json").read_text(encoding="utf-8"))
    cnli = json.loads((RES / "contractnli_full_report.json").read_text(encoding="utf-8"))
    c3pa_items = load_items("c3pa")
    cnli_items = load_items("contractnli")
    ev = HRAGEvaluator()
    c3pa_routed = rule_routed_subset(c3pa_items)
    cnli_routed = rule_routed_subset(cnli_items)
    rag_c3pa_routed = ev.evaluate(c3pa_routed, "rag").summary()
    rag_cnli_routed = ev.evaluate(cnli_routed, "rag").summary()

    body = []
    m = c3pa["methods"]
    body.extend(
        row(
            "C3PA",
            "All items",
            c3pa["n"],
            [
                ("RAG-Only", m["rag"]),
                ("OR-Ensemble", m["ensemble"]),
                (r"\HRAG{}", m["hrag"]),
            ],
        )
    )
    body.append("\\midrule")
    rs = c3pa["strata"]["rule_amenable"]
    body.extend(
        row(
            "",
            "Rule-amenable",
            rs["n"],
            [
                ("RAG-Only", rs["rag"]),
                ("OR-Ensemble", rs["ensemble"]),
                (r"\HRAG{}", rs["hrag"]),
            ],
        )
    )
    body.append("\\midrule")
    ca = c3pa["cascade_analysis"]
    body.extend(
        row(
            "",
            "Rule-routed",
            ca["hrag"]["n_rule_routed"],
            [
                ("RAG-Only", rag_c3pa_routed),
                ("OR-Ensemble", ca["ensemble"]),
                (r"\HRAG{}", ca["hrag"]),
            ],
        )
    )
    body.append("\\midrule")
    m2 = cnli["methods"]
    body.extend(
        row(
            "ContractNLI",
            "All items",
            cnli["n"],
            [
                ("RAG-Only", m2["rag"]),
                ("OR-Ensemble", m2["ensemble"]),
                (r"\HRAG{}", m2["hrag"]),
            ],
        )
    )
    body.append("\\midrule")
    rs2 = cnli["strata"]["rule_amenable"]
    body.extend(
        row(
            "",
            "Rule-amenable",
            rs2["n"],
            [
                ("RAG-Only", rs2["rag"]),
                ("OR-Ensemble", rs2["ensemble"]),
                (r"\HRAG{}", rs2["hrag"]),
            ],
        )
    )
    body.append("\\midrule")
    ca2 = cnli["cascade_analysis"]
    body.extend(
        row(
            "",
            "Rule-routed",
            ca2["hrag"]["n_rule_routed"],
            [
                ("RAG-Only", rag_cnli_routed),
                ("OR-Ensemble", ca2["ensemble"]),
                (r"\HRAG{}", ca2["hrag"]),
            ],
        )
    )

    tex = r"""\section{Hallucination metric sensitivity (lenient vs strict)}
\label{app:hallucination}

Main tables report the \textbf{lenient} hallucination rate: an unsupported positive (\texttt{pred=1}, \texttt{label=0}) counts as hallucination only when the method lacks valid retrieval evidence; rule-routed cascade decisions are treated as auditable rule outputs rather than generative claims. The \textbf{strict} rate counts every false positive regardless of routing path.

\begin{table}[h]
\centering
\caption{Lenient vs strict hallucination rates (\%, Tier-1 replication)}
\label{tab:hallucination_sensitivity}
\begin{tabular}{llrlcc}
\toprule
Dataset & Scope & $n$ & Method & Lenient & Strict \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}

Strict counting exposes \emph{rule-engine false positives} on rule-routed items (e.g., 7.5\% on C3PA rule-amenable for \HRAG{} and OR-Ensemble). Lenient counting isolates unsupported \emph{generative} claims: \HRAG{} reports 0\% on rule-routed items while RAG-Only remains high on semantic-only subsets. Both metrics should be reported together in compliance auditing.
"""
    OUT.write_text(tex, encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
