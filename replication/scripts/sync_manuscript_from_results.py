#!/usr/bin/env python3
"""Sync CNAS deployment table and ECE text from replication results."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPL = Path(__file__).resolve().parents[1]
EAai = REPL.parent
TIER2 = REPL / "results" / "tier2_baselines.json"
CAL = REPL / "results" / "calibration.json"
EXP_TEX = EAai / "manuscript" / "sections" / "03_experiments.tex"
METHOD_TEX = EAai / "manuscript" / "sections" / "02_methodology.tex"
CAL_TEX = EAai / "manuscript" / "sections" / "appendix_calibration.tex"


def fmt_row(name: str, m: dict, bold: bool = False) -> str:
    label = f"\\textbf{{\\HRAG{{}}}}" if bold else name.replace("HRAG", "\\HRAG{}")
    if name == "HRAG":
        label = "\\textbf{\\HRAG{}}"
    line = f"  & {label} & {m['accuracy']:.1f} & {m['hallucination']:.1f} & {m['hitl']:.1f} & {m['latency_min']:.1f} \\\\"
    return line.replace("Self-RAG", "Self-RAG").replace("ReAct", "ReAct")


def main() -> None:
    tier2 = json.loads(TIER2.read_text(encoding="utf-8"))
    cal = json.loads(CAL.read_text(encoding="utf-8"))
    anchor = json.loads((REPL / "config/deployment_anchor.json").read_text())

    rows = [
        ("Rule-Only", anchor["methods"]["Rule-Only"]),
        ("RAG-Only", anchor["methods"]["RAG-Only"]),
        ("Ensemble", anchor["methods"]["Ensemble"]),
        ("Self-RAG", tier2["deployment_calibrated"]["Self-RAG"]),
        ("ReAct", tier2["deployment_calibrated"]["ReAct"]),
        ("HRAG", anchor["methods"]["HRAG"]),
    ]

    body = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{CNAS Tier-2 deployment (partner-laboratory LLM pipeline)}",
            "\\label{tab:cnas}",
            "\\small",
            "\\begin{tabular}{@{}lcccc@{}}",
            "\\toprule",
            "Method & Acc.\\ (\\%) & Halluc.\\ (\\%) & HITL (\\%) & Latency (min) \\\\",
            "\\midrule",
            f"Rule-Only    & {rows[0][1]['accuracy']:.1f} & {rows[0][1]['hallucination']:.1f} & {rows[0][1]['hitl']:.1f} & {rows[0][1]['latency_min']:.1f} \\\\",
            f"RAG-Only     & {rows[1][1]['accuracy']:.1f} & {rows[1][1]['hallucination']:.1f} & {rows[1][1]['hitl']:.1f} & {rows[1][1]['latency_min']:.1f} \\\\",
            f"Ensemble     & {rows[2][1]['accuracy']:.1f} & {rows[2][1]['hallucination']:.1f} & {rows[2][1]['hitl']:.1f} & {rows[2][1]['latency_min']:.1f} \\\\",
            f"Self-RAG     & {rows[3][1]['accuracy']:.1f} & {rows[3][1]['hallucination']:.1f} & {rows[3][1]['hitl']:.1f} & {rows[3][1]['latency_min']:.1f} \\\\",
            f"ReAct        & {rows[4][1]['accuracy']:.1f} & {rows[4][1]['hallucination']:.1f} & {rows[4][1]['hitl']:.1f} & {rows[4][1]['latency_min']:.1f} \\\\",
            f"\\textbf{{\\HRAG{{}}}} & \\textbf{{{rows[5][1]['accuracy']:.1f}}} & \\textbf{{{rows[5][1]['hallucination']:.1f}}} & \\textbf{{{rows[5][1]['hitl']:.1f}}} & \\textbf{{{rows[5][1]['latency_min']:.1f}}} \\\\",
            "\\midrule",
            "Human audit (ref.) & 96.8 & 0.0 & -- & 35.0 \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\par\\vspace{0.3em}",
            "{\\footnotesize Self-RAG/ReAct Tier-2 rows transferred from C3PA proxy (\\texttt{replication/results/tier2\\_baselines.json}).}",
            "\\end{table}",
        ]
    )

    text = EXP_TEX.read_text(encoding="utf-8")
    start = text.find("\\label{tab:cnas}")
    if start < 0:
        raise SystemExit("tab:cnas label not found")
    begin = text.rfind("\\begin{table}", 0, start)
    end = text.find("\\end{table}", start) + len("\\end{table}")
    if begin < 0 or end <= begin:
        raise SystemExit("tab:cnas table block not found")
    text = text[:begin] + body + text[end:]
    EXP_TEX.write_text(text, encoding="utf-8")

    ece_raw, ece_cal = cal["ece_raw"], cal["ece_platt"]
    method = METHOD_TEX.read_text(encoding="utf-8")
    old_ece = re.search(
        r"reducing Expected Calibration Error \(ECE\) \\cite\{guo2017calibration\} from [0-9.]+ to [0-9.]+\.",
        method,
    )
    if old_ece:
        new_ece = (
            f"reducing Expected Calibration Error (ECE) \\cite{{guo2017calibration}} "
            f"from {ece_raw:.2f} to {ece_cal:.2f}."
        )
        method = method[: old_ece.start()] + new_ece + method[old_ece.end() :]
    METHOD_TEX.write_text(method, encoding="utf-8")

    cal_text = CAL_TEX.read_text(encoding="utf-8")
    old_cal = re.search(r"Before calibration, ECE\$=[0-9.]+\$; after Platt scaling, ECE\$=[0-9.]+\$.", cal_text)
    if old_cal:
        new_cal = f"Before calibration, ECE$={ece_raw:.2f}$; after Platt scaling, ECE$={ece_cal:.2f}$."
        cal_text = cal_text[: old_cal.start()] + new_cal + cal_text[old_cal.end() :]
    CAL_TEX.write_text(cal_text, encoding="utf-8")
    print("Synced table and ECE from replication results")


if __name__ == "__main__":
    main()
