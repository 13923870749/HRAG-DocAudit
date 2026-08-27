#!/usr/bin/env python3
"""Print credibility audit summary from full reports."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main():
    for name in ("c3pa", "contractnli"):
        path = RES / f"{name}_full_report.json"
        if not path.exists():
            continue
        rep = json.loads(path.read_text(encoding="utf-8"))
        print(f"\n## {name.upper()} (n={rep['n']})")
        cov = rep.get("coverage", {})
        print(f"  Rule-amenable: {cov.get('n_rule_amenable', '?')} ({_pct(cov.get('rule_amenable_rate', 0))})")
        m = rep["methods"]
        hrag, rag, ens = m["hrag"], m["rag"], m["ensemble"]
        print(f"  HRAG acc={hrag['accuracy']:.3f}  RAG acc={rag['accuracy']:.3f}  Ens acc={ens['accuracy']:.3f}")
        print(f"  HRAG halluc(lenient)={hrag.get('hallucination_rate', 0):.3f}  strict={hrag.get('hallucination_rate_strict', 0):.3f}")
        print(f"  Ens halluc(lenient)={ens.get('hallucination_rate', 0):.3f}  strict={ens.get('hallucination_rate_strict', 0):.3f}")
        print(f"  Rule-RAG conflict={hrag.get('conflict_rate', 0):.3f}")
        ca = rep.get("cascade_analysis", {})
        for method in ("hrag", "ensemble"):
            if method in ca and ca[method].get("n_rule_routed"):
                c = ca[method]
                print(
                    f"  [{method}] rule-routed n={c['n_rule_routed']} "
                    f"align={c['rule_alignment']:.3f} "
                    f"halluc={c.get('hallucination_rate', 0):.3f}"
                )
        for key in ("mcnemar_hrag_vs_rag", "mcnemar_hrag_vs_ensemble"):
            mc = rep.get(key, {})
            if mc:
                print(f"  {key}: p={mc.get('p_value', 1):.2e}")
        for sk, sv in rep.get("strata", {}).items():
            print(f"  Stratum [{sk}] n={sv['n']}: HRAG={sv['hrag']['accuracy']:.3f} RAG={sv['rag']['accuracy']:.3f}")

    abl = RES / "lambda_ablation.json"
    if abl.exists():
        a = json.loads(abl.read_text(encoding="utf-8"))
        print("\n## Lambda ablation (validation)")
        for tr in a.get("tracks", []):
            best = tr["best_lambda_ndcg"]
            ndcg = tr["best_ndcg_at_10"]
            print(f"  {tr['dataset']}: best λ={best} NDCG@10={ndcg:.3f}")

    cnas = RES / "cnas_results.json"
    if cnas.exists():
        c = json.loads(cnas.read_text(encoding="utf-8"))
        ht = c["holdout_test"]["methods"]["hrag"]
        print(f"\n## CNAS hold-out replication (n={c['holdout']['test_n']})")
        print(f"  HRAG acc={ht['accuracy']:.3f} halluc={ht.get('hallucination_rate', 0):.3f}")


if __name__ == "__main__":
    main()
