# Experiment Audit Report (v3)

**Date:** 2026-08-27  
**Scope:** Full experiment redesign, code fixes, re-run

## Changes in v3

| Area | Change |
|------|--------|
| Metrics | `rule_subspace` stats, `output_covariance_rule_method`, fixed hallucination bootstrap CI |
| Tests | McNemar HRAG vs Ensemble added |
| Ablation | `run_ablation.py` — λ sweep on validation splits |
| Cascade | `cascade_analysis` block in full reports |
| Runners | Legacy `run_c3pa.py` / `run_contractnli.py` → delegate to `run_public.py` |
| Manuscript | tab:cascade, tab:lambda; discussion aligned with stratified results |

## Tier-1 results (official test splits)

### C3PA (n=456)
| Method | Acc | Halluc |
|--------|-----|--------|
| RAG | 52.4% | 5.3% |
| Ensemble | 91.7% | 1.8% |
| HRAG | 91.7% | 0.0% |

- Coverage: 93.6% rule-amenable
- rule-routed (n=424): HRAG halluc 0% vs Ensemble 1.9%
- McNemar HRAG vs RAG: p≈3×10⁻³⁷

### ContractNLI (n=2091)
| Method | Acc | Halluc | NDCG@10 |
|--------|-----|--------|---------|
| RAG | 56.8% | 29.1% | 0.869 |
| Ensemble | 60.9% | 0.3% | — |
| HRAG | 60.9% | 27.8% | — |

- rule_amenable: HRAG 87.8% / 0% halluc vs RAG 67.1% / 6.3%
- McNemar HRAG vs RAG: p≈2×10⁻¹³

### λ ablation (validation)
- Peak NDCG@0.9: C3PA 0.459, ContractNLI 0.826
- Test accuracy favors λ=0.6 on ContractNLI → default retained

## Theory–experiment alignment

| Claim | Supported? |
|-------|------------|
| Rule/RAG division on rule-amenable items | **Yes** (large stratum gaps) |
| HRAG > RAG overall | **Yes** (McNemar) |
| Cascade safer on rule-routed items | **Yes** (0% vs ~1.5–1.9% halluc) |
| Cascade beats Ensemble on overall halluc (ContractNLI) | **No** — report honestly |
| Hybrid > sparse-only retrieval | **Yes** (λ ablation) |
| CNAS deployment via proxy | **No** — Tier-2 only |

## Reproduce

```bash
cd submission/eaai/replication
./run_all.sh
```
