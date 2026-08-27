# HRAG-DocAudit (anonymous review bundle)

Priority-cascaded hybrid RAG for regulated document compliance auditing.

## Quick start

```bash
cd replication
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_datasets.sh   # optional: fetch public benchmarks
python3 scripts/generate_cnas_holdout.py
bash run_all.sh
python3 run_tier2_baselines.py
python3 scripts/generate_calibration_curve.py
```

## Contents

| Path | Description |
|------|-------------|
| `replication/run_c3pa.py` | C3PA Tier-1 evaluation |
| `replication/run_contractnli.py` | ContractNLI Tier-1 evaluation |
| `replication/run_cnas.py` | CNAS de-identified hold-out proxy |
| `replication/run_tier2_baselines.py` | Self-RAG / ReAct C3PA proxy baselines |
| `replication/hrag_eval/` | Core evaluator (Rule / RAG / Ensemble / HRAG / Self-RAG / ReAct) |
| `replication/data/cnas_deidentified_public/` | 523 de-identified audit items (no client text) |
| `replication/config/rules_cnas.json` | CNAS rule definitions (Supplementary Table S1) |

## License

MIT (code). De-identified audit metadata only — no raw laboratory reports.

## Citation

Liu H, Lei Q, Feng R. Priority-cascaded hybrid RAG for automated compliance auditing of regulated documents. *Engineering Applications of Artificial Intelligence* (under review).
