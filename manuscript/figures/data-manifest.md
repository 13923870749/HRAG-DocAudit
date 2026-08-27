# Figure data manifest (EAAI submission)

| Figure | Data file | Real/mock | Source | Script | Outputs |
|---|---|---|---|---|---|
| Fig. 1 public benchmarks | `data/public_benchmarks.csv` | Real | Table `tab:public` (Acc/Halluc columns) | `generate_figures.py` | `fig1_public_benchmarks.{pdf,png,svg}` |
| Fig. 2 mechanism evidence | `data/stratified_cascade.csv`, `data/lambda_ablation.csv` | Real | Appendix `tab:strata`, `tab:cascade`, `tab:lambda` | `generate_figures.py` | `fig2_mechanism_evidence.{pdf,png,svg}` |
| Fig. 3 CNAS deployment | `data/cnas_deployment.csv` | Real | Table `tab:cnas` (methods executed on CNAS only) | `generate_figures.py` | `fig3_cnas_deployment.{pdf,png,svg}` |
| Fig. 4c λ–NDCG | `data/lambda_ablation.csv` | Real | `replication/results/lambda_ablation.json` validation NDCG@10 | `generate_figures.py` | `fig4c_lambda_ablation.{pdf,png,svg}` |
| Fig. 4d λ–Accuracy | `data/lambda_ablation.csv` | Real | same JSON, `rag_accuracy`; **not** a CNAS 20-report grid | `generate_figures.py` | `fig4d_lambda_accuracy.{pdf,png,svg}` |

Regenerate: `python3 manuscript/figures/generate_figures.py` or `bash build.sh`.
