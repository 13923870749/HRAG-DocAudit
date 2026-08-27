# HRAG-DocAudit Replication Bundle (planned)

## Public datasets — download

| Dataset | URL |
|---------|-----|
| C3PA | https://aclanthology.org/2024.emnlp-main.217/ |
| ContractNLI | https://stanfordnlp.github.io/contract-nli/ |

Place raw files under:

```
replication/data/c3pa/
replication/data/contract-nli/
replication/data/cnas/          # de-identified audit items only
```

## Scripts (to implement)

| Script | Purpose |
|--------|---------|
| `run_c3pa.py` | Public Track-A evaluation |
| `run_contractnli.py` | Public Track-B evaluation |
| `run_cnas.py` | Private Track evaluation (requires local CNAS data) |
| `export_deidentified.py` | Export CNAS audit-item subset for Zenodo |

## Environment

```bash
python >= 3.10
# sentence-transformers, chromadb, rank-bm25, openai-compatible LLM client
pip install -r requirements.txt   # to be added
```

## Outputs

Results JSON → paste into `manuscript/sections/03_experiments.tex` (replace `\placeholder{...}`).

## Zenodo release checklist

- [ ] LICENSE (MIT or Apache-2.0)
- [ ] README with citation to this paper
- [ ] Config YAML (rules, prompts, hyperparameters)
- [ ] De-identified CNAS audit items (≥300)
- [ ] Synthetic report templates (10)
