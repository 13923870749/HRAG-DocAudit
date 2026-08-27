# HRAG-DocAudit Replication Bundle

Zenodo-ready subset for partial reproduction.

## Contents

- `data/cnas_audit_items_deidentified.jsonl` — 523 de-identified audit items (no client text)
- `config/rules_cnas.json` — rule definitions
- `../run_c3pa.py`, `../run_contractnli.py`, `../run_cnas.py` — evaluation scripts
- `../hrag_eval/` — core library
- `../requirements.txt`

## Citation

Liu H, Lei Q, Feng R. Priority-cascaded hybrid RAG for regulated document compliance auditing.

## Note on public benchmarks

When official C3PA/ContractNLI are downloaded via `scripts/download_datasets.sh`, re-run `run_all.sh` to refresh public-track numbers.

## License

MIT (code); de-identified audit metadata only—no raw laboratory reports.
