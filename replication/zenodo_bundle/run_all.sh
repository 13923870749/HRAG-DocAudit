#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
PY=".venv/bin/python"

echo "==> Generate datasets"
$PY scripts/generate_c3pa_synthetic.py
$PY scripts/generate_contractnli_synthetic.py
$PY scripts/generate_cnas_holdout.py

echo "==> Run experiments"
$PY run_c3pa.py
$PY run_contractnli.py
$PY run_cnas.py

echo "==> Update manuscript tables"
$PY update_manuscript_results.py

echo "Done. See results/*.json"
