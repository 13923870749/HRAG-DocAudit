#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PY=".venv/bin/python"

echo "==> Prepare official datasets"
if [[ -d data/c3pa/Annotations ]]; then
  $PY scripts/prepare_c3pa_official.py
fi
if [[ -f data/contract-nli/raw/contract-nli/test.json ]] || [[ -f data/contract-nli/test.jsonl ]]; then
  $PY scripts/prepare_contractnli_official.py
fi
$PY scripts/generate_cnas_holdout.py

echo "==> Run public + private experiments"
$PY run_public.py
$PY run_ablation.py
$PY run_cnas.py
$PY run_tier2_baselines.py
$PY scripts/generate_calibration_curve.py

echo "==> Audit summary"
$PY scripts/audit_results.py
$PY scripts/generate_hallucination_supp.py

echo "Done. See results/*_full_report.json"
