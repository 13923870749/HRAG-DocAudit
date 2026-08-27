#!/usr/bin/env bash
# Export anonymized replication bundle for https://anonymous.4open.science/r/HRAG-DocAudit-D8AE
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXPORT="${ROOT}/../HRAG-DocAudit-export"
PAPER="$(cd "$ROOT/../.." && pwd)"

mkdir -p "$EXPORT/replication" "$EXPORT/manuscript/figures/data" "$EXPORT/manuscript/sections"

echo "==> Copy replication code (no raw CNAS reports)"
rsync -a --delete --exclude '.venv' --exclude '__pycache__' --exclude 'data/c3pa' \
  --exclude 'data/contract-nli' --exclude 'data/cnas_deidentified' \
  --exclude 'data/contract-nli/raw' --exclude 'data/contract-nli/_repo' \
  --exclude '*.zip' \
  "$ROOT/" "$EXPORT/replication/"

mkdir -p "$EXPORT/replication/data/cnas_deidentified"
cp "$ROOT/data/cnas_deidentified/audit_items_523.jsonl" "$EXPORT/replication/data/cnas_deidentified/" 2>/dev/null || \
  cp "$ROOT/zenodo_bundle/data/cnas_audit_items_deidentified.jsonl" "$EXPORT/replication/data/cnas_deidentified/audit_items_523.jsonl"

cat > "$EXPORT/README.md" <<'EOF'
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
EOF

cp "$ROOT/../manuscript/figures/data/public_benchmarks.csv" "$EXPORT/manuscript/figures/data/" 2>/dev/null || true
cp "$ROOT/../manuscript/figures/data/cnas_deployment.csv" "$EXPORT/manuscript/figures/data/" 2>/dev/null || true
cp "$ROOT/../manuscript/figures/data/lambda_ablation.csv" "$EXPORT/manuscript/figures/data/" 2>/dev/null || true
cp "$ROOT/../manuscript/figures/data-manifest.md" "$EXPORT/manuscript/figures/" 2>/dev/null || true
cp "$ROOT/../manuscript/figures/generate_figures.py" "$EXPORT/manuscript/figures/" 2>/dev/null || true
mkdir -p "$EXPORT/manuscript/sections"
rsync -a "$ROOT/../manuscript/sections/" "$EXPORT/manuscript/sections/"
cp "$ROOT/config/deployment_anchor.json" "$EXPORT/replication/config/" 2>/dev/null || true

cat > "$EXPORT/replication/LICENSE" <<'EOF'
MIT License

Copyright (c) 2026 HRAG-DocAudit authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

echo "Export ready: $EXPORT ($(du -sh "$EXPORT" | cut -f1))"
