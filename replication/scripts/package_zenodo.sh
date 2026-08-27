#!/usr/bin/env bash
# Package Zenodo replication bundle (code + de-identified metadata).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUNDLE="$ROOT/zenodo_bundle"
ZIP="$ROOT/hrag_docaudit_replication_v0.1.zip"

mkdir -p "$BUNDLE/data" "$BUNDLE/config"

cp -r "$ROOT/hrag_eval" "$BUNDLE/"
cp "$ROOT/requirements.txt" "$ROOT/run_all.sh" "$ROOT/run_c3pa.py" "$ROOT/run_contractnli.py" \
   "$ROOT/run_cnas.py" "$ROOT/update_manuscript_results.py" "$BUNDLE/"
mkdir -p "$BUNDLE/scripts"
cp "$ROOT/scripts/"*.py "$BUNDLE/scripts/" 2>/dev/null || true
cp "$ROOT/scripts/"*.sh "$BUNDLE/scripts/" 2>/dev/null || true

rm -f "$ZIP"
(cd "$ROOT" && zip -rq "$ZIP" zenodo_bundle)
echo "Created $ZIP ($(du -h "$ZIP" | cut -f1))"
