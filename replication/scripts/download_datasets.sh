#!/usr/bin/env bash
# Download official public datasets when network is available.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"
mkdir -p "$DATA/c3pa" "$DATA/contract-nli"

echo "Downloading C3PA (GitHub zip)..."
if [[ ! -d "$DATA/c3pa/Annotations" ]]; then
  curl -fsSL -o /tmp/c3pa.zip "https://github.com/MaazBinMusa/C3PA_Dataset/archive/refs/heads/main.zip"
  unzip -qo /tmp/c3pa.zip -d "$DATA"
  rm -rf "$DATA/c3pa" && mv "$DATA/C3PA_Dataset-main" "$DATA/c3pa"
fi

echo "Preparing C3PA jsonl splits..."
"$ROOT/.venv/bin/python" "$ROOT/scripts/prepare_c3pa_official.py"

echo "Downloading ContractNLI (Stanford gh-pages)..."
if [[ ! -f "$DATA/contract-nli/raw/contract-nli/train.json" ]]; then
  rm -rf "$DATA/contract-nli/_repo"
  git clone --depth 1 --branch gh-pages https://github.com/stanfordnlp/contract-nli.git "$DATA/contract-nli/_repo"
  unzip -qo "$DATA/contract-nli/_repo/resources/contract-nli.zip" -d "$DATA/contract-nli/raw"
fi

echo "Preparing ContractNLI jsonl splits..."
"$ROOT/.venv/bin/python" "$ROOT/scripts/prepare_contractnli_official.py"

echo "Official datasets ready. Re-run replication/run_all.sh"
