#!/usr/bin/env bash
# Download official public datasets when network is available.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"
mkdir -p "$DATA/c3pa" "$DATA/contract-nli"

echo "Downloading C3PA (GitHub zip)..."
curl -fsSL -o /tmp/c3pa.zip "https://github.com/MaazBinMusa/C3PA_Dataset/archive/refs/heads/main.zip"
unzip -qo /tmp/c3pa.zip -d "$DATA"
rm -rf "$DATA/c3pa" && mv "$DATA/C3PA_Dataset-main" "$DATA/c3pa"

echo "Downloading ContractNLI (HuggingFace parquet)..."
"$ROOT/.venv/bin/python" <<'PY'
from huggingface_hub import hf_hub_download
import pandas as pd, os
out = "/home/liuhui/shared/papers/基于混合检索增强生成的文档自动化合规性审核方法研究/submission/eaai/replication/data/contract-nli"
for split in ["train","validation","test"]:
    p = hf_hub_download(repo_id="presencesw/contract-nli", repo_type="dataset", filename=f"default/{split}/data-00000-of-00001.parquet")
    pd.read_parquet(p).to_json(os.path.join(out, f"{split}.jsonl"), orient="records", lines=True, force_ascii=False)
    print("saved", split)
PY

echo "Official datasets ready. Re-run replication/run_all.sh"
