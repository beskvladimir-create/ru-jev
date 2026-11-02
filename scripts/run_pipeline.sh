#!/usr/bin/env bash
# Full pipeline. Run on a GPU box with python 3.11.
# CONFIG defaults to v1.1; pass another config to reproduce an older run.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/v1_1.yaml}"
OUT=$(python - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["out_dir"])
PY
)

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

echo "== building unified dataset =="
python scripts/build_data.py

echo "== training ($CONFIG) =="
python scripts/train.py --config "$CONFIG"

echo "== evaluating =="
python scripts/eval.py --model "$OUT"

echo "== capability probe =="
python scripts/probe.py --model "$OUT" --quiet || true

echo "done. artifacts in $OUT"
