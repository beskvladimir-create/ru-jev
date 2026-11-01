#!/usr/bin/env bash
# Full v1 pipeline. Run on a GPU box (RunPod, python 3.11).
set -euo pipefail
cd "$(dirname "$0")/.."

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

echo "== building unified dataset =="
python scripts/build_data.py

echo "== training =="
python scripts/train.py --config configs/base.yaml

echo "== evaluating =="
python scripts/eval.py --model out/ru-jev-v1

echo "done. artifacts in out/ru-jev-v1"
