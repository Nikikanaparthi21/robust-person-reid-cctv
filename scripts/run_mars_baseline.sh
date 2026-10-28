#!/usr/bin/env bash
# One-command real-data baseline on MARS (run from the repo root on your own machine).
# Usage: bash scripts/run_mars_baseline.sh "/path/to/MARS dataset/data/bbox_test.zip" [mps|cpu|cuda]
# Steps: venv + deps, tests, dry run (expects 19776 images), frozen ResNet-50 extraction, evaluation.
set -euo pipefail

ZIP="${1:?pass the path to bbox_test.zip}"
DEVICE="${2:-mps}"
OUT=data/embeddings/mars_test

[ -f "$ZIP" ] || { echo "zip not found: $ZIP"; exit 1; }
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt torch torchvision

echo "== tests =="
python -m pytest -q

echo "== dry run (expect: 681089 images in zip -> 19776 selected) =="
python scripts/extract_embeddings.py --zip "$ZIP" --out "$OUT" --dry-run

echo "== extraction (frozen ImageNet ResNet-50, device=$DEVICE) =="
python scripts/extract_embeddings.py --zip "$ZIP" --out "$OUT" --model resnet50 --device "$DEVICE"

echo "== evaluation =="
python scripts/evaluate_features.py --features "$OUT.npy" --names "${OUT}_names.txt" \
    --out results/mars_resnet50_frozen --max-ids 100
echo "Done. Paste the table above (or results/mars_resnet50_frozen/results.csv) back to Claude."
