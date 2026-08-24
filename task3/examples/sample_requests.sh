#!/usr/bin/env bash
set -euo pipefail
BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "=== Health ==="
curl -s "$BASE_URL/health" | python -m json.tool

echo
echo "=== Predict sample sneaker ==="
curl -s -X POST "$BASE_URL/predict" \
  -F "file=@examples/samples/sample_sneaker.png" \
  | python -m json.tool
