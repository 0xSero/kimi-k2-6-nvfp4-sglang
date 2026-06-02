#!/usr/bin/env bash
set -euo pipefail

API_BASE="${API_BASE:-http://127.0.0.1:${PORT:-8000}}"

echo "GET $API_BASE/health"
curl -fsS -i "$API_BASE/health" | sed -n '1,12p'

echo
echo "GET $API_BASE/v1/models"
if command -v jq >/dev/null 2>&1; then
  curl -fsS "$API_BASE/v1/models" | jq .
else
  curl -fsS "$API_BASE/v1/models"
  echo
fi
