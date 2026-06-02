#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE. Copy .env.example to .env first." >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

: "${HF_MODEL_ID:=0xSero/Kimi-K2.6-519B-NVFP4}"
: "${MODEL_DIR:?MODEL_DIR is required}"

mkdir -p "$MODEL_DIR"
huggingface-cli download "$HF_MODEL_ID" --local-dir "$MODEL_DIR" --local-dir-use-symlinks False
