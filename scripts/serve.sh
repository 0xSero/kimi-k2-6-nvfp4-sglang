#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE. Copy .env.example to .env and edit paths/GPU ids first." >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

: "${MODEL_DIR:?MODEL_DIR is required}"
: "${LOG_DIR:?LOG_DIR is required}"
: "${METRICS_DIR:?METRICS_DIR is required}"
: "${CONTAINER_NAME:=sglang-kimi-k26}"
: "${SGLANG_IMAGE:=local/kimi-k2-6-nvfp4-sglang:latest}"
: "${SGLANG_BASE_IMAGE:=lmsysorg/sglang:v0.5.12.post1}"
: "${SHM_SIZE:=96g}"

mkdir -p "$LOG_DIR" "$METRICS_DIR"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model directory does not exist: $MODEL_DIR" >&2
  echo "Run ./scripts/pull-model.sh or set MODEL_DIR in .env." >&2
  exit 1
fi

docker build \
  --build-arg "SGLANG_BASE_IMAGE=$SGLANG_BASE_IMAGE" \
  -t "$SGLANG_IMAGE" \
  "$ROOT"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d --rm \
  --name "$CONTAINER_NAME" \
  --network host \
  --ipc=host \
  --privileged \
  --shm-size "$SHM_SIZE" \
  --gpus all \
  --env-file "$ENV_FILE" \
  -v "$MODEL_DIR:/workspace/model:ro" \
  -v "$LOG_DIR:/workspace/logs" \
  -v "$METRICS_DIR:/workspace/metrics" \
  "$SGLANG_IMAGE"

echo "Started $CONTAINER_NAME."
echo "API: http://127.0.0.1:${PORT:-8000}/v1"
echo "Model id: ${SERVED_MODEL_NAME:-kimi-k2.6}"
echo "Following logs. Ctrl-C stops log following, not the container."
docker logs -f "$CONTAINER_NAME"
