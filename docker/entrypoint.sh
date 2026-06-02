#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:=/workspace/model}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${SERVED_MODEL_NAME:=kimi-k2.6}"
: "${TP_SIZE:=4}"
: "${PP_SIZE:=1}"
: "${CONTEXT_LENGTH:=262144}"
: "${MAX_RUNNING_REQUESTS:=4}"
: "${MEM_FRACTION_STATIC:=0.94}"
: "${CHUNKED_PREFILL_SIZE:=8192}"
: "${MAX_PREFILL_TOKENS:=16384}"
: "${PAGE_SIZE:=64}"
: "${QUANTIZATION:=modelopt_fp4}"
: "${KV_CACHE_DTYPE:=fp8_e4m3}"
: "${ATTENTION_BACKEND:=flashinfer}"
: "${MOE_RUNNER_BACKEND:=auto}"
: "${TOOL_CALL_PARSER:=kimi_k2}"
: "${REASONING_PARSER:=kimi_k2}"
: "${CHAT_TEMPLATE:=/workspace/model/chat_template.jinja}"
: "${WATCHDOG_TIMEOUT:=3600}"
: "${CUDA_GRAPH_MAX_BS:=4}"
: "${MODEL_LOADER_THREADS:=32}"
: "${ENABLE_MULTIMODAL:=1}"
: "${TRUST_REMOTE_CODE:=1}"
: "${PRE_WARM_NCCL:=1}"
: "${ENABLE_METRICS:=1}"
: "${LOG_REQUESTS:=1}"
: "${LOG_REQUESTS_LEVEL:=3}"

cmd=(
  python3 -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size "$TP_SIZE"
  --pipeline-parallel-size "$PP_SIZE"
  --context-length "$CONTEXT_LENGTH"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --max-prefill-tokens "$MAX_PREFILL_TOKENS"
  --page-size "$PAGE_SIZE"
  --quantization "$QUANTIZATION"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --attention-backend "$ATTENTION_BACKEND"
  --moe-runner-backend "$MOE_RUNNER_BACKEND"
  --tool-call-parser "$TOOL_CALL_PARSER"
  --reasoning-parser "$REASONING_PARSER"
  --chat-template "$CHAT_TEMPLATE"
  --model-loader-extra-config "{\"enable_multithread_load\": true, \"num_threads\": $MODEL_LOADER_THREADS}"
  --watchdog-timeout "$WATCHDOG_TIMEOUT"
  --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
)

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  cmd+=(--trust-remote-code)
fi

if [[ "$ENABLE_MULTIMODAL" == "1" ]]; then
  cmd+=(--enable-multimodal)
fi

if [[ "$PRE_WARM_NCCL" == "1" ]]; then
  cmd+=(--pre-warm-nccl)
fi

if [[ "$ENABLE_METRICS" == "1" ]]; then
  cmd+=(
    --enable-metrics
    --enable-request-time-stats-logging
    --export-metrics-to-file
    --export-metrics-to-file-dir /workspace/metrics
  )
fi

if [[ "$LOG_REQUESTS" == "1" ]]; then
  cmd+=(
    --log-requests
    --log-requests-level "$LOG_REQUESTS_LEVEL"
    --log-requests-format json
    --log-requests-target stdout
  )
fi

exec "${cmd[@]}" "$@"
