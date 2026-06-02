#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Fast vLLM speculative-decoding recipe for Kimi-K2.6-519B-NVFP4 (REAP keep192).
#
# This is the recommended high-throughput path. vs the SGLang recipe it adds
# Eagle3 speculative decoding and the flashinfer_cutlass NVFP4 MoE kernel, which
# is the fastest NVFP4 MoE backend reachable on SM120 (RTX PRO 6000 Blackwell).
#
# Measured on 4x RTX PRO 6000 Blackwell (275W cap, NOT changed), single stream:
#   decode      ~71 tok/s aggregate (up to ~80 on structured/code), +3.6% vs plain cutlass
#   prefill     ~2070 tok/s
#   TTFT        ~130 ms (warm)
#   context     262144 (GPU KV ~811k tokens, ~3.09x concurrency)
#   prefix cache ON (observed ~74% hit on repeated prefixes)
#
# First boot runs flashinfer autotune (~9 min). A persistent JIT cache volume
# (JIT_CACHE_DIR) makes subsequent boots ~2.5 min.
#
# IMPORTANT: never raise GPU power caps/clocks. Decode here is power-bound at
# 275W; the speed comes from a cheaper MoE kernel + speculation, not more watts.
#
# READ the repetition-loop limitation in README.md before using for open-ended
# chat. This REAP-pruned checkpoint degenerates into repetition loops on long
# free-form generation; it is reliable for structured / tool / code / agentic use.
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/mnt/llm_models/Kimi-K2.6-519B-NVFP4}
DRAFT_MODEL=${DRAFT_MODEL:-lightseekorg/kimi-k2.6-eagle3-mla}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-kimi-k2.6}
CONTAINER_NAME=${CONTAINER_NAME:-vllm-kimi-k26}
IMAGE=${IMAGE:-voipmonitor/vllm:cu130-mtp-tuned-v3-20260423}
PORT=${PORT:-8000}
GPU_DEVICES=${GPU_DEVICES:-0,2,3,4}              # 4x RTX PRO 6000 Blackwell

MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
DECODE_CONTEXT_PARALLEL_SIZE=${DECODE_CONTEXT_PARALLEL_SIZE:-4}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.92}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e4m3}
MOE_BACKEND=${MOE_BACKEND:-flashinfer_cutlass}   # fastest supported NVFP4 MoE on SM120
NUM_SPECULATIVE_TOKENS=${NUM_SPECULATIVE_TOKENS:-3}
# Server-side default sampling: prevents the repetition loop for clients that
# send no sampling params (the bare model + vLLM defaults would loop). Per-request
# params still override. See README "Repetition-loop attractors".
OVERRIDE_GENERATION_CONFIG=${OVERRIDE_GENERATION_CONFIG:-'{"repetition_penalty":1.12,"temperature":0}'}
# Persistent flashinfer/triton/torch.compile cache -> fast subsequent boots.
JIT_CACHE_DIR=${JIT_CACHE_DIR:-/mnt/llm_models/cache/kimi-k26-jit}
HF_CACHE=${HF_CACHE:-/mnt/llm_models/hf}

SPECULATIVE_CONFIG=$(cat <<JSON
{"model":"${DRAFT_MODEL}","method":"eagle3","num_speculative_tokens":${NUM_SPECULATIVE_TOKENS},"draft_attention_backend":"TRITON_MLA","rejection_sample_method":"probabilistic","draft_kv_cache_dtype":"fp8"}
JSON
)

mkdir -p "$HF_CACHE" "$JIT_CACHE_DIR"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

exec docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus "\"device=${GPU_DEVICES}\"" \
  --ipc=host --shm-size=96g --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  --entrypoint /opt/venv/bin/vllm \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${JIT_CACHE_DIR}:/cache/jit" \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e VLLM_ATTENTION_BACKEND=TRITON_MLA \
  -e VLLM_USE_V1=1 \
  -e NCCL_P2P_LEVEL=PIX -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_HOST_ENABLE=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$IMAGE" \
  serve /model \
  --host 0.0.0.0 --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --decode-context-parallel-size "$DECODE_CONTEXT_PARALLEL_SIZE" \
  --attention-backend TRITON_MLA \
  --trust-remote-code \
  --quantization modelopt \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --moe-backend "$MOE_BACKEND" \
  --no-enable-expert-parallel \
  --reasoning-parser kimi_k2 \
  --enable-auto-tool-choice --tool-call-parser kimi_k2 \
  --speculative-config "$SPECULATIVE_CONFIG" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --disable-custom-all-reduce \
  --override-generation-config "$OVERRIDE_GENERATION_CONFIG" \
  --generation-config vllm
