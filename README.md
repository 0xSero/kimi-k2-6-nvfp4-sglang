# Kimi-K2.6 NVFP4 SGLang Deployment

> [!TIP]
> **[Support this work ->](https://donate.sybilsolutions.ai)** | [X](https://x.com/0xsero) | [GitHub](https://github.com/0xsero) | [Model card](https://huggingface.co/0xSero/Kimi-K2.6-519B-NVFP4)

Minimal Docker deployment for [0xSero/Kimi-K2.6-519B-NVFP4](https://huggingface.co/0xSero/Kimi-K2.6-519B-NVFP4) with SGLang, 256k context, multimodal input, and the Kimi reasoning/tool parsers enabled.

This repo only serves the model. It does not include benchmark runners, pruning code, traces, or checkpoint artifacts.

> [!WARNING]
> This is a **REAP-pruned checkpoint (half the experts removed)** and will fall into **repetition loops on open-ended / long-form generation**. It is reliable for structured, tool-calling, coding, math, and agentic use (bounded outputs). Read [Repetition-loop attractors](#repetition-loop-attractors) before using it as a free-form chat model.

## At a glance

| | |
|---|---|
| Model | `0xSero/Kimi-K2.6-519B-NVFP4` |
| Served model name | `kimi-k2.6` |
| Engine | SGLang `v0.5.12.post1` |
| Quantization | `modelopt_fp4` / NVFP4 |
| KV cache | `fp8_e4m3` |
| Context | `262,144` |
| Tensor parallel | `4` |
| Max running requests | `4` |
| Vision | enabled |
| Reasoning parser | `kimi_k2` |
| Tool-call parser | `kimi_k2` |
| Verified GPUs | `4x RTX PRO 6000 Blackwell`, `CUDA_VISIBLE_DEVICES=0,2,3,4` |
| Fast path | vLLM + Eagle3 speculative decoding (see below) |

## ⚡ Fast vLLM recipe (recommended)

For maximum single-stream speed, serve with vLLM + Eagle3 speculative decoding and the
`flashinfer_cutlass` NVFP4 MoE kernel instead of SGLang:

```bash
./scripts/serve-vllm.sh
```

Measured on the verified host (4x RTX PRO 6000 Blackwell, **275W power cap, unchanged**), single stream:

| Metric | Value |
|---|---|
| Decode | ~71 tok/s aggregate (up to ~80 on code/structured); **+3.6%** vs plain cutlass |
| Prefill | ~2070 tok/s |
| TTFT (warm) | ~130 ms |
| Context | 262,144 (GPU KV ~811k tokens, ~3.09x concurrency) |
| Prefix caching | **on** (`--enable-prefix-caching`; ~74% hit on repeated prefixes) |
| First boot | ~9 min (flashinfer autotune); ~2.5 min warm via persistent `JIT_CACHE_DIR` |

Key settings (see `scripts/serve-vllm.sh`):

| Setting | Value |
|---|---|
| Engine | vLLM `0.19.2rc1` (image `voipmonitor/vllm:cu130-mtp-tuned-v3-20260423`) |
| MoE backend | `flashinfer_cutlass` (fastest NVFP4 MoE on SM120; `flashinfer_trtllm` is unsupported there) |
| Speculative decoding | Eagle3 draft `lightseekorg/kimi-k2.6-eagle3-mla`, 3 tokens, probabilistic, draft KV fp8 |
| Attention | `TRITON_MLA` (FlashInfer-MLA fp8+DCP is unsupported in this build) |
| Tensor / decode-context parallel | 4 / 4 |
| KV cache | `fp8_e4m3` |
| Custom all-reduce | **disabled** (PCIe P2P custom all-reduce hangs on this no-NVLink topology) |
| Default sampling | `repetition_penalty=1.12, temperature=0` via `--override-generation-config` |

> Decode is **power-bound** at 275W (clocks throttle from 3090 → ~2840 MHz under load), so the
> speedup comes from a cheaper MoE kernel + speculation, **not** higher power — never raise the
> power cap. Kernel fusion (`fuse_norm_quant`) gave no gain (flashinfer quantizes internally).

## Repetition-loop attractors

> [!WARNING]
> This checkpoint is **REAP-pruned to 192 of 384 experts per layer (50% removed)**. That recovers
> most short-form quality but leaves a **repetition-loop pathology on open-ended / long-form
> generation**: the model starts coherent, then collapses into an endless single-token or
> short-phrase loop and never terminates.

**Observed behavior**

- On a simple open prompt ("tell me about cats and cat allergens"), generations degenerated into
  loops such as `saliva saliva saliva…`, `allergen allergen…`, `hairs hairs…`, `felis-felis-felis…`.
  The attractor token is prompt-dependent — any unbounded prose can trigger one.
- **Thinking off:** the loop appears directly in the answer.
- **Thinking on:** the model never closes the reasoning block — it fills the entire context window
  with un-terminated reasoning and returns empty `content`.

**Sampling does NOT reliably fix it.** Still looping on long-form with: `temperature=0 +
repetition_penalty=1.12` (clean on *short* structured tasks only), `temperature=0.7 +
repetition_penalty=1.05`, `temperature=0.6` with presence/frequency penalties, and
`repetition_penalty` up to `1.15`.

**Where it is reliable** (validated): structured JSON, tool/function calling, code, math, short Q&A,
and agentic/terminal tasks — i.e. bounded outputs. Vision, thinking, tool-calling, and speculative
decoding all work in these modes.

**Mitigations**

- Keep outputs bounded (set a sane `max_tokens`) and prefer structured/agentic use.
- Default `repetition_penalty≈1.12, temperature=0` (already the server default here) helps short tasks.
- Watch for repetition and stop generation if a loop starts.
- The real fix is restoring some pruned experts from the full `nvidia/Kimi-K2.6-NVFP4` checkpoint
  (not done here) — a memory/context tradeoff on 4x96GB.

**Root cause:** aggressive REAP expert pruning (keep192), uniform across all 60 MoE layers — a
property of the pruned weights, not the serving config.

## Quick start

Install Docker with the NVIDIA Container Toolkit, then clone this repo.

```bash
git clone https://github.com/0xSero/kimi-k2-6-nvfp4-sglang.git
cd kimi-k2-6-nvfp4-sglang
cp .env.example .env
```

Edit `.env` if your model path or GPU ids differ from the verified host.

Download the checkpoint:

```bash
./scripts/pull-model.sh
```

Start the server:

```bash
./scripts/serve.sh
```

First launch usually takes a few minutes: weight loading is about one minute on the verified host, then CUDA graph capture takes about two minutes. The server is ready when logs print `The server is fired up and ready to roll!`.

Check the endpoint:

```bash
./scripts/health.sh
```

Expected model response:

```json
{
  "id": "kimi-k2.6",
  "max_model_len": 262144
}
```

## Docker Compose

The direct launch script is the primary path. Compose is also available:

```bash
docker compose up -d --build
docker compose logs -f
```

## API smoke test

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kimi-k2.6",
    "messages": [{"role": "user", "content": "Briefly say READY."}],
    "temperature": 0,
    "max_tokens": 256
  }'
```

The Kimi chat template enables thinking by default. Short responses may place early tokens in `reasoning_content`; use enough `max_tokens` for the final answer.

## Current verified launch settings

The defaults in `.env.example` match the verified live launch:

| Setting | Value |
|---|---|
| Model path in container | `/workspace/model` |
| Host model dir | `/mnt/llm_models/Kimi-K2.6-519B-NVFP4` |
| API port | `8000` |
| `--served-model-name` | `kimi-k2.6` |
| `--context-length` | `262144` |
| `--max-running-requests` | `4` |
| `--chunked-prefill-size` | `8192` |
| `--max-prefill-tokens` | `16384` |
| `--mem-fraction-static` | `0.94` |
| `--cuda-graph-max-bs` | `4` |
| `--enable-multimodal` | enabled |
| `--tool-call-parser` | `kimi_k2` |
| `--reasoning-parser` | `kimi_k2` |

## Troubleshooting

If `/health` returns `503` immediately after startup, wait for the first internal health generation to finish and retry.

If `/v1/models` shows an old id such as `kimi-k2.6-519b-nvfp4`, stop stale containers and relaunch:

```bash
docker rm -f sglang-kimi-k26
./scripts/serve.sh
```

Then verify:

```bash
curl -sS http://127.0.0.1:8000/v1/models
```

The response should contain only `kimi-k2.6`.
