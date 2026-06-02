# Kimi-K2.6 NVFP4 SGLang Deployment

> [!TIP]
> **[Support this work ->](https://donate.sybilsolutions.ai)** | [X](https://x.com/0xsero) | [GitHub](https://github.com/0xsero) | [Model card](https://huggingface.co/0xSero/Kimi-K2.6-519B-NVFP4)

Minimal Docker deployment for [0xSero/Kimi-K2.6-519B-NVFP4](https://huggingface.co/0xSero/Kimi-K2.6-519B-NVFP4) with SGLang, 256k context, multimodal input, and the Kimi reasoning/tool parsers enabled.

This repo only serves the model. It does not include benchmark runners, pruning code, traces, or checkpoint artifacts.

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
