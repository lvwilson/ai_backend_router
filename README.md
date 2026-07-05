# Smart LLM Router

A FastAPI-based router that provides a **unified OpenAI-compatible API** over multiple AI backends, with **warm-by-default lifecycle management** and **VRAM-pressure-based eviction**.

## What it does

- **Launches backends on demand** — llama.cpp, CrispASR, and ComfyUI start automatically when first requested.
- **Keeps them warm** — backends stay running for fast reuse instead of cold-starting on every request.
- **Evicts under VRAM pressure** — when a large model won't fit, the smallest running backend is evicted first (smallest-first eviction), with real `nvidia-smi` confirmation.
- **Translates APIs** — ComfyUI's proprietary workflow API is translated to/from OpenAI `/v1/images/generations` format automatically.

### Supported backends

| Backend   | API format       | Routes                                      |
|-----------|------------------|---------------------------------------------|
| llama.cpp | OpenAI + Anthropic | `POST /v1/chat/completions`, `/v1/messages` |
| CrispASR  | OpenAI audio     | `POST /v1/audio/transcriptions`, `/v1/audio/speech`, `/v1/translate`, `GET /v1/voices` |
| ComfyUI   | Translated       | `POST /v1/images/generations`               |

## Quickstart

### 1. Install dependencies

```bash
pip install fastapi uvicorn aiohttp pyyaml
```

### 2. Configure backends

Copy `config.example.yaml` to `config.yaml` and edit paths, ports, and VRAM estimates:

```bash
cp config.example.yaml config.yaml
```

Key settings in `config.yaml`:

```yaml
router:
  total_vram: 48        # Your GPU's total VRAM in GB
  vram_reserve: 2       # VRAM to keep free

backends:
  qwen3.6-27b:
    type: llama
    model: ~/models/llm/your-model.gguf
    port: 8080
    vram_usage: 45      # Estimated VRAM; measured on launch

  image:
    type: comfyui
    venv: ~/ComfyUI/venv/bin/python
    working_dir: ~/ComfyUI
    models:
      krea2:
        workflow: ~/path/to/workflow.json
        vram_usage: 14
```

### 3. Run the router

```bash
python router.py config.yaml
```

The router starts on the configured host/port (default `0.0.0.0:8000`).

### 4. Use it

All requests go through the router. Backends launch automatically on first request:

```bash
# Chat completion — launches llama.cpp on first call
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Image generation — launches ComfyUI, translates to workflow
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "krea2",
    "prompt": "A cat in a spacesuit",
    "size": "1024x1024"
  }'

# List available models
curl http://localhost:8000/v1/models

# Fleet status (VRAM, backends)
curl http://localhost:8000/status
```

## How eviction works

When a backend is requested and there isn't enough free VRAM:

1. The router queries `nvidia-smi` for current VRAM usage (falls back to tracked bookkeeping if unavailable).
2. Running backends are sorted by VRAM usage (smallest first).
3. The smallest backend is stopped; VRAM release is confirmed via `nvidia-smi`.
4. Steps 2–3 repeat until the requested backend fits.
5. The requested backend is launched and warmed up.

A configurable `vram_reserve` (default 2 GB) is always kept free for the OS/compositor.

## Architecture

```
router.py          — FastAPI app, request routing, passthrough proxying
orchestrator.py    — VRAM budgeting, smallest-first eviction, warm reuse
service_loader.py  — Per-backend process lifecycle (start/stop/health/VRAM)
health_checker.py  — HTTP/TCP health probing
comfyui.py         — OpenAI ↔ ComfyUI workflow translation
config.py          — YAML config loader, per-type command builders
```

## Requirements

- Python 3.11+
- `nvidia-smi` available (for VRAM management; graceful fallback if absent)
- Backend binaries in `PATH` or specified by absolute path in config
