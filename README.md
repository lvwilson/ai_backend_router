# Smart LLM Router

A FastAPI-based router that provides a **unified OpenAI-compatible API** over multiple AI backends, with **warm-by-default lifecycle management** and **VRAM-pressure-based eviction**.

## What it does

- **Launches backends on demand** — llama.cpp, CrispASR, and ComfyUI start automatically when first requested.
- **Keeps them warm** — backends stay running for fast reuse instead of cold-starting on every request.
- **Evicts under VRAM pressure** — when a large model won't fit, the smallest running backend is evicted first (smallest-first eviction), with real `nvidia-smi` confirmation.
- **Queues requests (concurrency 1 per model)** — a new request for a busy model waits its turn instead of running concurrently, and the router never evicts a backend that has a request in flight. A new request can therefore never kill a previous one.
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

For `/v1/chat/completions` and `/v1/messages`, model names match
case-insensitively. A unique match ignoring separators (`-`, `_`, `.`, spaces)
also works: e.g. `qwen-3.8-27b` selects `qwen3.8-27b`. Other unknown or ambiguous
names fall back to **`qwen3.8-27b`**; it must be configured as a chat backend,
otherwise the request returns 400. Substituted names are rewritten in the upstream
payload. Omitting `model` still selects the first configured chat backend.
Embedding, audio, image, music, and video routing is unchanged.

```bash
# Chat completion — launches llama.cpp on first call
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Image generation — two pathways
# Models: krea2 | wai_illustrious
#
# Raw — your prompt goes straight to the model (no LLM rewrite):
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "krea2",
    "prompt": "A cat in a spacesuit, photorealistic, studio lighting",
    "size": "1024x1024"
  }'

# Augmented — an LLM first rewrites your brief idea into a detailed positive
# prompt (using the image prompt-writing guide), then the rewritten prompt is
# sent to the model. The rewritten prompt is echoed back in the response as
# `augmented_prompt`:
curl http://localhost:8000/v1/images/generations/augmented \
  -H "Content-Type: application/json" \
  -d '{
    "model": "krea2",
    "prompt": "a cat in a spacesuit",
    "size": "1024x1024"
  }'

Image parameters (both pathways):
  • model           — krea2 | wai_illustrious
  • prompt          — Full model prompt (raw) or a brief idea (augmented)
  • negative_prompt — Optional negative prompt (applies to both pathways;
                      omitted → the workflow template's own default)
  • size            — WIDTHxHEIGHT (e.g. "1024x1024")
  • n               — Number of images (default 1)
  • steps / cfg     — Optional sampler overrides
  • seed            — -1 for random, or a specific integer

Response shape:
  Raw       → { "created": <int>, "data": [ { "path": "...png", "comfyui": {…} } ] }
  Augmented → same, plus "augmented_prompt": "<the LLM's rewritten prompt>"

# Music generation — launches ComfyUI, translates to the model's workflow
# Models: ace_step_1.5_xl_turbo (28 GB) | minimax_music_3 (25 GB)
curl http://localhost:8000/v1/music/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimax_music_3",
    "tags": "lo-fi, chill, ambient, soft piano",
    "lyrics": "[Verse]\nWalking through the garden",
    "duration": 60
  }'

Music parameters:
  • tags          — Style descriptors (genre, mood, instrumentation)
                  (MiniMax Music 3: becomes the caption)
  • lyrics        — Structured lyrics with [Verse], [Chorus], etc.
  • bpm           — Tempo (Ace Step only)
  • keyscale      — Musical key (e.g. "C major", "A minor") (Ace Step only)
  • timesignature — Meter: "4" for 4/4, "3" for 3/4, "6" for 6/8 (Ace Step only)
  • language      — Vocal language code (e.g. "en", "ja", "fr") (Ace Step only)
  • duration      — Song length in seconds (MiniMax Music 3: max 360 s)
  • seed          — 0 for random, or a specific integer

ML-specific params (cfg_scale, temperature, top_p, etc.) have sensible
per-model defaults and need not be set manually.

# Video generation — two pathways (MiniMax H3)
# Models: minimax_h3_t2v_turbo | minimax_h3_i2v_turbo | minimax_h3_fl2va
#
# Raw — your prompt goes straight to the model (no LLM rewrite):
curl http://localhost:8000/v1/videos/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimax_h3_i2v_turbo",
    "prompt": "A lighthouse at dusk, waves crashing against the rock",
    "image": "first_frame.png",
    "duration": 3
  }'

# Augmented — an LLM first rewrites your brief idea into a detailed model
# prompt (using the MiniMax H3 guide and any reference frame image(s)), then
# the rewritten description is sent to the model. The rewritten description is
# echoed back in the response as `augmented_prompt`:
curl http://localhost:8000/v1/videos/generations/augmented \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimax_h3_fl2va",
    "prompt": "A lighthouse at dusk becomes a haunted ruin",
    "image": "first_frame.png",
    "last_image": "last_frame.png",
    "duration": 5
  }'

Video parameters:
  • model       — minimax_h3_t2v_turbo (text-to-video),
                  minimax_h3_i2v_turbo (single first-frame reference),
                  minimax_h3_fl2va (first + last frame reference)
  • prompt      — Full model prompt (raw) or a brief idea (augmented)
  • image       — First-frame filename in the ComfyUI input dir (I2V/FL2VA)
  • last_image  — Last-frame filename (FL2VA); an I2V request must include
                  at least one of image / last_image
  • duration    — Seconds (default 3)
  • seed        — 0 for random, or a specific integer

Response shape:
  Raw       → { "created": <int>, "data": [ { "path": "...mp4", "comfyui": {…} } ] }
  Augmented → same, plus "augmented_prompt": "<the LLM's rewritten description>"

(Both endpoints also accept multipart/form-data — fields prompt/model/duration/
seed/…, with `image_file` and `last_image_file` for frame uploads.)

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

## Request queuing (concurrency 1 per model)

Each backend has a concurrency-1 "request slot". A request holds that slot for
the full duration of its work (including the streamed response body for LLM
routes), which marks the backend **busy**. This gives two guarantees:

- **A second request for the same model waits its turn** rather than running
  concurrently — it is serialized behind the in-flight one.
- **The router never evicts a busy backend.** If a new request would otherwise
  need to free VRAM by evicting a backend that still has a request in flight,
  it *waits for that request to finish* and then proceeds — instead of killing
  the in-flight request.

The wait for a busy backend to drain happens outside the orchestrator lock, so
unrelated backends are not blocked while it waits. `GET /status` reports which
backends are currently busy under `busy_backends`.

> Concurrency is fixed at 1 per model for simplicity. Raising it later would
> mean allowing N in-flight requests per backend and evicting only when fewer
> than N are busy.

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

## Model unloading and external-process guards

These administration endpoints coordinate with **all routed model work**:
chat/messages, embeddings, audio, image, music, and video. Read-only router
endpoints (`/status`, `/v1/models`, file downloads, and the UI) remain available.

### Unload all models

```bash
curl -X POST http://localhost:8000/v1/models/unload-all
# {"unloaded":["qwen3.8-27b","image"]}
```

The router temporarily blocks new model work, waits for in-flight request slots
(including streamed responses) to drain, saves supported llama slot caches,
and stops its managed backend processes. This also releases models cached inside
ComfyUI and clears their memory accounting. Already-unloaded backends are harmless:
an idle fleet returns `{"unloaded":[]}`. Stop failures return HTTP 503 rather than
claiming success. External programs and process-guard registrations are untouched.

The operation waits rather than aborting generations; use a sufficiently long
HTTP client timeout. New requests can load models again after unloading finishes,
unless a process guard is still blocking them.

### Register an external memory user

```bash
# Replace 12345 with the actual PID of your program.
curl -X POST http://localhost:8000/v1/process-guards \
  -H 'Content-Type: application/json' \
  -d '{"pid":12345,"timeout":600}'
```

Returns HTTP 201, for example:

```json
{
  "pid": 12345,
  "name": "python",
  "state": "active",
  "blocking": true,
  "registered_at": 1791700000.0,
  "timeout": 600.0,
  "expires_at": 1791700600.0
}
```

- `pid` is required: an integer greater than 1 for an existing process owned by
  the router's OS user. The router, its ancestors, and managed backends (including
  their descendants) cannot be registered. Linux pidfd support is required.
- `timeout` is optional: a finite positive number of **seconds from registration**.
  Omit it or use `null` to block until the process exits.
- Registration returns immediately and blocks **new** model work across backends.
  Existing requests finish normally; registration neither pauses your program
  nor unloads resident models. Augmented pipelines may pause between their LLM
  and image/video stages.
- Natural exit (including a zombie awaiting reaping) automatically removes the
  registration and wakes waiting requests, normally within 0.1 seconds.
- Timeout changes the state to `timed_out` and releases its request block.
  **It does not kill the program.** The registration remains visible and excluded
  from automatic VRAM-hog killing until exit or explicit unregister. Such a process
  can still consume memory, so requests are not guaranteed to fit after timeout.
- Multiple registrations are supported; requests wait until **all** blocking
  guards have exited, timed out, or been removed. Registering the same PID again
  returns HTTP 409; unregister it first to replace its timeout.

For exclusive memory access, start your worker **paused or waiting on a signal**,
register its PID, then call unload-all and wait for the response before allowing
the worker to allocate memory. This prevents new model work from entering the
gap between unloading and registration. Choose a timeout long enough to include
draining/unloading, or omit it. Registering an already-running worker cannot undo
memory contention that has already occurred.

### Inspect, release, or kill a registration

```bash
# List registrations, whether new work is blocked, and in-flight backends.
curl http://localhost:8000/v1/process-guards

# Release the registration WITHOUT killing the program.
curl -X DELETE http://localhost:8000/v1/process-guards/12345

# Explicitly SIGKILL a registered program (including a timed-out registration).
curl -X POST http://localhost:8000/v1/process-guards/12345/kill
# {"pid":12345,"killed":true}
```

The kill endpoint uses the registered **pidfd**, not a numeric-PID signal, so it
cannot accidentally signal a different process after PID reuse. It kills only
that process, **not its process group or descendants**; register each actual
memory-holding worker if your program forks. Requests remain blocked until exit
is confirmed. If confirmation takes longer than five seconds, HTTP 504 is
returned and the guard stays in `killing` state until exit. Unknown registrations
return 404, forbidden ownership returns 403, and invalid registration bodies
return 422.

`GET /status` also includes `process_guards`, `requests_blocked`, and `unloading`;
`busy_backends` continues to describe actual in-flight managed-backend work.
Registrations are in memory only: a router restart removes them without killing
external processes, so re-register still-running programs after a restart.

**Security:** Like the existing unload/restart/hog-kill APIs, these endpoints are
administrative and have no built-in authentication. Keep the router on a trusted
network or protect it with an authenticated reverse proxy. A caller can otherwise
block model access or register and kill programs belonging to the router's user.
