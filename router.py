"""
router.py — Smart LLM router: unified OpenAI-style API over managed backends.

Routes:
  POST /v1/chat/completions      → llama backend (by `model`)   [passthrough]
  POST /v1/messages              → llama backend (by `model`)   [passthrough, Anthropic]
  POST /v1/audio/transcriptions  → CrispASR                     [passthrough]
  POST /v1/audio/speech          → CrispASR                     [passthrough]
  POST /v1/audio/speech-to-speech→ CrispASR                     [passthrough]
  POST /v1/translate             → CrispASR                     [passthrough]
  GET  /v1/voices                → CrispASR                     [passthrough]
  POST /v1/images/generations    → ComfyUI                      [translated]
  GET  /v1/models                → router-level model list
  GET  /status                   → orchestrator fleet status

Backends are launched on demand via the Orchestrator (warm-by-default,
VRAM-pressure eviction) and requests are forwarded once healthy.

Run: python router.py [config.yaml]
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from comfyui import (
    ComfyUIClient,
    ComfyUIError,
    build_openai_response,
    inject_parameters,
    parse_size,
)
from config import RouterConfig, load_config
from orchestrator import InsufficientVRAMError, Orchestrator

logger = logging.getLogger("router")

# Hop-by-hop headers that must not be forwarded either direction.
HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}


def create_app(config: RouterConfig) -> FastAPI:
    orch = Orchestrator(
        config.services,
        total_vram_gb=config.total_vram_gb,
        vram_reserve_gb=config.vram_reserve_gb,
        sysram_reserve_gb=config.sysram_reserve_gb,
        cache_dir=config.cache_dir,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=10)
        )
        yield
        await app.state.http.close()
        await orch.shutdown()

    app = FastAPI(title="Smart LLM Router", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.orch = orch
    app.state.config = config

    # ── Exception handler (catches unhandled route errors) ───────────────

    @app.exception_handler(Exception)
    async def catch_all(request: Request, exc: Exception):
        import traceback
        logger.error("Unhandled exception on %s %s:\n%s", request.method, request.url.path, traceback.format_exc())
        return error(500, f"Internal server error: {exc}")

    # ── Helpers ──────────────────────────────────────────────────────────

    def error(status: int, message: str) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": {"message": message}})

    async def proxy(request: Request, backend_name: str, body: bytes | None = None) -> StreamingResponse:
        """Ensure the backend is up, then stream the request through to it."""
        logger.debug("[%s] Proxying %s %s (body=%s bytes)", backend_name, request.method, request.url.path, len(body) if body else 0)
        try:
            loader = await orch.ensure_running(backend_name)
        except InsufficientVRAMError as exc:
            logger.error("[%s] Insufficient VRAM: %s", backend_name, exc)
            return error(507, str(exc))
        except (KeyError, RuntimeError) as exc:
            logger.error("[%s] Backend error: %s", backend_name, exc)
            return error(503, str(exc))

        port = loader.config.port
        url = f"http://127.0.0.1:{port}{request.url.path}"
        if request.url.query:
            url += f"?{request.url.query}"
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
        if body is None:
            body = await request.body()

        http: aiohttp.ClientSession = app.state.http
        try:
            resp = await http.request(request.method, url, data=body, headers=headers)
        except aiohttp.ClientError as exc:
            logger.error("[%s] Backend request failed: %s", backend_name, exc)
            return error(503, f"Backend '{backend_name}' unreachable: {exc}")

        logger.debug("[%s] Backend responded: %d", backend_name, resp.status)

        async def stream():
            try:
                async for chunk in resp.content.iter_any():
                    yield chunk
            finally:
                resp.release()

        out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in HOP_HEADERS}
        return StreamingResponse(stream(), status_code=resp.status, headers=out_headers)

    async def body_model(request: Request) -> tuple[bytes, str | None]:
        """Read the body and extract the `model` field (JSON requests)."""
        body = await request.body()
        try:
            return body, json.loads(body).get("model")
        except (json.JSONDecodeError, AttributeError):
            return body, None

    # ── LLM routes (passthrough, routed by model) ────────────────────────

    @app.post("/v1/chat/completions")
    @app.post("/v1/messages")
    async def llm(request: Request):
        body, model = await body_model(request)
        try:
            backend = config.resolve_llama(model)
        except KeyError as exc:
            return error(400, str(exc))
        return await proxy(request, backend, body)

    # ── Audio routes (passthrough, routed by model) ──────────────────────

    async def audio_model(request: Request) -> tuple[bytes, str | None]:
        """Read body and extract the `model` field (JSON or multipart)."""
        body = await request.body()
        # Try JSON first
        try:
            model = json.loads(body).get("model")
            if model:
                return body, model
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            pass
        # Try multipart form-data: name="model" followed by value on next line
        try:
            import re
            m = re.search(rb'name="model"\r?\n\r?\n([^\r\n]+)', body)
            if m:
                return body, m.group(1).decode("utf-8", errors="replace")
        except Exception:
            pass
        return body, None  # Let resolve_audio pick the default

    @app.post("/v1/audio/transcriptions")
    @app.post("/v1/audio/speech")
    @app.post("/v1/audio/speech-to-speech")
    @app.post("/v1/translate")
    async def audio(request: Request):
        if not config.audio_backends:
            return error(400, "No audio backends configured")
        body, model = await audio_model(request)
        backend = config.resolve_audio(model)
        logger.info("Audio route: model=%r → backend=%s", model, backend)
        return await proxy(request, backend, body)

    @app.get("/v1/voices")
    async def voices(request: Request):
        if not config.audio_backends:
            return error(400, "No audio backends configured")
        # Route to the first TTS-capable backend (customvoice or voicedesign).
        # STT-only backends (qwen3-asr-*) don't support /v1/voices.
        for name in config.audio_backends:
            if "talker" in name.lower() or "tts" in name.lower():
                return await proxy(request, name)
        # Fallback: try the first backend
        return await proxy(request, config.audio_backends[0])

    # ── Image route (translated) ─────────────────────────────────────────

    @app.post("/v1/images/generations")
    async def images(request: Request):
        try:
            req = await request.json()
        except json.JSONDecodeError:
            return error(400, "Invalid JSON body")

        prompt = req.get("prompt")
        if not prompt:
            return error(400, "Missing required field: prompt")

        try:
            img_model = config.resolve_image_model(req.get("model"))
        except KeyError as exc:
            return error(400, str(exc))

        try:
            workflow = json.loads(Path(img_model.workflow).read_text())
            width, height = parse_size(req.get("size"))
            workflow = inject_parameters(
                workflow,
                prompt=prompt,
                negative_prompt=req.get("negative_prompt"),
                width=width,
                height=height,
                batch_size=int(req.get("n", 1)),
                seed=req.get("seed"),
                steps=req.get("steps"),
                cfg=req.get("cfg"),
            )
        except (OSError, json.JSONDecodeError) as exc:
            return error(500, f"Workflow '{img_model.workflow}' unavailable: {exc}")
        except ComfyUIError as exc:
            return error(400, str(exc))

        # Launch/reuse ComfyUI with room for this specific model's VRAM.
        try:
            loader = await orch.ensure_running(img_model.backend, extra_vram_gb=img_model.vram_gb)
        except InsufficientVRAMError as exc:
            return error(507, str(exc))
        except (KeyError, RuntimeError) as exc:
            return error(503, str(exc))

        client = ComfyUIClient(
            port=loader.config.port,
            output_dir=config.comfyui_output_dirs.get(img_model.backend),
        )
        try:
            results = await client.generate(workflow)
        except ComfyUIError as exc:
            return error(502, f"Image generation failed: {exc}")
        except aiohttp.ClientError as exc:
            return error(503, f"ComfyUI unreachable: {exc}")

        # Model is now resident in the warm ComfyUI process — track its VRAM.
        orch.note_extra_vram(img_model.backend, img_model.vram_gb)

        return build_openai_response(
            results,
            created=int(time.time()),
            response_format=req.get("response_format", "path"),
        )

    # ── Ops routes ───────────────────────────────────────────────────────

    @app.get("/v1/models")
    async def models():
        data = [{"id": n, "object": "model", "owned_by": "llama"} for n in config.llama_backends]
        data += [{"id": n, "object": "model", "owned_by": "crispasr"} for n in config.audio_backends]
        data += [{"id": n, "object": "model", "owned_by": "comfyui"} for n in config.image_models]
        return {"object": "list", "data": data}

    @app.get("/status")
    async def status():
        return await orch.get_status()

    return app


def main() -> None:
    import uvicorn

    log_file = Path("router.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_file, mode="a"),
        ],
    )
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


def get_app() -> FastAPI:
    """Zero-arg factory for uvicorn --factory mode."""
    config = load_config("config.yaml")
    return create_app(config)


if __name__ == "__main__":
    main()
