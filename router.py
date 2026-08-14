"""
router.py — Smart LLM router: unified OpenAI-style API over managed backends.

Routes:
  POST /v1/chat/completions      → llama backend (by `model`)   [passthrough]
  POST /v1/messages              → llama backend (by `model`)   [passthrough, Anthropic]
  POST /v1/embeddings            → llama backend (by `model`)   [passthrough, embedding-only]
  POST /v1/audio/transcriptions  → CrispASR                     [passthrough]
  POST /v1/audio/speech          → CrispASR                     [passthrough]
  POST /v1/audio/speech-to-speech→ CrispASR                     [passthrough]
  POST /v1/translate             → CrispASR                     [passthrough]
  GET  /v1/voices                → CrispASR                     [passthrough]
  POST /v1/images/generations    → ComfyUI                      [translated]
  POST /v1/music/generations     → ComfyUI                      [translated]
  POST /v1/videos/generations    → ComfyUI                      [translated, raw]
  POST /v1/videos/generations/augmented → LLM + ComfyUI        [augmented pipeline]
  GET  /v1/models                → router-level model list
  GET  /status                   → orchestrator fleet status

Backends are launched on demand via the Orchestrator (warm-by-default,
VRAM-pressure eviction) and requests are forwarded once healthy.

Run: python router.py [config.yaml]
"""

from __future__ import annotations

import atexit
import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from src.comfyui import (
    ComfyUIClient,
    ComfyUIError,
    build_music_openai_response,
    build_openai_response,
    build_video_openai_response,
    inject_music_parameters,
    inject_parameters,
    inject_video_parameters,
    parse_size,
)
from src.config import RouterConfig, load_config
from src.orchestrator import InsufficientVRAMError, Orchestrator

logger = logging.getLogger("router")

# Hop-by-hop headers that must not be forwarded either direction.
HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}


def _install_orphan_sweeper(orch: "Orchestrator") -> None:
    """
    Register a last-resort atexit hook that SIGKILLs any backend process
    groups still alive when the router interpreter exits.

    This runs after the async lifespan shutdown (so gracefully-stopped
    backends are already gone and unaffected), and covers the paths where
    graceful shutdown never ran — interpreter error, uvicorn force-exit —
    that would otherwise orphan GPU-holding backends.
    """
    def sweep() -> None:
        import signal as _signal
        for loader in orch.services.values():
            proc = loader._process
            if proc is not None and proc.returncode is None:
                try:
                    os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
    atexit.register(sweep)


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
        # Check for unmanaged VRAM consumers on startup.
        await orch.check_unmanaged_vram()
        orch.start_vram_monitor()
        yield
        await app.state.http.close()
        await orch.shutdown()

    _install_orphan_sweeper(orch)
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

    # ── Embedding routes (passthrough, routed by model) ──────────────────

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        if not config.embedding_backends:
            return error(400, "No embedding backends configured")
        body, model = await body_model(request)
        try:
            backend = config.resolve_embedding(model)
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
        # Route to the first TTS-capable backend (Qwen-Talker, Kokoro, ...).
        # STT-only backends (qwen3-asr-*) don't support /v1/voices.
        if config.tts_backends:
            return await proxy(request, config.tts_backends[0])
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

    # ── Music route (translated) ─────────────────────────────────────────

    @app.post("/v1/music/generations")
    async def music(request: Request):
        try:
            req = await request.json()
        except json.JSONDecodeError:
            return error(400, "Invalid JSON body")

        tags = req.get("tags")
        lyrics = req.get("lyrics")
        if not tags and not lyrics:
            return error(400, "Missing required fields: at least one of 'tags' or 'lyrics' is required")

        try:
            music_model = config.resolve_music_model(req.get("model"))
        except KeyError as exc:
            return error(400, str(exc))

        try:
            workflow = json.loads(Path(music_model.workflow).read_text())
            workflow = inject_music_parameters(
                workflow,
                tags=tags,
                lyrics=lyrics,
                duration=req.get("duration", 144),
                bpm=req.get("bpm", 120),
                seed=req.get("seed"),
                timesignature=req.get("timesignature", "4"),
                language=req.get("language", "en"),
                keyscale=req.get("keyscale", "E minor"),
                cfg_scale=req.get("cfg_scale", 2.0),
                temperature=req.get("temperature", 0.85),
                top_p=req.get("top_p", 0.9),
                top_k=req.get("top_k", 0),
                min_p=req.get("min_p", 0.0),
                steps=req.get("steps"),
                cfg=req.get("cfg"),
            )
        except (OSError, json.JSONDecodeError) as exc:
            return error(500, f"Workflow '{music_model.workflow}' unavailable: {exc}")
        except ComfyUIError as exc:
            return error(400, str(exc))

        # Launch/reuse ComfyUI with room for this specific model's VRAM.
        try:
            loader = await orch.ensure_running(music_model.backend, extra_vram_gb=music_model.vram_gb)
        except InsufficientVRAMError as exc:
            return error(507, str(exc))
        except (KeyError, RuntimeError) as exc:
            return error(503, str(exc))

        client = ComfyUIClient(
            port=loader.config.port,
            output_dir=config.comfyui_output_dirs.get(music_model.backend),
        )
        t0 = time.monotonic()
        try:
            results = await client.generate_audio(workflow)
            elapsed = time.monotonic() - t0
            logger.info("Music generation completed in %.1fs", elapsed)
        except ComfyUIError as exc:
            elapsed = time.monotonic() - t0
            logger.error("Music generation failed after %.1fs: %s", elapsed, exc)
            return error(502, f"Music generation failed: {exc}")
        except aiohttp.ClientError as exc:
            elapsed = time.monotonic() - t0
            logger.error("Music generation unreachable after %.1fs: %s", elapsed, exc)
            return error(503, f"ComfyUI unreachable: {exc}")

        # Model is now resident in the warm ComfyUI process — track its VRAM.
        orch.note_extra_vram(music_model.backend, music_model.vram_gb)

        return build_music_openai_response(
            results,
            created=int(time.time()),
        )

    # ── Video routes ─────────────────────────────────────────────────────

    # The Minimax H3 prompt-writing guide — read once at startup.
    _VIDEO_GUIDE_PATH = Path(__file__).parent / "untracked" / "minimax_h3_guide.md"
    _VIDEO_GUIDE = _VIDEO_GUIDE_PATH.read_text() if _VIDEO_GUIDE_PATH.exists() else ""

    async def _do_video_generation(
        req: dict,
        prompt: str,
        image_filename: str | None = None,
    ) -> JSONResponse | StreamingResponse:
        """Core video generation: inject params, run ComfyUI workflow, return response."""
        try:
            video_model = config.resolve_video_model(req.get("model"))
        except KeyError as exc:
            return error(400, str(exc))

        try:
            workflow = json.loads(Path(video_model.workflow).read_text())
            workflow = inject_video_parameters(
                workflow,
                prompt=prompt,
                duration=req.get("duration", 3.0),
                seed=req.get("seed"),
                image=image_filename,
                megapixels=req.get("megapixels"),
                width=req.get("width"),
                height=req.get("height"),
            )
        except (OSError, json.JSONDecodeError) as exc:
            return error(500, f"Workflow '{video_model.workflow}' unavailable: {exc}")
        except ComfyUIError as exc:
            return error(400, str(exc))

        try:
            loader = await orch.ensure_running(video_model.backend, extra_vram_gb=video_model.vram_gb)
        except InsufficientVRAMError as exc:
            return error(507, str(exc))
        except (KeyError, RuntimeError) as exc:
            return error(503, str(exc))

        client = ComfyUIClient(
            port=loader.config.port,
            output_dir=config.comfyui_output_dirs.get(video_model.backend),
        )
        t0 = time.monotonic()
        try:
            results = await client.generate_video(workflow)
            elapsed = time.monotonic() - t0
            logger.info("Video generation completed in %.1fs", elapsed)
        except ComfyUIError as exc:
            elapsed = time.monotonic() - t0
            logger.error("Video generation failed after %.1fs: %s", elapsed, exc)
            return error(502, f"Video generation failed: {exc}")
        except aiohttp.ClientError as exc:
            elapsed = time.monotonic() - t0
            logger.error("Video generation unreachable after %.1fs: %s", elapsed, exc)
            return error(503, f"ComfyUI unreachable: {exc}")

        orch.note_extra_vram(video_model.backend, video_model.vram_gb)
        return build_video_openai_response(results, created=int(time.time()))

    @app.post("/v1/videos/generations")
    async def videos(request: Request):
        """Raw video generation — prompt sent directly to ComfyUI (no LLM augmentation).
        
        Accepts either JSON body or multipart form-data.
        For multipart: fields are 'prompt', 'model', 'duration', 'seed', 'megapixels',
        'width', 'height', and 'image_file' (uploaded image for I2V).
        For JSON: 'image' field specifies a filename already in ComfyUI input dir.
        """
        content_type = request.headers.get("content-type", "")

        if "multipart/form-data" in content_type:
            # Multipart form-data with potential file upload
            form = await request.form()
            prompt = form.get("prompt")
            if not prompt:
                return error(400, "Missing required field: prompt")

            req = {
                "model": form.get("model"),
                "duration": float(form["duration"]) if form.get("duration") else None,
                "seed": int(form["seed"]) if form.get("seed") else None,
                "megapixels": float(form["megapixels"]) if form.get("megapixels") else None,
                "width": int(form["width"]) if form.get("width") else None,
                "height": int(form["height"]) if form.get("height") else None,
            }

            # Handle uploaded image file
            image_file = form.get("image_file")
            image_filename = None
            if image_file and hasattr(image_file, "filename") and image_file.filename:
                # Save uploaded file to ComfyUI input directory
                video_model = config.resolve_video_model(req.get("model"))
                comfyui_output_dir = config.comfyui_output_dirs.get(video_model.backend)
                if comfyui_output_dir:
                    input_dir = Path(comfyui_output_dir).parent / "input"
                    input_dir.mkdir(parents=True, exist_ok=True)
                    image_filename = image_file.filename
                    dest = input_dir / image_filename
                    dest.write_bytes(await image_file.read())
                    logger.info("Saved uploaded image to %s", dest)
        else:
            try:
                req = await request.json()
            except json.JSONDecodeError:
                return error(400, "Invalid JSON body")

            prompt = req.get("prompt")
            if not prompt:
                return error(400, "Missing required field: prompt")

            image_filename = req.get("image")

        return await _do_video_generation(req, prompt, image_filename)

    @app.post("/v1/videos/generations/augmented")
    async def videos_augmented(request: Request):
        """
        Augmented video generation — prompt first enriched by an LLM call
        (qwen3.6-27b-instruct) using the MiniMax H3 prompt-writing guide,
        then sent to ComfyUI for generation.
        
        Accepts either JSON body or multipart form-data.
        For multipart: fields are 'prompt', 'model', 'duration', 'seed', 'megapixels',
        'width', 'height', and 'image_file' (uploaded image for I2V).
        For JSON: 'image' field specifies a filename already in ComfyUI input dir.
        """
        content_type = request.headers.get("content-type", "")
        image_filename = None

        if "multipart/form-data" in content_type:
            form = await request.form()
            prompt = form.get("prompt")
            if not prompt:
                return error(400, "Missing required field: prompt")

            req = {
                "model": form.get("model"),
                "duration": float(form["duration"]) if form.get("duration") else None,
                "seed": int(form["seed"]) if form.get("seed") else None,
                "megapixels": float(form["megapixels"]) if form.get("megapixels") else None,
                "width": int(form["width"]) if form.get("width") else None,
                "height": int(form["height"]) if form.get("height") else None,
            }

            # Handle uploaded image file
            image_file = form.get("image_file")
            if image_file and hasattr(image_file, "filename") and image_file.filename:
                video_model = config.resolve_video_model(req.get("model"))
                comfyui_output_dir = config.comfyui_output_dirs.get(video_model.backend)
                if comfyui_output_dir:
                    input_dir = Path(comfyui_output_dir).parent / "input"
                    input_dir.mkdir(parents=True, exist_ok=True)
                    image_filename = image_file.filename
                    dest = input_dir / image_filename
                    dest.write_bytes(await image_file.read())
                    logger.info("Saved uploaded image to %s", dest)
        else:
            try:
                req = await request.json()
            except json.JSONDecodeError:
                return error(400, "Invalid JSON body")

            prompt = req.get("prompt")
            if not prompt:
                return error(400, "Missing required field: prompt")

            image_filename = req.get("image")

        # Resolve model to determine mode (T2VA vs I2VA).
        try:
            video_model = config.resolve_video_model(req.get("model"))
        except KeyError as exc:
            return error(400, str(exc))

        is_i2v = "i2v" in video_model.name.lower()
        mode_label = "I2VA" if is_i2v else "T2VA"

        # ── Augment via LLM ──────────────────────────────────────────────
        augment_backend = "qwen3.6-27b-instruct"

        system_msg = (
            f"You are a video prompt engineer. Rewrite the user's brief idea into a "
            f"detailed, creative {mode_label} prompt for the MiniMax H3 video model. "
            f"Follow the guide below exactly.\n\n"
            f"## MiniMax H3 Video Prompt Writing Guide\n\n"
            f"{_VIDEO_GUIDE}\n\n"
            f"## Instructions\n\n"
            f"- Produce a highly detailed, vivid, cinematic prompt.\n"
            f"- Use temperature 1.0 — be creative and imaginative.\n"
            f"- Output ONLY the final prompt (instruction line if I2VA, then the three "
            f"core fields). Do not include any explanation or markdown wrapping.\n"
            f"- For I2VA: begin with the first-frame instruction, then a blank line, "
            f"then the three core fields. Reference <Picture 1> in the description.\n"
            f"- For T2VA: begin directly with the three core fields (no instruction line).\n"
            f"- The video duration is {req.get('duration', 3.0)} seconds.\n"
        )

        user_msg = f"Create a {mode_label} video from this idea:\n\n{prompt}"

        llm_request = {
            "model": augment_backend,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 1.0,
            "max_tokens": 1024,
        }

        try:
            loader = await orch.ensure_running(augment_backend)
        except InsufficientVRAMError as exc:
            return error(507, str(exc))
        except (KeyError, RuntimeError) as exc:
            return error(503, str(exc))

        llm_port = loader.config.port
        llm_url = f"http://127.0.0.1:{llm_port}/v1/chat/completions"
        http: aiohttp.ClientSession = app.state.http

        try:
            async with http.post(llm_url, json=llm_request) as llm_resp:
                llm_body = await llm_resp.json()
                if llm_resp.status != 200:
                    return error(502, f"Prompt augmentation failed: {llm_body}")
                augmented_prompt = llm_body.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if not augmented_prompt:
                    return error(502, "Prompt augmentation returned empty response")
        except aiohttp.ClientError as exc:
            return error(503, f"LLM backend unreachable for prompt augmentation: {exc}")

        logger.info("Augmented video prompt (%d chars): %.200s…", len(augmented_prompt), augmented_prompt)

        return await _do_video_generation(req, augmented_prompt, image_filename)

    # ── Ops routes ───────────────────────────────────────────────────────

    @app.get("/")
    async def frontend():
        """Serve the interactive test console frontend."""
        html_path = Path(__file__).parent / "frontend.html"
        return HTMLResponse(
            content=html_path.read_text(),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    @app.get("/mobile")
    async def mobile():
        """Serve the mobile-friendly frontend."""
        html_path = Path(__file__).parent / "mobile.html"
        return HTMLResponse(
            content=html_path.read_text(),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    _MIME_MAP = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".gif": "image/gif", ".wav": "audio/wav",
        ".mp3": "audio/mpeg", ".mp4": "video/mp4",
    }

    @app.get("/v1/data/{file_path:path}")
    async def file_data(file_path: str):
        """Return a generated file (image/audio) as base64 JSON for inline display."""
        import base64 as b64
        import urllib.parse

        def _serve(path: Path) -> JSONResponse:
            data = b64.b64encode(path.read_bytes()).decode()
            ct = _MIME_MAP.get(path.suffix.lower(), "application/octet-stream")
            return JSONResponse(content={"b64_json": data, "content_type": ct})

        decoded = urllib.parse.unquote(file_path)
        logger.debug("file_data: file_path=%r, decoded=%r", file_path, decoded)
        logger.debug("file_data: comfyui_output_dirs=%s", list(config.comfyui_output_dirs.values()))
        # Try as absolute path first
        abs_path = Path(decoded)
        logger.debug("file_data: abs_path=%s, exists=%s", abs_path, abs_path.is_file())
        if abs_path.is_file():
            return _serve(abs_path)
        # Search configured ComfyUI output dirs
        for output_dir in config.comfyui_output_dirs.values():
            for candidate in (Path(output_dir) / decoded, Path(output_dir) / file_path):
                logger.debug("file_data: candidate=%s, exists=%s", candidate, candidate.is_file())
                if candidate.is_file():
                    return _serve(candidate)
        return error(404, f"File not found: {decoded}")

    @app.get("/v1/models")
    async def models():
        data = [{"id": n, "object": "model", "owned_by": "llama"} for n in config.llama_backends]
        data += [{"id": n, "object": "model", "owned_by": "crispasr"} for n in config.audio_backends]
        data += [{"id": n, "object": "model", "owned_by": "llama", "type": "embedding"} for n in config.embedding_backends]
        data += [{"id": n, "object": "model", "owned_by": "comfyui"} for n in config.image_models]
        data += [{"id": n, "object": "model", "owned_by": "comfyui", "type": "music"} for n in config.music_models]
        data += [{"id": n, "object": "model", "owned_by": "comfyui", "type": "video"} for n in config.video_models]
        return {"object": "list", "data": data}

    @app.get("/status")
    async def status():
        return await orch.get_status()

    @app.post("/v1/models/unload-all")
    async def unload_all():
        """Unload all running backends. Useful for test teardown."""
        running = orch._running()
        if running:
            logger.info("Unload-all: stopping %d backend(s)", len(running))
            await asyncio.gather(*(s.stop() for s in running))
        return {"unloaded": [s.config.name for s in running]}

    @app.post("/v1/models/restart-router")
    async def restart_router():
        """
        Gracefully shut down all backends and exit the router process.

        The watchdog will detect the exit and restart the router with a
        clean slate (no loaded models, fresh process). Useful for tests
        and maintenance windows.
        """
        running = orch._running()
        if running:
            logger.info("Restart-router: stopping %d backend(s) before exit", len(running))
            await asyncio.gather(*(s.stop() for s in running))
        logger.info("Restart-router: shutting down router process (watchdog will restart)")

        async def _exit():
            # Brief pause to let the response flush to the client.
            await asyncio.sleep(0.5)
            os._exit(0)

        asyncio.create_task(_exit())
        return {"restarted": True, "stopped": [s.config.name for s in running]}

    @app.get("/v1/models/vram-hogs")
    async def vram_hogs():
        """List external GPU processes consuming VRAM (not managed by the router)."""
        hogs = await orch.detect_vram_hogs()
        return {
            "hogs": [
                {"pid": pid, "vram_gb": round(gb, 2), "name": name}
                for pid, gb, name in hogs
            ],
            "total_hog_vram_gb": round(sum(gb for _, gb, _ in hogs), 2),
        }

    @app.post("/v1/models/kill-vram-hogs")
    async def kill_vram_hogs():
        """Kill external GPU processes that are consuming VRAM (not managed backends)."""
        freed = await orch.kill_vram_hogs()
        return {
            "killed": freed > 0,
            "freed_gb": round(freed, 2),
        }

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
    # timeout_graceful_shutdown bounds how long uvicorn waits for in-flight
    # (e.g. streaming) requests before forcing exit — without it a hung stream
    # can stall shutdown past the watchdog's patience, and the ensuing SIGKILL
    # would abort backend cleanup mid-flight.
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
        timeout_graceful_shutdown=10,
    )


def get_app() -> FastAPI:
    """Zero-arg factory for uvicorn --factory mode."""
    config = load_config("config.yaml")
    return create_app(config)


if __name__ == "__main__":
    main()
