"""
comfyui.py — Translation wrapper: OpenAI images API ⇄ ComfyUI workflow API.

ComfyUI is not OpenAI-compatible: it takes a JSON workflow graph via
POST /prompt and reports progress over WebSocket. This module:

  1. Loads a preconfigured workflow template (ComfyUI *API format*:
     {node_id: {inputs, class_type}, ...}).
  2. Injects dynamic parameters into well-known node types:
       • KSampler            → seed, steps, cfg
       • its positive/negative links → CLIPTextEncode.text (prompt / negative)
       • EmptyLatentImage    → width, height, batch_size
  3. Submits via POST /prompt, waits for completion on the WebSocket
     (`executing` message with node=None for our prompt_id).
  4. Fetches results from GET /history/{prompt_id} and returns an
     OpenAI-style response. Each image entry carries the local `path`
     where ComfyUI saved the file (plus b64_json when requested).
"""

from __future__ import annotations

import copy
import json
import logging
import random
import time
import uuid
from base64 import b64encode
from pathlib import Path
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

GENERATION_TIMEOUT = 600.0    # Max seconds to wait for a workflow to finish

SAMPLER_TYPES = {"KSampler", "KSamplerAdvanced"}
LATENT_TYPES = {"EmptyLatentImage", "EmptySD3LatentImage"}


class ComfyUIError(Exception):
    """Raised when workflow submission or execution fails."""


# ── Workflow injection ─────────────────────────────────────────────────────

def _linked_node(workflow: dict, link: Any) -> dict | None:
    """Resolve a workflow link ([node_id, slot]) to its target node."""
    if isinstance(link, list) and link:
        return workflow.get(str(link[0]))
    return None


def inject_parameters(
    workflow: dict,
    prompt: str,
    negative_prompt: str | None = None,
    width: int | None = None,
    height: int | None = None,
    batch_size: int = 1,
    seed: int | None = None,
    steps: int | None = None,
    cfg: float | None = None,
) -> dict:
    """
    Return a copy of the workflow with request parameters injected.

    Follows the KSampler's positive/negative links to find the prompt
    encode nodes, so it adapts to any workflow using this common shape.
    """
    wf = copy.deepcopy(workflow)
    injected_prompt = False

    for node in wf.values():
        ctype = node.get("class_type", "")
        inputs = node.get("inputs", {})

        if ctype in SAMPLER_TYPES:
            inputs["seed"] = seed if seed is not None else random.randint(0, 2**48)
            if steps is not None:
                inputs["steps"] = steps
            if cfg is not None:
                inputs["cfg"] = cfg

            pos = _linked_node(wf, inputs.get("positive"))
            if pos is not None and pos.get("class_type") == "CLIPTextEncode":
                pos["inputs"]["text"] = prompt
                injected_prompt = True
            neg = _linked_node(wf, inputs.get("negative"))
            if neg is not None and neg.get("class_type") == "CLIPTextEncode" and negative_prompt is not None:
                neg["inputs"]["text"] = negative_prompt

        elif ctype in LATENT_TYPES:
            if width is not None:
                inputs["width"] = width
            if height is not None:
                inputs["height"] = height
            inputs["batch_size"] = batch_size

    if not injected_prompt:
        raise ComfyUIError(
            "Workflow has no KSampler→CLIPTextEncode positive link; "
            "cannot inject prompt"
        )
    return wf


def parse_size(size: str | None) -> tuple[int | None, int | None]:
    """Parse an OpenAI-style size string ('1024x1024') into (width, height)."""
    if not size:
        return None, None
    try:
        w, h = size.lower().split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        raise ComfyUIError(f"Invalid size '{size}' — expected WIDTHxHEIGHT")


def inject_music_parameters(
    workflow: dict,
    tags: str,
    lyrics: str,
    duration: int = 144,
    bpm: int = 120,
    seed: int | None = None,
    timesignature: str = "4",
    language: str = "en",
    keyscale: str = "E minor",
    cfg_scale: float = 2.0,
    temperature: float = 0.85,
    top_p: float = 0.9,
    top_k: int = 0,
    min_p: float = 0.0,
    steps: int | None = None,
    cfg: float | None = None,
) -> dict:
    """
    Return a copy of the Ace Step music workflow with request parameters injected.

    Updates nodes:
      • KSampler (3)       → seed, steps, cfg
      • PrimitiveInt (109) → seed value (shared seed source)
      • TextEncodeAceStepAudio1.5 (94) → tags, lyrics, seed, bpm, duration, timesignature, language, keyscale, cfg_scale, temperature, top_p, top_k, min_p
      • EmptyAceStep1.5LatentAudio (98) → seconds (duration)
    """
    wf = copy.deepcopy(workflow)
    seed_value = seed if seed is not None and seed != 0 else random.randint(0, 2**48)

    for node in wf.values():
        ctype = node.get("class_type", "")
        inputs = node.get("inputs", {})

        if ctype == "KSampler":
            # seed is a link [node_id, slot] — don't change the slot index,
            # the actual value comes from the linked PrimitiveInt node.
            if steps is not None:
                inputs["steps"] = steps
            if cfg is not None:
                inputs["cfg"] = cfg

        elif ctype == "PrimitiveInt":
            # The shared seed source node — set the actual seed value here.
            inputs["value"] = seed_value

        elif ctype == "TextEncodeAceStepAudio1.5":
            if tags is not None:
                inputs["tags"] = tags
            if lyrics is not None:
                inputs["lyrics"] = lyrics
            inputs["bpm"] = bpm
            inputs["duration"] = duration
            inputs["timesignature"] = timesignature
            inputs["language"] = language
            inputs["keyscale"] = keyscale
            inputs["cfg_scale"] = cfg_scale
            inputs["temperature"] = temperature
            inputs["top_p"] = top_p
            inputs["top_k"] = top_k
            inputs["min_p"] = min_p
            # seed is a link [node_id, slot] — leave it, value comes from PrimitiveInt

        elif ctype == "EmptyAceStep1.5LatentAudio":
            inputs["seconds"] = duration

    return wf


def build_music_openai_response(
    audios: list[dict[str, Any]],
    created: int,
) -> dict[str, Any]:
    """
    Build an OpenAI-style /v1/music/generations response.

    Each data item includes the local `path` where ComfyUI saved the audio file.
    """
    data = []
    for audio in audios:
        item: dict[str, Any] = {}
        if "path" in audio:
            item["path"] = audio["path"]
        item["comfyui"] = {k: audio[k] for k in ("filename", "subfolder", "type") if k in audio}
        data.append(item)
    return {"created": created, "data": data}


# ── Client ─────────────────────────────────────────────────────────────────

class ComfyUIClient:
    """Submits workflows to a running ComfyUI server and collects results."""

    def __init__(self, port: int, output_dir: str | None = None, host: str = "127.0.0.1"):
        self.base = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.output_dir = Path(output_dir) if output_dir else None

    async def generate(self, workflow: dict) -> list[dict[str, Any]]:
        """
        Run a workflow to completion.

        Returns a list of image records: {"path": str, "filename": str, ...}.
        """
        client_id = uuid.uuid4().hex

        timeout = aiohttp.ClientTimeout(total=GENERATION_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Connect the WebSocket *before* submitting so we can't miss
            # the completion message of a fast workflow.
            async with session.ws_connect(f"{self.ws_url}?clientId={client_id}") as ws:
                prompt_id = await self._submit(session, workflow, client_id)
                await self._wait_for_completion(ws, prompt_id)
            return await self._collect_results(session, prompt_id)

    async def _submit(self, session: aiohttp.ClientSession, workflow: dict, client_id: str) -> str:
        payload = {"prompt": workflow, "client_id": client_id}
        async with session.post(f"{self.base}/prompt", json=payload) as resp:
            body = await resp.json()
            if resp.status != 200:
                raise ComfyUIError(f"POST /prompt failed ({resp.status}): {body}")
            if "error" in body:
                raise ComfyUIError(f"Workflow rejected: {body['error']}")
            prompt_id = body.get("prompt_id")
            if not prompt_id:
                raise ComfyUIError(f"No prompt_id in response: {body}")
            logger.info("ComfyUI accepted workflow, prompt_id=%s", prompt_id)
            return prompt_id

    async def _wait_for_completion(self, ws: aiohttp.ClientWSResponse, prompt_id: str) -> None:
        """Listen on the WebSocket until our prompt finishes (or errors)."""
        t0 = time.monotonic()
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue  # Binary preview frames — ignore
            data = json.loads(msg.data)
            mtype = data.get("type")
            payload = data.get("data", {})

            if payload.get("prompt_id") != prompt_id:
                continue
            if mtype == "execution_error":
                elapsed = time.monotonic() - t0
                logger.error(
                    "ComfyUI execution_error for prompt_id=%s after %.1fs: %s",
                    prompt_id, elapsed, payload,
                )
                raise ComfyUIError(f"Workflow execution error: {payload}")
            if mtype == "execution_interrupted":
                elapsed = time.monotonic() - t0
                logger.error(
                    "ComfyUI execution_interrupted for prompt_id=%s after %.1fs",
                    prompt_id, elapsed,
                )
                raise ComfyUIError("Workflow execution was interrupted")
            # Completion signal: 'executing' with node=None, or 'execution_success'
            if mtype == "execution_success":
                elapsed = time.monotonic() - t0
                logger.info(
                    "ComfyUI execution_success for prompt_id=%s after %.1fs",
                    prompt_id, elapsed,
                )
                return
            if mtype == "executing" and payload.get("node") is None:
                elapsed = time.monotonic() - t0
                logger.info(
                    "ComfyUI completed (executing=None) for prompt_id=%s after %.1fs",
                    prompt_id, elapsed,
                )
                return
        elapsed = time.monotonic() - t0
        logger.error(
            "ComfyUI WebSocket closed before workflow completed for prompt_id=%s after %.1fs "
            "(possible ComfyUI crash/OOM during generation)",
            prompt_id, elapsed,
        )
        raise ComfyUIError(
            f"WebSocket closed after {elapsed:.0f}s before workflow completed "
            f"(prompt_id={prompt_id}) — ComfyUI may have crashed"
        )

    async def _collect_results(self, session: aiohttp.ClientSession, prompt_id: str) -> list[dict[str, Any]]:
        """Fetch saved image records from /history and resolve local paths."""
        async with session.get(f"{self.base}/history/{prompt_id}") as resp:
            if resp.status != 200:
                raise ComfyUIError(f"GET /history failed ({resp.status})")
            history = await resp.json()

        entry = history.get(prompt_id, {})
        images: list[dict[str, Any]] = []
        for node_output in entry.get("outputs", {}).values():
            for img in node_output.get("images", []):
                if img.get("type") != "output":
                    continue  # Skip temp/preview images
                record = dict(img)  # filename, subfolder, type
                if self.output_dir is not None:
                    record["path"] = str(
                        self.output_dir / img.get("subfolder", "") / img["filename"]
                    )
                images.append(record)

        if not images:
            raise ComfyUIError(f"Workflow {prompt_id} completed but produced no output images")
        return images

    async def generate_audio(self, workflow: dict) -> list[dict[str, Any]]:
        """
        Run a workflow to completion and collect audio outputs.

        Returns a list of audio records: {"path": str, "filename": str, ...}.
        """
        client_id = uuid.uuid4().hex

        timeout = aiohttp.ClientTimeout(total=GENERATION_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(f"{self.ws_url}?clientId={client_id}") as ws:
                prompt_id = await self._submit(session, workflow, client_id)
                await self._wait_for_completion(ws, prompt_id)
            return await self._collect_audio_results(session, prompt_id)

    async def _collect_audio_results(self, session: aiohttp.ClientSession, prompt_id: str) -> list[dict[str, Any]]:
        """Fetch saved audio records from /history and resolve local paths."""
        async with session.get(f"{self.base}/history/{prompt_id}") as resp:
            if resp.status != 200:
                raise ComfyUIError(f"GET /history failed ({resp.status})")
            history = await resp.json()

        entry = history.get(prompt_id, {})
        audios: list[dict[str, Any]] = []
        for node_output in entry.get("outputs", {}).values():
            # Audio nodes output under "audio" key (e.g. SaveAudioMP3)
            for audio in node_output.get("audio", []):
                if audio.get("type") != "output":
                    continue
                record = dict(audio)
                if self.output_dir is not None:
                    record["path"] = str(
                        self.output_dir / audio.get("subfolder", "") / audio["filename"]
                    )
                audios.append(record)

        if not audios:
            raise ComfyUIError(f"Workflow {prompt_id} completed but produced no output audio")
        return audios


# ── OpenAI-style response assembly ─────────────────────────────────────────

def build_openai_response(
    images: list[dict[str, Any]],
    created: int,
    response_format: str = "path",
) -> dict[str, Any]:
    """
    Build an OpenAI /v1/images/generations style response.

    Each data item always includes the local `path` where ComfyUI saved the
    image. When response_format='b64_json' and the file is readable, the
    base64 payload is included as well.
    """
    data = []
    for img in images:
        item: dict[str, Any] = {}
        if "path" in img:
            item["path"] = img["path"]
        if response_format == "b64_json" and "path" in img:
            try:
                item["b64_json"] = b64encode(Path(img["path"]).read_bytes()).decode()
            except OSError as exc:
                logger.warning("Could not read image %s for b64: %s", img["path"], exc)
        item["comfyui"] = {k: img[k] for k in ("filename", "subfolder", "type") if k in img}
        data.append(item)
    return {"created": created, "data": data}
