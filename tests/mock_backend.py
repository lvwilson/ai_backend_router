#!/usr/bin/env python3
"""
mock_backend.py — Impersonates llama-server / CrispASR / ComfyUI for e2e tests.

Launched by the orchestrator like a real backend. Mode selects behavior:

  --mode llama     GET /health, POST /v1/chat/completions (echo)
  --mode crispasr  GET /v1/models, POST /v1/audio/transcriptions
  --mode comfyui   GET /system_stats, POST /prompt, WS /ws, GET /history/{id},
                   GET /last_workflow (test introspection); writes a real
                   image file into --output-dir and reports it via history.

Unknown CLI flags (e.g. llama's -m/-c/-ngl) are ignored so this script can
be dropped in as the `binary` for any backend type.
"""

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

from aiohttp import web, WSMsgType


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["llama", "crispasr", "comfyui"])
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--output-dir", default="/tmp/mock_comfyui_output")
    args, _unknown = p.parse_known_args()
    return args


# ── llama mock ─────────────────────────────────────────────────────────────

def llama_app() -> web.Application:
    app = web.Application()

    async def health(_):
        return web.json_response({"status": "ok"})

    async def chat(request):
        body = await request.json()
        return web.json_response({
            "object": "chat.completion",
            "model": body.get("model", "mock"),
            "choices": [{"index": 0, "message": {
                "role": "assistant",
                "content": f"mock-llama-reply: {body['messages'][-1]['content']}",
            }, "finish_reason": "stop"}],
        })

    app.router.add_get("/health", health)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/v1/messages", chat)
    return app


# ── crispasr mock ──────────────────────────────────────────────────────────

def crispasr_app() -> web.Application:
    app = web.Application()

    async def models(_):
        return web.json_response({"data": [{"id": "mock-parakeet"}]})

    async def transcribe(request):
        await request.read()
        return web.json_response({"text": "mock transcription result"})

    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/audio/transcriptions", transcribe)
    return app


# ── comfyui mock ───────────────────────────────────────────────────────────

def comfyui_app(output_dir: str) -> web.Application:
    app = web.Application()
    app["completed"] = asyncio.Queue()   # prompt_ids awaiting WS announcement
    app["last_workflow"] = None
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    async def system_stats(_):
        return web.json_response({"system": {"os": "mock"}})

    async def prompt(request):
        body = await request.json()
        app["last_workflow"] = body["prompt"]
        prompt_id = uuid.uuid4().hex
        # "Generate" the image immediately.
        (out / f"{prompt_id}.png").write_bytes(b"\x89PNG mock image data")
        await app["completed"].put(prompt_id)
        return web.json_response({"prompt_id": prompt_id, "number": 1})

    async def ws_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        # Announce completion of the next submitted prompt, then hold open.
        prompt_id = await app["completed"].get()
        await ws.send_json({"type": "executing", "data": {"node": None, "prompt_id": prompt_id}})
        async for msg in ws:
            if msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                break
        return ws

    async def history(request):
        pid = request.match_info["pid"]
        return web.json_response({pid: {"outputs": {"23": {"images": [
            {"filename": f"{pid}.png", "subfolder": "", "type": "output"}
        ]}}}})

    async def last_workflow(_):
        return web.json_response(app["last_workflow"] or {})

    app.router.add_get("/system_stats", system_stats)
    app.router.add_post("/prompt", prompt)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/history/{pid}", history)
    app.router.add_get("/last_workflow", last_workflow)
    return app


def main():
    args = parse_args()
    if args.mode == "llama":
        app = llama_app()
    elif args.mode == "crispasr":
        app = crispasr_app()
    else:
        app = comfyui_app(args.output_dir)
    print(f"mock_backend [{args.mode}] listening on :{args.port}", file=sys.stderr)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
