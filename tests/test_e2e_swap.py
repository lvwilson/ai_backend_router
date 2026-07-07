#!/usr/bin/env python3
"""
test_e2e_swap.py — End-to-end router test: swap between all three backends.

Uses tests/mock_backend.py subprocesses as llama / CrispASR / ComfyUI stand-ins
and the real Krea2 workflow for translation. GPU telemetry is stubbed so VRAM
budgeting uses tracked bookkeeping (deterministic on any machine).

Budget: total 10 GB, reserve 1 GB → 9 GB usable.
  llama-small = 4 GB, asr = 2 GB, comfyui = 1 GB process + 7 GB krea2 model.

Scenario:
  1. Chat        → llama-small launches (4/9 used)
  2. Transcribe  → asr launches (6/9 used)
  3. Image       → needs 8 GB → evicts asr AND llama-small, launches ComfyUI,
                   translates via Krea2 workflow, returns image filepath
  4. Chat again  → needs 4 GB, only 1 free → evicts ComfyUI (8 GB incl. model)

Run: pytest tests/test_e2e_swap.py -v
"""

import asyncio
import json
import os
import signal
import shutil
import sys
import tempfile
from pathlib import Path

import aiohttp
import httpx
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from router import create_app

MOCK = HERE / "mock_backend.py"
WORKFLOW = HERE / "krea2_basic.json"

# Ports used by mock backends in this test — isolated from live backends.
MOCK_PORTS = [18081, 18082, 18083]


def _kill_stale_mocks() -> None:
    """Kill any leftover mock_backend processes from a previous test run."""
    import subprocess
    result = subprocess.run(
        ["pgrep", "-f", str(MOCK)],
        capture_output=True, text=True,
    )
    pids = result.stdout.strip().split()
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGKILL)
        except (ProcessLookupError, ValueError):
            pass
    if pids:
        time.sleep(0.5)


import time


CONFIG_TEMPLATE = """
router:
  total_vram: 10
  vram_reserve: 1

backends:
  llama-small:
    type: llama
    binary: "{mock}"
    model: fake-model.gguf
    port: 18081
    vram_usage: 4
    extra_args: ["--mode", "llama"]

  asr:
    type: crispasr
    binary: "{mock}"
    model: fake-parakeet.gguf
    port: 18082
    vram_usage: 2
    extra_args: ["--mode", "crispasr"]

  image:
    type: comfyui
    venv: python3
    main: "{mock}"
    port: 18083
    vram_usage: 1
    output_dir: "{outdir}"
    extra_args: ["--mode", "comfyui", "--output-dir", "{outdir}"]
    models:
      krea2:
        workflow: "{workflow}"
        vram_usage: 7
"""


@pytest.fixture(scope="module")
def event_loop():
    """Module-scoped event loop for the e2e tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def e2e_env():
    """Create the e2e test environment: config, app, orchestrator, temp dir."""
    _kill_stale_mocks()

    outdir = tempfile.mkdtemp(prefix="router_e2e_")
    cfg_file = Path(outdir) / "config.yaml"
    cfg_file.write_text(CONFIG_TEMPLATE.format(mock=MOCK, outdir=outdir, workflow=WORKFLOW))

    config = load_config(cfg_file)
    app = create_app(config)
    orch = app.state.orch

    yield {
        "app": app,
        "orch": orch,
        "outdir": outdir,
        "config": config,
    }

    shutil.rmtree(outdir, ignore_errors=True)


@pytest.fixture(scope="module")
async def e2e_client(e2e_env):
    """Create an httpx client for the e2e router app, with auto-unload on teardown."""
    app = e2e_env["app"]

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://router", timeout=120) as client:
            yield client

            # Unload all models after tests
            try:
                await client.post("/v1/models/unload-all")
            except Exception:
                pass


def running(orch) -> set[str]:
    return {s.config.name for s in orch.services.values() if s.is_alive}


class TestE2ESwap:
    """End-to-end swap tests: chat → transcribe → image → chat."""

    async def test_chat_launches_llama(self, e2e_client, e2e_env):
        """Chat → llama-small launches."""
        orch = e2e_env["orch"]
        r = await e2e_client.post("/v1/chat/completions", json={
            "model": "llama-small",
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert r.status_code == 200, r.text
        assert "mock-llama-reply: hello" in r.text
        assert running(orch) == {"llama-small"}, f"Expected only llama-small, got {running(orch)}"

    async def test_transcription_launches_asr(self, e2e_client, e2e_env):
        """Transcription → asr joins (both fit)."""
        orch = e2e_env["orch"]
        r = await e2e_client.post("/v1/audio/transcriptions",
                                  files={"file": ("a.wav", b"RIFFdata")})
        assert r.status_code == 200, r.text
        assert r.json().get("text") == "mock transcription result"
        assert running(orch) == {"llama-small", "asr"}, f"Expected llama+asr, got {running(orch)}"

    async def test_image_evicts_llama_and_asr(self, e2e_client, e2e_env):
        """Image → ComfyUI (8 GB) evicts llama + asr."""
        orch = e2e_env["orch"]
        r = await e2e_client.post("/v1/images/generations", json={
            "model": "krea2",
            "prompt": "a red fox in the snow",
            "size": "512x512",
            "steps": 4,
            "seed": 1234,
        })
        assert r.status_code == 200, r.text
        assert running(orch) == {"image"}, f"Expected only comfyui, got {running(orch)}"

        data = r.json()["data"]
        assert len(data) == 1
        path = data[0].get("path", "")
        assert path, json.dumps(data)
        assert Path(path).is_file(), path

        # Verify translation: real Krea2 workflow with our parameters injected.
        async with aiohttp.ClientSession() as s:
            async with s.get("http://127.0.0.1:18083/last_workflow") as resp:
                wf = await resp.json()
        assert wf["6"]["inputs"]["text"] == "a red fox in the snow"
        assert wf["24"]["inputs"]["text"] == "Gridlines"
        assert wf["10"]["inputs"]["width"] == 512 and wf["10"]["inputs"]["height"] == 512
        assert wf["2"]["inputs"]["seed"] == 1234
        assert wf["2"]["inputs"]["steps"] == 4

    async def test_chat_after_image_evicts_comfyui(self, e2e_client, e2e_env):
        """Chat again → llama evicts ComfyUI (model VRAM tracked)."""
        orch = e2e_env["orch"]
        r = await e2e_client.post("/v1/chat/completions", json={
            "model": "llama-small",
            "messages": [{"role": "user", "content": "back again"}],
        })
        assert r.status_code == 200, r.text
        assert running(orch) == {"llama-small"}, f"Expected only llama, got {running(orch)}"

    async def test_status_endpoint(self, e2e_client):
        """Status endpoint returns fleet info."""
        r = await e2e_client.get("/status")
        assert r.status_code == 200
        assert len(r.json()["services"]) == 3

    async def test_unload_all_endpoint(self, e2e_client, e2e_env):
        """POST /v1/models/unload-all stops all running backends."""
        orch = e2e_env["orch"]
        # First ensure something is running
        await e2e_client.post("/v1/chat/completions", json={
            "model": "llama-small",
            "messages": [{"role": "user", "content": "test"}],
        })
        assert running(orch) == {"llama-small"}

        r = await e2e_client.post("/v1/models/unload-all")
        assert r.status_code == 200
        assert "llama-small" in r.json()["unloaded"]
        assert running(orch) == set()
