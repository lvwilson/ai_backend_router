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

Run: python tests/test_e2e_swap.py
"""

import asyncio
import json
import os
import signal
import shutil
import sys
import tempfile
import time
from pathlib import Path

import aiohttp
import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import src.orchestrator as orch_mod
import src.service_loader as sl_mod
from src.config import load_config
from router import create_app

# Stub GPU telemetry → deterministic tracked bookkeeping.
# NOTE: applied/restored inside main() so that merely importing this module
# (e.g. during pytest collection) does not pollute global state for other tests.
async def _no_gpu(*args, **kwargs):
    return None

async def _no_per_process_vram(*args, **kwargs):
    return {}

MOCK = HERE / "mock_backend.py"
WORKFLOW = HERE / "krea2_basic.json"

# Ports used by mock backends in this test — isolated from live backends.
MOCK_PORTS = [18081, 18082, 18083]


def _kill_stale_mocks() -> None:
    """
    Kill any leftover mock_backend processes from a previous test run.

    Mock backends run on ports 18081–18083, completely separate from live
    backends (8080, 8082, 9000–9005, 8188). This ensures stale processes
    from a crashed test don't block the next run.
    """
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
        # Brief pause to let the kernel release ports.
        time.sleep(0.5)


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

PASS = FAIL = 0


def check(label: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail and not ok else ""))
    PASS += ok
    FAIL += not ok


def running(orch) -> set[str]:
    return {s.config.name for s in orch.services.values() if s.is_alive}


async def main() -> int:
    # Stub GPU telemetry for the duration of this test only.
    # Both total VRAM and per-process queries must be stubbed so the test
    # uses deterministic tracked bookkeeping (not live nvidia-smi readings).
    orig_orch_query = orch_mod.query_vram_used_gb
    orig_sl_query = sl_mod.query_vram_used_gb
    orig_orch_per_pid = orch_mod.query_per_process_vram
    orig_sl_per_pid = sl_mod.query_per_process_vram
    orch_mod.query_vram_used_gb = _no_gpu
    sl_mod.query_vram_used_gb = _no_gpu
    orch_mod.query_per_process_vram = _no_per_process_vram
    sl_mod.query_per_process_vram = _no_per_process_vram
    try:
        return await _run()
    finally:
        orch_mod.query_vram_used_gb = orig_orch_query
        sl_mod.query_vram_used_gb = orig_sl_query
        orch_mod.query_per_process_vram = orig_orch_per_pid
        sl_mod.query_per_process_vram = orig_sl_per_pid


async def _run() -> int:
    # Clean up stale mock backends from previous runs (ports 18081–18083).
    _kill_stale_mocks()

    outdir = tempfile.mkdtemp(prefix="router_e2e_")
    cfg_file = Path(outdir) / "config.yaml"
    cfg_file.write_text(CONFIG_TEMPLATE.format(mock=MOCK, outdir=outdir, workflow=WORKFLOW))

    config = load_config(cfg_file)
    app = create_app(config)
    orch = app.state.orch

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://router", timeout=120) as client:

            print("\n=== 1. Chat → llama-small launches ===")
            r = await client.post("/v1/chat/completions", json={
                "model": "llama-small",
                "messages": [{"role": "user", "content": "hello"}],
            })
            check("chat 200", r.status_code == 200, r.text)
            check("mock llama replied", "mock-llama-reply: hello" in r.text)
            check("llama-small running", running(orch) == {"llama-small"}, str(running(orch)))

            print("\n=== 2. Transcription → asr joins (both fit) ===")
            r = await client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"RIFFdata")})
            check("transcription 200", r.status_code == 200, r.text)
            check("mock transcription text", r.json().get("text") == "mock transcription result")
            check("llama + asr running", running(orch) == {"llama-small", "asr"}, str(running(orch)))

            print("\n=== 3. Image → ComfyUI (8 GB) evicts llama + asr ===")
            r = await client.post("/v1/images/generations", json={
                "model": "krea2",
                "prompt": "a red fox in the snow",
                "size": "512x512",
                "steps": 4,
                "seed": 1234,
            })
            check("image 200", r.status_code == 200, r.text)
            check("only comfyui running", running(orch) == {"image"}, str(running(orch)))

            data = r.json()["data"]
            check("one image returned", len(data) == 1)
            path = data[0].get("path", "")
            check("response includes filepath", bool(path), json.dumps(data))
            check("image file exists on disk", Path(path).is_file(), path)

            # Verify translation: real Krea2 workflow with our parameters injected.
            async with aiohttp.ClientSession() as s:
                async with s.get("http://127.0.0.1:18083/last_workflow") as resp:
                    wf = await resp.json()
            check("prompt injected into node 6", wf["6"]["inputs"]["text"] == "a red fox in the snow")
            check("negative prompt preserved", wf["24"]["inputs"]["text"] == "Gridlines")
            check("size injected into node 10",
                  wf["10"]["inputs"]["width"] == 512 and wf["10"]["inputs"]["height"] == 512)
            check("seed injected into KSampler", wf["2"]["inputs"]["seed"] == 1234)
            check("steps injected into KSampler", wf["2"]["inputs"]["steps"] == 4)

            print("\n=== 4. Chat again → llama evicts ComfyUI (model VRAM tracked) ===")
            r = await client.post("/v1/chat/completions", json={
                "model": "llama-small",
                "messages": [{"role": "user", "content": "back again"}],
            })
            check("chat 200 after swap-back", r.status_code == 200, r.text)
            check("comfyui evicted, llama running", running(orch) == {"llama-small"}, str(running(orch)))

            print("\n=== 5. Status endpoint ===")
            r = await client.get("/status")
            check("status 200", r.status_code == 200)
            check("status lists 3 services", len(r.json()["services"]) == 3)

    shutil.rmtree(outdir, ignore_errors=True)
    print(f"\n{'=' * 50}\nRESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
