#!/usr/bin/env python3
"""
vram_benchmark.py — Measure peak VRAM usage of each GPU backend under load.

For each GPU backend:
  1. Start it alone (everything else stopped).
  2. Poll VRAM at 0.5s intervals while idle (baseline).
  3. Send a representative request to exercise the backend.
  4. Keep polling until VRAM settles back to baseline.
  5. Report peak VRAM observed during the whole window.

Usage: python vram_benchmark.py [config.yaml]
"""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config
from orchestrator import Orchestrator
from service_loader import query_per_process_vram

# How long to poll after the request completes before considering VRAM settled.
SETTLE_TIMEOUT = 10.0
POLL_INTERVAL = 0.5


async def measure_backend(loader, orch, exercise_fn):
    """Start backend, exercise it, poll VRAM throughout, return peak GB."""
    name = loader.config.name
    declared = loader.config.expected_vram_gb

    # --- Start ---
    print(f"\n{'─' * 60}")
    print(f"  {name}  (declared {declared:.1f} GB)")
    print(f"{'─' * 60}")

    started = await loader.start()
    if not started:
        print(f"  FAIL: could not start")
        return None

    # --- Poll VRAM: collect (timestamp, peak_for_pid) ---
    pid = loader.pid
    readings = []  # list of (t, vram_gb)

    async def poll():
        """Background task: poll per-process VRAM every POLL_INTERVAL."""
        nonlocal readings
        while True:
            per = await query_per_process_vram()
            vram = per.get(pid, 0.0)
            readings.append((time.monotonic(), vram))
            await asyncio.sleep(POLL_INTERVAL)

    poll_task = asyncio.create_task(poll())

    # Wait for baseline to establish (a few polls after start)
    await asyncio.sleep(2)
    baseline_readings = [v for _, v in readings if v > 0]
    baseline = sum(baseline_readings) / len(baseline_readings) if baseline_readings else 0.0
    print(f"  baseline: {baseline:.2f} GB")

    # --- Exercise ---
    print(f"  exercising …", end="", flush=True)
    t0 = time.monotonic()
    result = await exercise_fn(loader)
    elapsed = time.monotonic() - t0
    print(f" done ({elapsed:.1f}s)")
    if result is not None:
        print(f"  result: {result}")

    # --- Settle: keep polling until VRAM returns near baseline ---
    deadline = time.monotonic() + SETTLE_TIMEOUT
    while time.monotonic() < deadline:
        await asyncio.sleep(POLL_INTERVAL)
        recent = [v for t, v in readings if time.monotonic() - t < 3]
        if recent:
            avg = sum(recent) / len(recent)
            if abs(avg - baseline) < 0.3:  # within 0.3 GB of baseline
                break

    poll_task.cancel()
    try:
        await poll_task
    except asyncio.CancelledError:
        pass

    # --- Compute peak ---
    peak = max(v for _, v in readings) if readings else 0.0
    return peak, baseline


async def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)

    orch = Orchestrator(
        config.services,
        total_vram_gb=config.total_vram_gb,
        vram_reserve_gb=config.vram_reserve_gb,
        sysram_reserve_gb=config.sysram_reserve_gb,
        cache_dir=config.cache_dir,
    )

    # Identify GPU backends (expected_vram_gb > 0)
    gpu_names = [
        c.name for c in config.services if c.expected_vram_gb > 0
    ]
    if not gpu_names:
        print("No GPU backends found.")
        return

    print(f"VRAM Benchmark — {len(gpu_names)} GPU backend(s), "
          f"budget {config.total_vram_gb} GB")

    results = {}

    for name in gpu_names:
        loader = orch.services[name]

        # Build an exercise function based on backend type
        exercise_fn = _make_exercise(loader, config)

        async with orch._lock:
            # Stop everything else first
            others = [s for s in orch._running() if s.config.name != name]
            if others:
                print(f"  stopping {len(others)} other backend(s): "
                      f"{', '.join(s.config.name for s in others)}")
                await asyncio.gather(*(s.stop() for s in others))

            peak_baseline = await measure_backend(loader, orch, exercise_fn)

        if peak_baseline:
            peak, baseline = peak_baseline
            results[name] = {"declared": loader.config.expected_vram_gb,
                             "baseline": round(baseline, 2),
                             "peak": round(peak, 2)}
            print(f"  peak:   {peak:.2f} GB")
            print(f"  suggested vram_usage: {max(peak * 1.15, loader.config.expected_vram_gb):.1f} GB")
        else:
            results[name] = {"declared": loader.config.expected_vram_gb,
                             "baseline": None, "peak": None}

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Backend':<40} {'Declared':>8} {'Baseline':>10} {'Peak':>10} {'Suggested':>10}")
    print(f"  {'-' * 40} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10}")
    for name, r in results.items():
        suggested = round(max(r["peak"] * 1.15, r["declared"]), 1) if r["peak"] else "—"
        print(f"  {name:<40} {r['declared']:>8.1f} "
              f"{str(r['baseline']):>10} {str(r['peak']):>10} "
              f"{suggested:>10}")

    # Save results
    out = Path("vram_benchmark_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {out}")

    await orch.shutdown()


def _detect_type(loader):
    """Detect backend type from name/binary patterns."""
    name = loader.config.name.lower()
    binary = Path(loader.config.binary).name.lower()
    if "llama" in binary or "llama" in name:
        return "llama"
    elif "crispasr" in binary or "crispasr" in name:
        return "crispasr"
    elif "comfyui" in binary or "python" in binary or "comfyui" in name:
        return "comfyui"
    elif "main.py" in loader.config.binary:
        return "comfyui"
    return "unknown"


def _make_exercise(loader, config):
    """Return an async exercise function for the given backend."""
    import aiohttp

    ctype = _detect_type(loader)
    port = loader.config.port
    base = f"http://127.0.0.1:{port}"

    if ctype == "llama":
        async def exercise_llama(_):
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=120)
            ) as sess:
                async with sess.post(
                    f"{base}/v1/chat/completions",
                    json={
                        "model": "any",
                        "messages": [
                            {"role": "user",
                             "content": "Count from one to ten, spelling out each number."}
                        ],
                        "max_tokens": 60,
                        "stream": False,
                    },
                ) as resp:
                    data = await resp.json()
                    content = (data.get("choices", [{}])[0]
                               .get("message", {}).get("content", ""))
                    return f"{len(content)} chars"
        return exercise_llama

    elif ctype == "crispasr":
        # Determine if STT or TTS from the backend name
        if "asr" in loader.config.name.lower():
            async def exercise_asr(_):
                audio = _generate_test_wav()
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as sess:
                    from aiohttp import FormData
                    form = FormData()
                    form.add_field("file", audio, filename="test.wav",
                                   content_type="audio/wav")
                    form.add_field("model", "any")
                    async with sess.post(
                        f"{base}/v1/audio/transcriptions", data=form,
                    ) as resp:
                        data = await resp.json()
                        return f"transcript={data.get('text', '')[:80]}"
            return exercise_asr
        else:
            # TTS (talker)
            async def exercise_tts(_):
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as sess:
                    async with sess.post(
                        f"{base}/v1/audio/speech",
                        json={
                            "model": "any",
                            "input": ("The quick brown fox jumps over the lazy dog. "
                                      "Pack my box with five dozen liquor jugs. "
                                      "How vexingly quick daft zebras jump!"),
                        },
                    ) as resp:
                        body = await resp.read()
                        return f"{len(body)} bytes audio"
            return exercise_tts

    elif ctype == "comfyui":
        async def exercise_comfyui(_):
            # Find first image model's workflow
            img_models = getattr(config, "image_models", {})
            if not img_models:
                return "no image models configured"
            first_model = list(img_models.keys())[0]
            model_cfg = img_models[first_model]
            workflow_path = Path(model_cfg.workflow)
            workflow = json.loads(workflow_path.read_text())

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300)
            ) as sess:
                # Submit prompt
                async with sess.post(
                    f"{base}/prompt",
                    json={"prompt": workflow},
                ) as resp:
                    pdata = await resp.json()
                    prompt_id = pdata.get("prompt_id", "?")

                # Poll for completion
                ws_url = f"ws://127.0.0.1:{port}/ws?clientId=benchmark"
                async with sess.ws_connect(ws_url) as ws:
                    deadline = time.monotonic() + 250
                    while time.monotonic() < deadline:
                        msg = await asyncio.wait_for(ws.receive(), timeout=5)
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if data.get("type") == "executing":
                                if not data["data"].get("node"):
                                    return f"completed ({prompt_id})"
            return "timed out"
        return exercise_comfyui

    else:
        async def exercise_none(_):
            return "no exercise defined"
        return exercise_none


def _generate_test_wav(duration_s=3.0, rate=16000):
    """Generate a simple sine-wave WAV in memory."""
    import struct
    import math
    n_samples = int(duration_s * rate)
    samples = b""
    for i in range(n_samples):
        t = i / rate
        val = int(32767 * 0.8 * math.sin(2 * math.pi * 440 * t))
        samples += struct.pack("<h", val)

    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(samples))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + struct.pack("<H", 1)   # PCM
        + struct.pack("<H", 1)   # mono
        + struct.pack("<I", rate)
        + struct.pack("<I", rate * 2)
        + struct.pack("<H", 2)
        + struct.pack("<H", 16)
        + b"data"
        + struct.pack("<I", len(samples))
    )
    return header + samples


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
