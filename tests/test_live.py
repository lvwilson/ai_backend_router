#!/usr/bin/env python3
"""
test_live.py — End-to-end smoke test for every router endpoint against a
live running router.

These tests hit a real router (default: http://127.0.0.1:8000) and validate
all 11+ endpoints. They require the router to be running with real backends.

At session start the tests restart the router (via the watchdog-managed
process) to get a clean slate with the latest code. After each test,
POST /v1/models/unload-all is called to unload all loaded models.

Usage:
  pytest tests/test_live.py -v                          # default base URL
  pytest tests/test_live.py -v --base-url=http://localhost:9000

Requires: httpx (sync HTTP client)
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

AUDIO_FILE = Path("~/models/funny.wav").expanduser()


def parse_sse_content(raw_text: str) -> str:
    """Extract concatenated content from SSE stream (data: {...} lines)."""
    content_parts = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                obj = json.loads(data_str)
                delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if delta:
                    content_parts.append(delta)
            except json.JSONDecodeError:
                pass
    return "".join(content_parts)


def _find_router_pid() -> int | None:
    """Find the PID of the running router.py process."""
    result = subprocess.run(
        ["pgrep", "-f", "python.*router\\.py"],
        capture_output=True, text=True,
    )
    pids = result.stdout.strip().split()
    if pids:
        return int(pids[0])
    return None


def _restart_router() -> None:
    """
    Restart the router process so tests get the latest code.

    Uses the POST /v1/models/restart-router endpoint which gracefully
    stops all backends and exits. Falls back to SIGTERM if the endpoint
    isn't available (e.g. old router code). The watchdog restarts the
    router automatically, then we wait for /status to become available.
    """
    base = "http://127.0.0.1:8000"
    client = httpx.Client(base_url=base, timeout=10)

    # Try the graceful restart endpoint first.
    try:
        r = client.post("/v1/models/restart-router")
        if r.status_code == 200:
            stopped = r.json().get("stopped", [])
            print(f"\n[live] Router restarting (stopped: {stopped})...")
            client.close()
        else:
            # Endpoint exists but failed — fall through to SIGTERM.
            print(f"\n[live] restart-router returned {r.status_code}, falling back to SIGTERM...")
            client.close()
            _sigterm_router()
            return
    except httpx.ConnectError:
        # Router not reachable — nothing to restart.
        client.close()
        return
    except Exception:
        client.close()
        _sigterm_router()
        return

    # Wait for watchdog to restart the router (up to 60s).
    client = httpx.Client(base_url=base, timeout=10)
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            r = client.get("/status")
            if r.status_code == 200:
                client.close()
                print(f"[live] Router is up ({r.elapsed:.1f}s)")
                return
        except Exception:
            pass
        time.sleep(2)
    client.close()
    raise RuntimeError("Router did not become healthy within 60s")


def _sigterm_router() -> None:
    """Kill the router via SIGTERM as a fallback."""
    pid = _find_router_pid()
    if pid is not None:
        print(f"\n[live] Restarting router via SIGTERM (PID {pid})...")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        time.sleep(5)


# ── Session-scoped: restart router once before all live tests ──────────────

@pytest.fixture(scope="module", autouse=True)
def restart_router_for_tests():
    """Restart the router once before live tests to ensure clean state."""
    _restart_router()
    yield


@pytest.fixture(scope="module")
def base_url(request):
    """Get the base URL for the live router. Configurable via CLI or env."""
    return getattr(request.config.option, "base_url", None) \
        or os.environ.get("ROUTER_BASE_URL", "http://127.0.0.1:8000")


@pytest.fixture(scope="module")
def live_client(base_url):
    """
    Create a synchronous httpx client for the live router.

    Uses sync httpx because live tests involve long-running backend calls
    (model loading, inference) that are easier to reason about synchronously.
    """
    client = httpx.Client(base_url=base_url, timeout=300)
    yield client
    client.close()


# ── Helper: unload all models between tests ───────────────────────────────

@pytest.fixture(autouse=True)
def unload_after_test(live_client):
    """Unload all models after each live test for clean state."""
    yield
    try:
        live_client.post("/v1/models/unload-all")
    except Exception:
        pass


# ── Live endpoint tests ───────────────────────────────────────────────────

class TestLiveEndpoints:
    """Smoke tests against a live running router."""

    def test_status(self, live_client):
        """GET /status — orchestrator fleet status."""
        r = live_client.get("/status")
        assert r.status_code == 200, f"status={r.status_code}"
        data = r.json()
        assert "services" in data, f"keys={list(data.keys())}"
        assert "total_vram_gb" in data, f"total={data.get('total_vram_gb')}GB"

    def test_models(self, live_client):
        """GET /v1/models — model list."""
        r = live_client.get("/v1/models")
        assert r.status_code == 200, f"status={r.status_code}"
        data = r.json()
        assert "data" in data and isinstance(data["data"], list)
        assert len(data.get("data", [])) > 0
        owners = set(m.get("owned_by", "?") for m in data.get("data", []))
        assert len(owners) >= 2, f"owners={owners}"

    def test_chat_streamed(self, live_client):
        """POST /v1/chat/completions — LLM passthrough (streamed)."""
        payload = {
            "model": "qwen3.6-27b-instruct",
            "messages": [{"role": "user", "content": "Say hello in three words."}],
            "max_tokens": 20,
            "stream": True,
        }
        t0 = time.time()
        r = live_client.post("/v1/chat/completions", json=payload)
        assert 200 <= r.status_code < 300, f"status={r.status_code}"
        content = parse_sse_content(r.text)
        assert len(content) > 0, f"content='{content[:80]}' ({time.time()-t0:.1f}s)"

    def test_chat_non_streamed(self, live_client):
        """POST /v1/chat/completions — LLM passthrough (non-streamed)."""
        payload = {
            "model": "qwen3.6-27b-instruct",
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "max_tokens": 10,
            "stream": False,
        }
        t0 = time.time()
        r = live_client.post("/v1/chat/completions", json=payload)
        assert 200 <= r.status_code < 300, f"status={r.status_code}"
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        assert len(content) > 0, f"content='{content[:80]}' ({time.time()-t0:.1f}s)"

    def test_messages(self, live_client):
        """POST /v1/messages — Anthropic-style passthrough."""
        payload = {
            "model": "qwen3.6-27b-instruct",
            "messages": [{"role": "user", "content": "Say hi briefly."}],
            "max_tokens": 20,
        }
        r = live_client.post("/v1/messages", json=payload)
        assert 200 <= r.status_code < 300, f"status={r.status_code}"
        try:
            data = r.json()
            content = ""
            if "content" in data and isinstance(data["content"], list):
                for item in data["content"]:
                    if item.get("type") == "text":
                        content += item.get("text", "")
            if not content:
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception:
            content = parse_sse_content(r.text)
        assert len(content) > 0, f"content='{content[:80]}'"

    def test_transcription_multipart(self, live_client):
        """POST /v1/audio/transcriptions — CrispASR STT (multipart)."""
        assert AUDIO_FILE.exists(), f"Audio file not found: {AUDIO_FILE}"
        with open(AUDIO_FILE, "rb") as f:
            files = {"file": ("roundtable_en.wav", f, "audio/wav")}
            data = {"model": "qwen3-asr-1.7b"}
            r = live_client.post("/v1/audio/transcriptions", files=files, data=data)
        assert 200 <= r.status_code < 300, f"status={r.status_code}, body={r.text[:200]}"
        td = r.json()
        txt = td.get("text", "")
        assert len(txt) > 0, f"text='{txt[:100]}'"

    def test_transcription_cpu(self, live_client):
        """POST /v1/audio/transcriptions — CrispASR STT (CPU backend)."""
        assert AUDIO_FILE.exists(), f"Audio file not found: {AUDIO_FILE}"
        with open(AUDIO_FILE, "rb") as f:
            files = {"file": ("roundtable_en.wav", f, "audio/wav")}
            data = {"model": "qwen3-asr-1.7b-cpu"}
            r = live_client.post("/v1/audio/transcriptions", files=files, data=data)
        assert 200 <= r.status_code < 300, f"status={r.status_code}, body={r.text[:200]}"
        td = r.json()
        txt = td.get("text", "")
        assert len(txt) > 0, f"text='{txt[:100]}'"

    def test_tts_gpu(self, live_client):
        """POST /v1/audio/speech — CrispASR TTS (GPU)."""
        payload = {
            "model": "qwen-talker-1.7b-customvoice",
            "input": "Hello, this is a test of the text-to-speech system.",
        }
        r = live_client.post("/v1/audio/speech", json=payload)
        assert 200 <= r.status_code < 300, f"status={r.status_code}"
        assert len(r.content) > 0, f"bytes={len(r.content)}"

    def test_tts_cpu(self, live_client):
        """POST /v1/audio/speech — CrispASR TTS (CPU)."""
        payload = {
            "model": "qwen-talker-1.7b-customvoice-cpu",
            "input": "Testing CPU text-to-speech.",
        }
        r = live_client.post("/v1/audio/speech", json=payload)
        assert 200 <= r.status_code < 300, f"status={r.status_code}"
        assert len(r.content) > 0, f"bytes={len(r.content)}"

    def test_voices(self, live_client):
        """GET /v1/voices — CrispASR voice list."""
        r = live_client.get("/v1/voices")
        assert 200 <= r.status_code < 300, f"status={r.status_code}"
        assert len(r.text) > 0, f"body={r.text[:200]}"

    def test_image_generation(self, live_client):
        """POST /v1/images/generations — ComfyUI image gen."""
        payload = {
            "model": "krea2",
            "prompt": "A serene mountain lake at sunrise, photorealistic",
            "size": "512x512",
            "n": 1,
        }
        t0 = time.time()
        r = live_client.post("/v1/images/generations", json=payload)
        assert 200 <= r.status_code < 300, f"status={r.status_code} ({time.time()-t0:.1f}s)"
        data = r.json()
        assert "data" in data and isinstance(data["data"], list)
        imgs = data.get("data", [])
        assert len(imgs) > 0

    def test_error_unknown_model(self, live_client):
        """Error handling — unknown model returns 400."""
        payload = {"model": "nonexistent-model", "messages": [{"role": "user", "content": "hi"}]}
        r = live_client.post("/v1/chat/completions", json=payload)
        assert r.status_code == 400, f"status={r.status_code}"
        data = r.json()
        assert "error" in data

    def test_error_missing_prompt(self, live_client):
        """Error handling — missing prompt on image gen returns 400."""
        payload = {"model": "krea2"}
        r = live_client.post("/v1/images/generations", json=payload)
        assert r.status_code == 400, f"status={r.status_code}"

    def test_unload_all(self, live_client):
        """POST /v1/models/unload-all stops all running backends."""
        r = live_client.post("/v1/models/unload-all")
        assert r.status_code == 200
        assert "unloaded" in r.json()
