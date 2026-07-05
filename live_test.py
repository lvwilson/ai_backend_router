#!/usr/bin/env python3
"""
live_test.py — End-to-end smoke test for every router endpoint.

Usage: python live_test.py [BASE_URL]
"""
import sys, json, time, os, re
from pathlib import Path

try:
    import requests
except ImportError:
    os.system("pip install requests")
    import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
AUDIO_FILE = Path("~/roundtable_en.wav").expanduser()

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def header(label: str):
    print(f"\n{BOLD}{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}{RESET}")


tests_passed = 0
tests_failed = 0


def run(label: str, ok: bool, detail: str = ""):
    global tests_passed, tests_failed
    if ok:
        tests_passed += 1
    else:
        tests_failed += 1
    tag = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  {tag} {label}" + (f"  — {detail}" if detail else ""))
    return ok


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


# ──────────────────────────────────────────────────────────────
header("1. GET /status  —  orchestrator fleet status")
try:
    r = requests.get(f"{BASE}/status", timeout=10)
    run("Returns 200", r.status_code == 200, f"status={r.status_code}")
    data = r.json()
    run("Has 'services' key", "services" in data, f"keys={list(data.keys())}")
    run("Has VRAM info", "total_vram_gb" in data, f"total={data.get('total_vram_gb')}GB")
except Exception as e:
    run("Status endpoint reachable", False, str(e))

# ──────────────────────────────────────────────────────────────
header("2. GET /v1/models  —  model list")
try:
    r = requests.get(f"{BASE}/v1/models", timeout=10)
    run("Returns 200", r.status_code == 200, f"status={r.status_code}")
    data = r.json()
    run("Has 'data' array", "data" in data and isinstance(data["data"], list))
    run(f"Lists {len(data.get('data', []))} models", len(data.get("data", [])) > 0)
    owners = set(m.get("owned_by", "?") for m in data.get("data", []))
    run("Has multiple backend types", len(owners) >= 2, f"owners={owners}")
    model_ids = [m["id"] for m in data.get("data", [])]
    print(f"    Models: {model_ids}")
except Exception as e:
    run("Models endpoint reachable", False, str(e))

# ──────────────────────────────────────────────────────────────
header("3. POST /v1/chat/completions  —  LLM passthrough (streamed)")
try:
    payload = {
        "model": "qwen3.6-27b-instruct",
        "messages": [{"role": "user", "content": "Say hello in three words."}],
        "max_tokens": 20,
        "stream": True,
    }
    t0 = time.time()
    with requests.post(f"{BASE}/v1/chat/completions", json=payload,
                       stream=True, timeout=120) as r:
        run("Returns 2xx", 200 <= r.status_code < 300, f"status={r.status_code}")
        raw = r.text
        elapsed = time.time() - t0
        content = parse_sse_content(raw)
        run("Got non-empty content", len(content) > 0,
            f"content='{content[:80]}' ({elapsed:.1f}s)")
except Exception as e:
    run("Chat completions reachable", False, str(e))

# ──────────────────────────────────────────────────────────────
header("4. POST /v1/chat/completions  —  LLM passthrough (non-streamed)")
try:
    payload = {
        "model": "qwen3.6-27b-instruct",
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 10,
        "stream": False,
    }
    t0 = time.time()
    r = requests.post(f"{BASE}/v1/chat/completions", json=payload, timeout=120)
    run("Returns 2xx", 200 <= r.status_code < 300, f"status={r.status_code}")
    if r.status_code == 200:
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        run("Got non-empty content", len(content) > 0,
            f"content='{content[:80]}' ({time.time()-t0:.1f}s)")
    else:
        run("Response body", False, r.text[:200])
except Exception as e:
    run("Chat completions (non-stream) reachable", False, str(e))

# ──────────────────────────────────────────────────────────────
header("5. POST /v1/messages  —  Anthropic-style passthrough")
try:
    payload = {
        "model": "qwen3.6-27b-instruct",
        "messages": [{"role": "user", "content": "Say hi briefly."}],
        "max_tokens": 20,
    }
    r = requests.post(f"{BASE}/v1/messages", json=payload, timeout=120)
    run("Returns 2xx", 200 <= r.status_code < 300, f"status={r.status_code}")
    if r.status_code == 200:
        # Try JSON first, then SSE
        try:
            data = r.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception:
            content = parse_sse_content(r.text)
        run("Got non-empty content", len(content) > 0, f"content='{content[:80]}'")
    else:
        run("Response body", False, r.text[:200])
except Exception as e:
    run("Messages endpoint reachable", False, str(e))

# ──────────────────────────────────────────────────────────────
header("6. POST /v1/audio/transcriptions  —  CrispASR STT (multipart)")
try:
    with open(AUDIO_FILE, "rb") as f:
        files = {"file": ("roundtable_en.wav", f, "audio/wav")}
        data = {"model": "qwen3-asr-1.7b"}
        r = requests.post(f"{BASE}/v1/audio/transcriptions",
                          files=files, data=data, timeout=60)
    run("Returns 2xx", 200 <= r.status_code < 300,
        f"status={r.status_code}, body={r.text[:200]}")
    if r.status_code == 200:
        try:
            td = r.json()
            txt = td.get("text", "")
            run("Got transcription text", len(txt) > 0, f"text='{txt[:100]}'")
        except Exception:
            run("Got response body", len(r.text) > 0, r.text[:200])
    else:
        run("Response body", False, r.text[:200])
except Exception as e:
    run("Transcriptions (multipart) reachable", False, str(e))

# ──────────────────────────────────────────────────────────────
header("7. POST /v1/audio/transcriptions  —  CrispASR STT (CPU backend)")
try:
    with open(AUDIO_FILE, "rb") as f:
        files = {"file": ("roundtable_en.wav", f, "audio/wav")}
        data = {"model": "qwen3-asr-1.7b-cpu"}
        r = requests.post(f"{BASE}/v1/audio/transcriptions",
                          files=files, data=data, timeout=120)
    run("Returns 2xx", 200 <= r.status_code < 300,
        f"status={r.status_code}, body={r.text[:200]}")
    if r.status_code == 200:
        try:
            td = r.json()
            txt = td.get("text", "")
            run("Got transcription text", len(txt) > 0, f"text='{txt[:100]}'")
        except Exception:
            run("Got response body", len(r.text) > 0, r.text[:200])
except Exception as e:
    run("Transcriptions (CPU) reachable", False, str(e))

# ──────────────────────────────────────────────────────────────
header("8. POST /v1/audio/speech  —  CrispASR TTS (GPU)")
try:
    payload = {
        "model": "qwen-talker-1.7b-voicedesign",
        "input": "Hello, this is a test of the text-to-speech system.",
    }
    r = requests.post(f"{BASE}/v1/audio/speech", json=payload, timeout=60)
    run("Returns 2xx", 200 <= r.status_code < 300,
        f"status={r.status_code}")
    if r.status_code == 200:
        run("Got audio data", len(r.content) > 0,
            f"bytes={len(r.content)}, ct={r.headers.get('content-type','?')[:40]}")
    else:
        run("Response", False, r.text[:200])
except Exception as e:
    run("Speech (TTS) reachable", False, str(e))

# ──────────────────────────────────────────────────────────────
header("9. POST /v1/audio/speech  —  CrispASR TTS (CPU)")
try:
    payload = {
        "model": "qwen-talker-1.7b-customvoice-cpu",
        "input": "Testing CPU text-to-speech.",
    }
    r = requests.post(f"{BASE}/v1/audio/speech", json=payload, timeout=120)
    run("Returns 2xx", 200 <= r.status_code < 300,
        f"status={r.status_code}")
    if r.status_code == 200:
        run("Got audio data", len(r.content) > 0, f"bytes={len(r.content)}")
    else:
        run("Response", False, r.text[:200])
except Exception as e:
    run("Speech (TTS CPU) reachable", False, str(e))

# ──────────────────────────────────────────────────────────────
header("10. POST /v1/audio/speech-to-speech  —  CrispASR S2S")
try:
    with open(AUDIO_FILE, "rb") as f:
        files = {"file": ("roundtable_en.wav", f, "audio/wav")}
        data = {"model": "qwen-talker-1.7b-voicedesign"}
        r = requests.post(f"{BASE}/v1/audio/speech-to-speech",
                          files=files, data=data, timeout=60)
    run("Returns 2xx", 200 <= r.status_code < 300,
        f"status={r.status_code}")
    if r.status_code == 200:
        ct = r.headers.get("content-type", "")
        if "text" in ct or "json" in ct:
            run("Got text response", len(r.text) > 0, r.text[:200])
        else:
            run("Got audio data", len(r.content) > 0, f"bytes={len(r.content)}")
    else:
        run("Response", False, r.text[:200])
except Exception as e:
    run("Speech-to-speech reachable", False, str(e))

# ──────────────────────────────────────────────────────────────
header("11. POST /v1/translate  —  CrispASR translation")
try:
    with open(AUDIO_FILE, "rb") as f:
        files = {"file": ("roundtable_en.wav", f, "audio/wav")}
        data = {"model": "qwen3-asr-1.7b", "target_lang": "en"}
        r = requests.post(f"{BASE}/v1/translate",
                          files=files, data=data, timeout=60)
    run("Returns 2xx", 200 <= r.status_code < 300,
        f"status={r.status_code}, body={r.text[:200]}")
except Exception as e:
    run("Translate endpoint reachable", False, str(e))

# ──────────────────────────────────────────────────────────────
header("12. GET /v1/voices  —  CrispASR voice list")
try:
    r = requests.get(f"{BASE}/v1/voices", timeout=30)
    run("Returns 2xx", 200 <= r.status_code < 300, f"status={r.status_code}")
    if r.status_code == 200:
        run("Got voice data", len(r.text) > 0, f"body={r.text[:200]}")
    else:
        run("Response", False, r.text[:200])
except Exception as e:
    run("Voices endpoint reachable", False, str(e))

# ──────────────────────────────────────────────────────────────
header("13. POST /v1/images/generations  —  ComfyUI image gen")
try:
    payload = {
        "model": "krea2",
        "prompt": "A serene mountain lake at sunrise, photorealistic",
        "size": "512x512",
        "n": 1,
    }
    t0 = time.time()
    r = requests.post(f"{BASE}/v1/images/generations", json=payload, timeout=300)
    run("Returns 2xx", 200 <= r.status_code < 300,
        f"status={r.status_code} ({time.time()-t0:.1f}s)")
    if r.status_code == 200:
        data = r.json()
        run("Has 'data' array", "data" in data and isinstance(data["data"], list))
        imgs = data.get("data", [])
        run(f"Got {len(imgs)} image(s)", len(imgs) > 0)
        if imgs:
            first = imgs[0]
            print(f"    Image: {json.dumps(first, indent=2)[:300]}")
    else:
        run("Response body", False, r.text[:300])
except Exception as e:
    run("Image generation reachable", False, str(e))

# ──────────────────────────────────────────────────────────────
header("14. Error handling  —  unknown model returns 400")
try:
    payload = {"model": "nonexistent-model", "messages": [{"role": "user", "content": "hi"}]}
    r = requests.post(f"{BASE}/v1/chat/completions", json=payload, timeout=30)
    run("Unknown model returns 400", r.status_code == 400, f"status={r.status_code}")
    if r.status_code == 400:
        data = r.json()
        run("Error has message", "error" in data, data.get("error", {}).get("message", ""))
except Exception as e:
    run("Error handling works", False, str(e))

# ──────────────────────────────────────────────────────────────
header("15. Error handling  —  missing prompt on image gen")
try:
    payload = {"model": "krea2"}
    r = requests.post(f"{BASE}/v1/images/generations", json=payload, timeout=10)
    run("Missing prompt returns 400", r.status_code == 400, f"status={r.status_code}")
except Exception as e:
    run("Image error handling works", False, str(e))

# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"  {GREEN}{tests_passed} PASSED{RESET}, {RED}{tests_failed} FAILED{RESET}")
print(f"{'=' * 60}\n")
sys.exit(0 if tests_failed == 0 else 1)
