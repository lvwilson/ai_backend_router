#!/usr/bin/env python3
"""
test_audio_to_voice.py — Send audio to gemma-4-12b-it, then voice the response via TTS.

Pipelines:
  1. Read audio file → base64 encode
  2. Send to gemma-4-12b-it via /v1/chat/completions (multimodal)
  3. Extract text response
  4. Send text to TTS via /v1/audio/speech
  5. Save output .wav and optionally play

Usage:
  python test_audio_to_voice.py [AUDIO_FILE] [BASE_URL]

Defaults:
  AUDIO_FILE = ~/models/funny.wav
  BASE_URL   = http://127.0.0.1:8000
"""
import sys
import json
import base64
import mimetypes
import time
import subprocess
from pathlib import Path

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"],
                          stdout=subprocess.DEVNULL)
    import requests

BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"
AUDIO_FILE = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path("~/models/funny.wav").expanduser()
GEMMA_MODEL = "gemma-4-12b-it"
TTS_MODEL = "qwen-talker-1.7b-customvoice"
OUTPUT_WAV = Path("gemma_response.wav")

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def step(label: str):
    print(f"\n{BOLD}{CYAN}▶ {label}{RESET}")


def ok(msg: str):
    print(f"  {GREEN}✓ {msg}{RESET}")


def fail(msg: str):
    print(f"  {RED}✗ {msg}{RESET}")


def main():
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  Audio → Gemma → Voice  Pipeline Test{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")

    # ── Step 1: Load and encode audio ──────────────────────────────────
    step("1. Loading audio file")

    if not AUDIO_FILE.exists():
        fail(f"File not found: {AUDIO_FILE}")
        sys.exit(1)

    size_kb = AUDIO_FILE.stat().st_size / 1024
    ok(f"Found: {AUDIO_FILE.name} ({size_kb:.1f} KB)")

    audio_bytes = AUDIO_FILE.read_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    mime_type, _ = mimetypes.guess_type(AUDIO_FILE)
    mime_type = mime_type or "audio/wav"

    ok(f"Encoded as base64 ({len(audio_b64)} chars), mime={mime_type}")

    # ── Step 2: Send audio to gemma ────────────────────────────────────
    step("2. Sending audio to gemma-4-12b-it")

    # Build multimodal message — llama.cpp gemma accepts audio as image_url
    # with raw base64 (no data: URI prefix)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Listen to this audio and briefly describe what you hear. Respond in 2-3 sentences.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": audio_b64},
                },
            ],
        }
    ]

    payload = {
        "model": GEMMA_MODEL,
        "messages": messages,
        "max_tokens": 8000,
        "stream": False,
    }

    print(f"  Prompt: {messages[0]['content'][0]['text'][:80]}...")

    t0 = time.time()
    try:
        r = requests.post(
            f"{BASE}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        elapsed = time.time() - t0
    except Exception as e:
        fail(f"Request failed: {e}")
        sys.exit(1)

    if r.status_code != 200:
        fail(f"HTTP {r.status_code}: {r.text[:200]}")
        sys.exit(1)

    ok(f"Response received in {elapsed:.1f}s (HTTP {r.status_code})")

    # Extract text — prefer content, fall back to reasoning_content
    try:
        data = r.json()
        msg = data.get("choices", [{}])[0].get("message", {})
        gemma_text = msg.get("content", "") or msg.get("reasoning_content", "")
    except Exception:
        gemma_text = ""

    if not gemma_text:
        fail("Empty response from gemma")
        print(f"  Raw: {r.text[:300]}")
        sys.exit(1)

    ok(f"Gemma said ({len(gemma_text)} chars):")
    print(f"  {GREEN}\"{gemma_text}\"{RESET}")

    # ── Step 3: Send gemma's text to TTS ───────────────────────────────
    step("3. Voicing response via TTS")

    tts_payload = {
        "model": TTS_MODEL,
        "input": gemma_text,
    }

    t0 = time.time()
    try:
        r = requests.post(
            f"{BASE}/v1/audio/speech",
            json=tts_payload,
            timeout=60,
        )
        elapsed = time.time() - t0
    except Exception as e:
        fail(f"TTS request failed: {e}")
        sys.exit(1)

    if r.status_code != 200:
        fail(f"TTS HTTP {r.status_code}: {r.text[:200]}")
        sys.exit(1)

    audio_out = r.content
    out_size_kb = len(audio_out) / 1024
    ct = r.headers.get("content-type", "unknown")

    ok(f"Audio generated in {elapsed:.1f}s ({out_size_kb:.1f} KB, {ct})")

    # ── Step 4: Save output ────────────────────────────────────────────
    step("4. Saving output")

    OUTPUT_WAV.write_bytes(audio_out)
    ok(f"Saved to: {OUTPUT_WAV.absolute()}")

    # ── Step 5: Play (optional) ────────────────────────────────────────
    step("5. Playing audio")

    players = ["ffplay", "aplay", "paplay", "vlc"]
    played = False
    for player in players:
        try:
            subprocess.check_call(
                [player, "-nodisp", "-autoexit", str(OUTPUT_WAV)]
                if player == "ffplay"
                else [player, str(OUTPUT_WAV)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            ok(f"Played with {player}")
            played = True
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    if not played:
        print(f"  {YELLOW}⚠ No audio player found (tried: {', '.join(players)}){RESET}")
        print(f"  Play manually: ffplay {OUTPUT_WAV}")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  Pipeline complete!{RESET}")
    print(f"  Input:  {AUDIO_FILE.name}")
    print(f"  Gemma:  \"{gemma_text[:80]}{'...' if len(gemma_text) > 80 else ''}\"")
    print(f"  Output: {OUTPUT_WAV} ({out_size_kb:.1f} KB)")
    print(f"{BOLD}{'=' * 60}{RESET}\n")


if __name__ == "__main__":
    main()
