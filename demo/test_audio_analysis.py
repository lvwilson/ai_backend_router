#!/usr/bin/env python3
"""Isolated test: compare Gemma's analysis of two different MP3s, including lyrics."""

import sys
import base64
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8000"
GEMMA_MODEL = "gemma-4-12b-it"

PROMPT = """You are a music analyst. Listen to this audio track carefully and tell me:
1. The genre and style
2. The vocal characteristics (gender, tone, delivery)
3. The instrumentation
4. The mood and themes
5. The LYRICS — transcribe the key lyrics you can hear (verses, chorus, notable lines)

Be concise. Wrap your response in triple backticks."""

def send_audio(mp3_path: Path, label: str):
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"File: {mp3_path.name} ({mp3_path.stat().st_size / 1024:.0f} KB)")

    audio_bytes = mp3_path.read_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    print(f"Base64 length: {len(audio_b64):,} chars")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": audio_b64}},
            ],
        }
    ]

    payload = {
        "model": GEMMA_MODEL,
        "messages": messages,
        "max_tokens": 4096,
        "stream": False,
    }

    t0 = time.time()
    try:
        r = requests.post(
            f"{BASE}/v1/chat/completions",
            json=payload,
            timeout=300,
        )
        elapsed = time.time() - t0
        print(f"Response in {elapsed:.1f}s, status={r.status_code}")

        if r.status_code != 200:
            print(f"ERROR: {r.text[:500]}")
            return

        data = r.json()
        msg = data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "")

        if reasoning:
            print(f"\nReasoning: {reasoning[:400]}...")

        print(f"\nResponse:")
        print(content)

    except Exception as e:
        print(f"Exception: {e}")


def main():
    zappa = Path("untracked/Frank Zappa - Over-Nite Sensation - 01 - Camarillo Brillo.mp3")
    sonlux = Path("untracked/Son Lux - Tomorrows III - 02 - A Different Kind of Love.mp3")

    if not zappa.exists():
        print(f"Zappa file not found: {zappa}")
        sys.exit(1)
    if not sonlux.exists():
        print(f"Son Lux file not found: {sonlux}")
        sys.exit(1)

    send_audio(sonlux, "Son Lux")
    send_audio(zappa, "Frank Zappa")


if __name__ == "__main__":
    main()
