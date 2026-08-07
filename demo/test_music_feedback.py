#!/usr/bin/env python3
"""End-to-end test: generate a jazzy song, then feed it to Gemma for audio critique."""

import sys, json, base64, time, shutil, requests
from pathlib import Path

BASE = "http://127.0.0.1:8000"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

def ok(msg): print(f"  {GREEN}✓ {msg}{RESET}")
def fail(msg): print(f"  {RED}✗ {msg}{RESET}")
def info(msg): print(f"  {DIM}{msg}{RESET}")

# ── Step 1: Generate a jazzy song ──────────────────────────────────────
print(f"\n{BOLD}{CYAN}▶ Step 1: Generate a jazzy song{RESET}")

payload = {
    "model": "ace_step_1.5_xl_turbo",
    "tags": "jazz, smooth, saxophone, piano, upright bass, brushed drums, relaxed, cool",
    "lyrics": "[Verse 1]\nSmoke curls in the lamplight glow\nA saxophone begins to flow\nWalking slow on a midnight street\nWhere the jazz notes softly meet\n\n[Chorus]\nTake me slow, take me deep\nThis is the sound that makes me sleep\nBlue notes floating on the breeze\nSwaying gently on the keys\n\n[Verse 2]\nBass walks soft across the floor\nJazz is knocking at the door\nTrumpet whispers, drums reply\nUnderneath the starlit sky\n\n[Chorus]\nTake me slow, take me deep\nThis is the sound that makes me sleep\nBlue notes floating on the breeze\nSwaying gently on the keys",
    "duration": 60,
    "bpm": 95,
    "keyscale": "D minor",
    "timesignature": "4",
    "language": "en",
    "seed": 42,
}

t0 = time.time()
r = requests.post(f"{BASE}/v1/music/generations", json=payload, timeout=(15, 600))
elapsed = time.time() - t0

if r.status_code != 200:
    fail(f"HTTP {r.status_code}: {r.text[:300]}")
    sys.exit(1)

data = r.json()
if not data.get("data"):
    fail("No audio data in response")
    sys.exit(1)

audio_path = data["data"][0]["path"]
ok(f"Generated in {elapsed:.1f}s → {Path(audio_path).name}")

# Copy to output dir
out_file = OUTPUT_DIR / "jazzy_test.mp3"
shutil.copy2(audio_path, out_file)
info(f"Copied to: {out_file}")
info(f"Size: {out_file.stat().st_size / 1024:.0f} KB")

# ── Step 2: Feed audio to Gemma for critique ───────────────────────────
print(f"\n{BOLD}{CYAN}▶ Step 2: Feed audio to Gemma for critique{RESET}")

audio_bytes = out_file.read_bytes()
audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
info(f"Base64 length: {len(audio_b64):,} chars")

critique_prompt = """You are an expert music critic. Listen to this AI-generated jazz track and provide:

1. A detailed critique (genre realization, instrumentation quality, vocal performance, mood)
2. A score from 1-10
3. Suggestions for improvement (tags, lyrics, tempo, key)

Wrap your response in triple backticks:
```
score: <1-10>
tags: <improved tags>
feedback: <detailed critique>
```"""

messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": critique_prompt},
        {"type": "image_url", "image_url": {"url": audio_b64}},
    ],
}]

gemma_payload = {
    "model": "gemma-4-12b-it",
    "messages": messages,
    "max_tokens": 2048,
    "stream": False,
}

t0 = time.time()
r = requests.post(f"{BASE}/v1/chat/completions", json=gemma_payload, timeout=300)
elapsed = time.time() - t0

if r.status_code != 200:
    fail(f"Gemma HTTP {r.status_code}: {r.text[:500]}")
    sys.exit(1)

data = r.json()
msg = data.get("choices", [{}])[0].get("message", {})
content = msg.get("content", "")
reasoning = msg.get("reasoning_content", "")

print(f"\n{BOLD}Gemma responded in {elapsed:.1f}s{RESET}")
if reasoning:
    print(f"\n{BOLD}Reasoning:{RESET}")
    print(reasoning)
print(f"\n{BOLD}Critique:{RESET}")
print(content)

# ── Step 3: Voice the feedback ─────────────────────────────────────────
print(f"\n{BOLD}{CYAN}▶ Step 3: Voice the feedback via TTS{RESET}")

tts_payload = {
    "model": "qwen-talker-1.7b-customvoice",
    "input": content[:500],  # Keep it reasonable for TTS
}

t0 = time.time()
r = requests.post(f"{BASE}/v1/audio/speech", json=tts_payload, timeout=60)
elapsed = time.time() - t0

if r.status_code == 200:
    tts_file = OUTPUT_DIR / "feedback_voice.wav"
    tts_file.write_bytes(r.content)
    ok(f"TTS feedback generated in {elapsed:.1f}s → {tts_file.name}")
    info(f"Size: {tts_file.stat().st_size / 1024:.0f} KB")
else:
    fail(f"TTS HTTP {r.status_code}: {r.text[:200]}")

print(f"\n{BOLD}All outputs in: {OUTPUT_DIR.absolute()}{RESET}\n")
