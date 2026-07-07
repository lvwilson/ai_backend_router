#!/usr/bin/env python3
"""
iterative_music.py — Gemma iteratively improves a generated music track.

Pipeline:
  1. Generate initial music via the router's /v1/music/generations endpoint.
  2. Send the audio to gemma-4-12b-it for critique and improvement commands.
  3. Parse gemma's structured feedback (3-backtick command block).
  4. Regenerate with improved parameters (up to 3 iterations).
  5. Pick the best track based on gemma's scoring.

Usage:
  python demo/iterative_music.py [BASE_URL] [INITIAL_TAGS]

Defaults:
  BASE_URL     = http://127.0.0.1:8000
  INITIAL_TAGS = "lo-fi, chill, ambient, piano"

Gemma command format (3 backticks):
```
score: 7
tags: lo-fi, chill, ambient, soft piano, warm
lyrics: (if applicable)
feedback: The track is pleasant but lacks energy in the middle section.
```
"""

import sys
import json
import base64
import time
import subprocess
import tempfile
from pathlib import Path

try:
    import requests
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "requests"],
        stdout=subprocess.DEVNULL,
    )
    import requests

# ── Configuration ────────────────────────────────────────────────────────

BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"
INITIAL_TAGS = sys.argv[3] if len(sys.argv) > 3 else "lo-fi, chill, ambient, piano"
INITIAL_LYRICS = ""
DURATION = 30          # seconds per generation (short for iteration)
BPM = 90
SEED = 0               # 0 = randomize each generation
MAX_ITERATIONS = 3
GEMMA_MODEL = "gemma-4-12b-it"
MUSIC_MODEL = "ace_step_1.5_xl_turbo"
OUTPUT_DIR = Path(__file__).parent / "output"

# ── Colours ──────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def step(label: str):
    print(f"\n{BOLD}{CYAN}▶ {label}{RESET}")


def ok(msg: str):
    print(f"  {GREEN}✓ {msg}{RESET}")


def fail(msg: str):
    print(f"  {RED}✗ {msg}{RESET}")


def info(msg: str):
    print(f"  {DIM}{msg}{RESET}")


# ── Core functions ───────────────────────────────────────────────────────

def generate_music(tags: str, lyrics: str, iteration: int) -> Path | None:
    """Generate a music track via the router. Returns the output file path."""
    payload = {
        "model": MUSIC_MODEL,
        "tags": tags,
        "lyrics": lyrics,
        "duration": DURATION,
        "bpm": BPM,
        "seed": SEED,
    }

    t0 = time.time()
    try:
        r = requests.post(f"{BASE}/v1/music/generations", json=payload, timeout=300)
    except Exception as e:
        fail(f"Generation request failed: {e}")
        return None

    if r.status_code != 200:
        fail(f"HTTP {r.status_code}: {r.text[:200]}")
        return None

    elapsed = time.time() - t0
    data = r.json()

    if not data.get("data"):
        fail("No audio data in response")
        return None

    item = data["data"][0]
    audio_path = item.get("path")
    if not audio_path:
        fail("No path in response")
        return None

    # Copy to our output dir with a clear name
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / f"track_iter{iteration}.mp3"
    import shutil
    shutil.copy2(audio_path, out)
    ok(f"Generated in {elapsed:.1f}s → {out.name}")
    return out


def critique_audio(audio_path: Path, prev_tags: str, prev_lyrics: str, iteration: int) -> dict:
    """
    Send audio to gemma for critique. Returns parsed command block or None.

    Gemma is asked to output a structured block delimited by 3 backticks.
    """
    audio_bytes = audio_path.read_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

    prompt = f"""You are a music critic and producer. Listen to this generated track and evaluate it.

Current generation parameters:
  Tags: {prev_tags}
  Lyrics: {prev_lyrics if prev_lyrics else "(instrumental)"}
  Duration: {DURATION}s, BPM: {BPM}
  Iteration: {iteration}

Please provide your critique and improvement suggestions in the following format.
Wrap your response in exactly 3 backticks (```) as shown:

```
score: <integer 1-10>
tags: <improved comma-separated tags>
lyrics: <improved lyrics or "instrumental">
feedback: <brief explanation of what to improve and why>
```

The score should reflect overall quality. The tags should suggest improvements
to the musical style. The feedback should explain what works and what doesn't.
Be specific and constructive."""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": audio_b64}},
            ],
        }
    ]

    payload = {
        "model": GEMMA_MODEL,
        "messages": messages,
        "max_tokens": 8000,
        "stream": False,
    }

    t0 = time.time()
    try:
        r = requests.post(f"{BASE}/v1/chat/completions", json=payload, timeout=120)
    except Exception as e:
        fail(f"Gemma request failed: {e}")
        return None

    if r.status_code != 200:
        fail(f"Gemma HTTP {r.status_code}: {r.text[:200]}")
        return None

    elapsed = time.time() - t0
    data = r.json()
    msg = data.get("choices", [{}])[0].get("message", {})
    text = msg.get("content", "") or msg.get("reasoning_content", "")

    info(f"Gemma responded in {elapsed:.1f}s ({len(text)} chars)")

    # Parse the 3-backtick command block
    parsed = parse_command_block(text)
    if parsed:
        ok(f"Parsed command: score={parsed.get('score', '?')}/10")
    else:
        info(f"Raw gemma response: {text[:200]}")

    return parsed


def parse_command_block(text: str) -> dict | None:
    """
    Extract a structured command block delimited by triple backticks.

    Expected format:
    ```
    score: 7
    tags: lo-fi, chill
    lyrics: instrumental
    feedback: Needs more energy
    ```
    """
    # Find content between ``` markers
    parts = text.split("```")
    if len(parts) >= 2:
        block = parts[1].strip()
    elif len(parts) >= 1:
        # Fallback: try to parse the whole text if no backticks found
        block = text.strip()
    else:
        return None

    result = {}
    for line in block.split("\n"):
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "score":
            try:
                result["score"] = int(value)
            except ValueError:
                result["score"] = 5  # default
        else:
            result[key] = value

    return result if "score" in result else None


def voice_feedback(feedback: str) -> Path | None:
    """Optionally voice the gemma feedback via TTS."""
    if not feedback:
        return None

    payload = {
        "model": "qwen-talker-1.7b-customvoice",
        "input": feedback,
    }

    try:
        r = requests.post(f"{BASE}/v1/audio/speech", json=payload, timeout=60)
        if r.status_code != 200:
            return None
        OUTPUT_DIR.mkdir(exist_ok=True)
        out = OUTPUT_DIR / "feedback.wav"
        out.write_bytes(r.content)
        return out
    except Exception:
        return None


def play_file(path: Path):
    """Play an audio file using the first available player."""
    players = [
        ["ffplay", "-nodisp", "-autoexit", str(path)],
        ["aplay", str(path)],
        ["paplay", str(path)],
    ]
    for cmd in players:
        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ok(f"Played with {cmd[0]}")
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    info(f"No audio player found. Play: ffplay {path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  Iterative Music Improvement with Gemma{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"  Router:   {BASE}")
    print(f"  Music:    {MUSIC_MODEL}")
    print(f"  Critic:   {GEMMA_MODEL}")
    print(f"  Tags:     {INITIAL_TAGS}")
    print(f"  Duration: {DURATION}s, Max iterations: {MAX_ITERATIONS}")
    print(f"{BOLD}{'=' * 60}{RESET}")

    tags = INITIAL_TAGS
    lyrics = INITIAL_LYRICS
    tracks = []  # (iteration, path, score, feedback)

    for iteration in range(1, MAX_ITERATIONS + 1):
        step(f"Iteration {iteration}/{MAX_ITERATIONS}")

        # Generate
        info(f"Tags: {tags}")
        if lyrics:
            info(f"Lyrics: {lyrics}")

        audio_path = generate_music(tags, lyrics, iteration)
        if not audio_path:
            fail("Generation failed, stopping.")
            break

        # Critique
        critique = critique_audio(audio_path, tags, lyrics, iteration)
        if not critique:
            fail("Critique failed, stopping.")
            break

        score = critique.get("score", 5)
        feedback = critique.get("feedback", "")
        new_tags = critique.get("tags", tags)
        new_lyrics = critique.get("lyrics", lyrics)

        print(f"\n  {BOLD}Score: {score}/10{RESET}")
        if feedback:
            print(f"  {BOLD}Feedback:{RESET} {feedback}")
        if new_tags != tags:
            print(f"  {YELLOW}New tags:{RESET} {new_tags}")

        tracks.append((iteration, audio_path, score, feedback))

        # Voice the feedback
        voice_feedback(feedback)

        # Decide whether to continue iterating
        if score >= 8:
            ok(f"Score {score}/10 — good enough, stopping early.")
            break

        # Update parameters for next iteration
        tags = new_tags
        lyrics = new_lyrics if new_lyrics != "instrumental" else ""
        info(f"Carrying forward improved parameters to iteration {iteration + 1}")

    if not tracks:
        fail("No tracks generated.")
        sys.exit(1)

    # ── Pick the best track ────────────────────────────────────────────
    step("Selecting best track")

    best = max(tracks, key=lambda t: t[2])
    best_iter, best_path, best_score, best_feedback = best

    print(f"\n  {BOLD}Best track: Iteration {best_iter}{RESET}")
    print(f"  {BOLD}Score: {best_score}/10{RESET}")
    print(f"  {BOLD}File: {best_path.name}{RESET}")

    # Copy best to a clear name
    best_final = OUTPUT_DIR / "best_track.mp3"
    import shutil
    shutil.copy2(best_path, best_final)
    ok(f"Saved as: {best_final}")

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  Results Summary{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"  {'Iter':<6} {'Score':<8} {'File':<25} {'Feedback'}")
    print(f"  {'─' * 56}")
    for it, path, sc, fb in tracks:
        marker = " ★" if it == best_iter else ""
        fb_short = (fb[:50] + "..") if len(fb) > 50 else fb
        print(f"  {it:<6} {sc:<8} {path.name:<25}{fb_short}{marker}")

    # ── Play the best ──────────────────────────────────────────────────
    step("Playing best track")
    play_file(best_final)

    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  All outputs in: {OUTPUT_DIR.absolute()}{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}\n")


if __name__ == "__main__":
    main()
