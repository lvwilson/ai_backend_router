#!/usr/bin/env python3
"""
iterative_music.py — Gemma drives an agentic music creation loop.

Pipeline:
  1. Gemma decides the song concept: genre, style, tags, and lyrics.
  2. Generate the music via the router's /v1/music/generations endpoint.
  3. Send the audio back to gemma for critique and improvement commands.
  4. Gemma decides new tags/lyrics/feedback for the next iteration.
  5. Repeat up to MAX_ITERATIONS or until gemma scores 8+.
  6. Pick the best track based on gemma's scoring.

Usage:
  python demo/iterative_music.py [BASE_URL] [CONCEPT]

Defaults:
  BASE_URL     = http://127.0.0.1:8000
  CONCEPT      = "a relaxing evening song"  (gemma expands this)

Gemma command format (3 backticks):
```
score: 7
tags: lo-fi, chill, ambient, soft piano, warm
lyrics: [Verse]...
feedback: The track is pleasant but lacks energy in the middle section.
```
"""

import sys
import json
import base64
import time
import subprocess
import shutil
import os
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

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
CONCEPT = sys.argv[2] if len(sys.argv) > 2 else "a relaxing evening song"
DURATION = 120          # 2-minute songs
BPM = None              # Let gemma decide
SEED = 0               # 0 = randomize each generation
MAX_ITERATIONS = 4
MAX_RETRIES = 2        # retries per generation on 502
GEN_TIMEOUT = 600      # 10 minutes for generation
GEMMA_MODEL = "gemma-4-12b-it"
MUSIC_MODEL = "ace_step_1.5_xl_turbo"
OUTPUT_DIR = Path(__file__).parent / "output"

# Keep tags under this character limit to avoid TextEncode token overflow.
MAX_TAGS_CHARS = 120


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

def trim_tags(tags: str) -> str:
    """
    Ensure tags stay within MAX_TAGS_CHARS to avoid TextEncode overflow.

    Strategy: keep tags in order, dropping the last ones until we fit.
    """
    if len(tags) <= MAX_TAGS_CHARS:
        return tags
    parts = [t.strip() for t in tags.split(",") if t.strip()]
    trimmed = []
    total = 0
    for part in parts:
        cost = len(part) + 2  # tag + ", "
        if total + cost > MAX_TAGS_CHARS and trimmed:
            break
        trimmed.append(part)
        total += cost
    result = ", ".join(trimmed)
    info(f"Trimmed tags from {len(tags)} to {len(result)} chars")
    return result


def generate_music(tags: str, lyrics: str, bpm: int, iteration: int) -> Path | None:
    """Generate a music track via the router. Returns the output file path."""
    tags = trim_tags(tags)

    payload = {
        "model": MUSIC_MODEL,
        "tags": tags,
        "lyrics": lyrics,
        "duration": DURATION,
        "seed": SEED,
    }
    if bpm is not None:
        payload["bpm"] = bpm

    for attempt in range(1, MAX_RETRIES + 1):
        t0 = time.time()
        try:
            r = requests.post(
                f"{BASE}/v1/music/generations",
                json=payload,
                timeout=(15, GEN_TIMEOUT),  # (connect_timeout, read_timeout)
            )
        except requests.exceptions.ReadTimeout:
            elapsed = time.time() - t0
            info(f"Read timeout after {elapsed:.0f}s (attempt {attempt}), checking response...")
            try:
                data = r.json()
                if data.get("data"):
                    item = data["data"][0]
                    audio_path = item.get("path")
                    if audio_path:
                        OUTPUT_DIR.mkdir(exist_ok=True)
                        out = OUTPUT_DIR / f"track_iter{iteration}.mp3"
                        shutil.copy2(audio_path, out)
                        ok(f"Generated in {elapsed:.1f}s → {out.name} (after timeout)")
                        return out
            except Exception:
                pass
            if attempt < MAX_RETRIES:
                fail(f"Timeout with no valid response, retrying...")
                time.sleep(3)
                continue
            return None
        except Exception as e:
            fail(f"Generation request failed (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(3)
                continue
            return None

        if r.status_code == 502 and attempt < MAX_RETRIES:
            elapsed = time.time() - t0
            info(f"502 on attempt {attempt} ({elapsed:.0f}s), retrying...")
            time.sleep(3)
            continue

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

        OUTPUT_DIR.mkdir(exist_ok=True)
        out = OUTPUT_DIR / f"track_iter{iteration}.mp3"
        shutil.copy2(audio_path, out)
        ok(f"Generated in {elapsed:.1f}s → {out.name}")
        return out

    fail(f"Generation failed after {MAX_RETRIES} attempts")
    return None


# ── System prompts for gemma ─────────────────────────────────────────────

CONCEPT_PROMPT = """You are a creative music director. Given a brief concept, you will
design the initial parameters for an AI music generation model.

ABOUT THE MUSIC GENERATION MODEL:
The model is Ace Step 1.5 XL Turbo, a music diffusion model.
It accepts these text inputs:

  • tags — A concise, comma-separated list of 5–8 style descriptors
    (genre, mood, instrumentation, production style). Under 120 chars.
    This is the PRIMARY control of the output.

  • lyrics — The vocal content. For vocal tracks, provide structured
    lyrics with section markers like [Verse], [Chorus], [Bridge], [Outro].
    For a {duration}s song, plan lyrics accordingly — roughly
    {lines} lines of lyrics work well. Leave empty string for instrumental.

  • bpm — The tempo. Choose something appropriate for the genre.

YOUR TASK:
Based on the concept below, design the initial tags, lyrics, and BPM.
Be creative and specific. Wrap your response in exactly 3 backticks (```):

```
score: 0
tags: <5-8 concise comma-separated style descriptors>
lyrics: <structured lyrics with section markers, or empty for instrumental>
bpm: <integer tempo>
feedback: <brief explanation of your creative choices>
```

Set score to 0 (this is the initial concept, not yet evaluated).

CONCEPT: {concept}"""


CRITIQUE_PROMPT = """You are an expert music critic and producer evaluating AI-generated music.

ABOUT THE MUSIC GENERATION MODEL:
The track was generated by Ace Step 1.5 XL Turbo, a music diffusion model.
It accepts two text inputs that fully control the output:

  • tags — A concise, comma-separated list of 5–8 style descriptors.
    This is the PRIMARY control — it determines genre, mood, instrumentation,
    and production style. Be specific but concise. Do NOT accumulate tags
    across iterations; always provide a fresh, focused set. Under 120 chars.

  • lyrics — The vocal content. Leave empty for instrumental tracks.
    For vocal tracks, provide structured lyrics with section markers
    like [Verse], [Chorus], [Bridge], [Outro]. For a {duration}s song,
    plan approximately {lines} lines of lyrics.

  • bpm — The tempo. Adjust as needed.

YOUR TASK:
Listen to the track, evaluate its quality, and suggest improved tags,
lyrics, and BPM for the next generation. Be constructive and specific.
Wrap your response in exactly 3 backticks (```):

```
score: <integer 1-10>
tags: <5-8 concise comma-separated style descriptors, under 120 characters>
lyrics: <improved structured lyrics, or empty string for instrumental>
bpm: <integer tempo>
feedback: <2-3 sentences explaining what to improve and why>
```

IMPORTANT:
- Tags must be 5-8 focused descriptors, under 120 characters total.
- Do NOT accumulate tags — pick a fresh set each time.
- If the track should be instrumental, write empty string for lyrics.
- If vocal, provide structured lyrics appropriate for {duration}s.
- Score 8+ means the track is excellent and no further iterations needed."""


def gemma_request(messages: list, max_tokens: int = 8000) -> str | None:
    """Send a request to gemma and return the response text."""
    payload = {
        "model": GEMMA_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
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
    return text


KNOWN_KEYS = {"score", "tags", "lyrics", "bpm", "feedback"}


def parse_command_block(text: str) -> dict | None:
    """
    Extract a structured command block delimited by triple backticks.

    Supports multi-line values: if a line does NOT start with a known
    key (score/tags/lyrics/bpm/feedback), it is appended to the previous
    key's value. This is essential for multi-line lyrics.

    Expected format:
    score: 7
    tags: lo-fi, chill
    lyrics: [Verse 1]
    Walking through the garden
    bpm: 90
    feedback: Needs more energy
    """
    parts = text.split("```")
    if len(parts) >= 2:
        block = parts[1].strip()
    else:
        block = text.strip()

    lines = block.split("\n")
    result = {}
    current_key = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        # Detect if this line starts a new known key
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip()

            if key in KNOWN_KEYS:
                current_key = key
                if key == "score":
                    try:
                        result["score"] = int(value)
                    except ValueError:
                        result["score"] = 5
                elif key == "bpm":
                    try:
                        result["bpm"] = int(value)
                    except ValueError:
                        result["bpm"] = 90
                else:
                    result[key] = value
                continue

        # Line is a continuation of the previous key's value
        if current_key is not None and current_key in result:
            result[current_key] += "\n" + line

    return result if "score" in result else None


def gemma_concept(concept: str) -> dict | None:
    """Ask gemma to design the initial song concept."""
    lines = max(8, DURATION // 5)  # rough guideline for lyric lines
    prompt = CONCEPT_PROMPT.format(
        concept=concept,
        duration=DURATION,
        lines=lines,
    )

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    text = gemma_request(messages, max_tokens=8000)
    if not text:
        return None

    parsed = parse_command_block(text)
    if parsed:
        ok(f"Concept designed: {parsed.get('tags', 'N/A')}")
        if parsed.get("lyrics"):
            info("Lyrics:")
            for lline in parsed["lyrics"].split("\n"):
                info(f"  {lline}")
        else:
            info("Instrumental (no lyrics)")
        if parsed.get("bpm"):
            info(f"BPM: {parsed['bpm']}")
    else:
        info(f"Raw gemma response: {text[:200]}")

    return parsed


def gemma_critique(
    audio_path: Path,
    prev_tags: str,
    prev_lyrics: str,
    prev_bpm: int,
    iteration: int,
    history: list,
) -> dict | None:
    """
    Send audio to gemma for critique. Returns parsed command block or None.
    Includes iteration history so gemma can see what's been tried.
    """
    audio_bytes = audio_path.read_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

    lines = max(8, DURATION // 5)

    # Build history summary
    history_text = ""
    if history:
        history_text = "\nPrevious iterations:\n"
        for i, h in enumerate(history):
            hist_lyr = h.get("lyrics", "")
            lyr_summary = hist_lyr[:120] + "..." if len(hist_lyr) > 120 else hist_lyr
            history_text += f"  Iter {i+1}: score={h['score']}, tags={h.get('tags', '')}, bpm={h.get('bpm', '?')}, lyrics={lyr_summary}, feedback={h.get('feedback', '')[:100]}\n"

    lyrics_info = prev_lyrics if prev_lyrics else "(instrumental — no lyrics supplied)"

    prompt = f"""{CRITIQUE_PROMPT.format(duration=DURATION, lines=lines)}

{history_text}
Current generation parameters:
  Tags: {prev_tags}
  Lyrics: {lyrics_info}
  BPM: {prev_bpm}
  Duration: {DURATION}s
  Iteration: {iteration}

Please evaluate and suggest improvements."""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": audio_b64}},
            ],
        }
    ]

    text = gemma_request(messages, max_tokens=8000)
    if not text:
        return None

    parsed = parse_command_block(text)
    if parsed:
        ok(f"Parsed critique: score={parsed.get('score', '?')}/10")
    else:
        info(f"Raw gemma response: {text[:200]}")

    return parsed


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


def restart_router() -> bool:
    """Restart the router via the watchdog."""
    try:
        r = requests.post(f"{BASE}/v1/models/restart-router", timeout=10)
        if r.status_code == 200:
            ok("Router restart requested")
            # Wait for the router to come back up
            for i in range(30):
                time.sleep(2)
                try:
                    r2 = requests.get(f"{BASE}/status", timeout=5)
                    if r2.status_code == 200:
                        ok("Router is back online")
                        return True
                except Exception:
                    pass
            fail("Router did not come back online in time")
            return False
        else:
            fail(f"Restart failed: {r.text[:200]}")
            return False
    except Exception as e:
        fail(f"Could not reach router: {e}")
        return False


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Restart the router for clean state
    step("Restarting router")
    if not restart_router():
        fail("Router restart failed, proceeding anyway...")
        time.sleep(3)

    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  Agentic Music Creation with Gemma{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"  Router:   {BASE}")
    print(f"  Music:    {MUSIC_MODEL}")
    print(f"  Critic:   {GEMMA_MODEL}")
    print(f"  Concept:  {CONCEPT}")
    print(f"  Duration: {DURATION}s, Max iterations: {MAX_ITERATIONS}")
    print(f"  Timeout:  {GEN_TIMEOUT}s")
    print(f"{BOLD}{'=' * 60}{RESET}")

    # ── Phase 1: Gemma designs the concept ─────────────────────────────
    step("Phase 1: Gemma designs the song concept")
    info(f"Concept prompt: \"{CONCEPT}\"")

    concept = gemma_concept(CONCEPT)
    if not concept:
        fail("Gemma failed to design a concept.")
        sys.exit(1)

    tags = concept.get("tags", "")
    lyrics = concept.get("lyrics", "")
    bpm = concept.get("bpm", 90)
    if lyrics and lyrics.lower() in ("instrumental", "empty", ""):
        lyrics = ""

    print(f"\n  {BOLD}Initial design:{RESET}")
    print(f"  {BOLD}Tags:{RESET} {tags}")
    if lyrics:
        print(f"  {BOLD}Lyrics:{RESET}")
        for line in lyrics.split("\n"):
            print(f"    {line}")
    else:
        print(f"  {BOLD}Lyrics:{RESET} (instrumental)")
    print(f"  {BOLD}BPM:{RESET} {bpm}")

    # ── Phase 2: Agentic generation loop ────────────────────────────────
    tracks = []  # (iteration, path, score, feedback, tags, lyrics, bpm)
    history = []  # for gemma context

    for iteration in range(1, MAX_ITERATIONS + 1):
        step(f"Iteration {iteration}/{MAX_ITERATIONS}")

        info(f"Tags: {tags}")
        if lyrics:
            nlines = lyrics.count("\n") + 1
            info(f"Lyrics: ({nlines} lines)")
            for lline in lyrics.split("\n"):
                info(f"  {lline}")
        else:
            info("Lyrics: (instrumental)")
        info(f"BPM: {bpm}")

        # Generate
        audio_path = generate_music(tags, lyrics, bpm, iteration)
        if not audio_path:
            fail("Generation failed, stopping.")
            break

        # Critique
        critique = gemma_critique(audio_path, tags, lyrics, bpm, iteration, history)
        if not critique:
            fail("Critique failed, stopping.")
            break

        score = critique.get("score", 5)
        feedback = critique.get("feedback", "")
        new_tags = critique.get("tags", tags)
        new_lyrics = critique.get("lyrics", lyrics)
        new_bpm = critique.get("bpm", bpm)

        print(f"\n  {BOLD}Score: {score}/10{RESET}")
        if feedback:
            print(f"  {BOLD}Feedback:{RESET}")
            for fline in feedback.split("\n"):
                print(f"    {fline}")
        if new_tags != tags:
            print(f"  {YELLOW}New tags:{RESET} {new_tags}")
        if new_lyrics != lyrics:
            print(f"  {YELLOW}New lyrics:{RESET}")
            if new_lyrics:
                for lline in new_lyrics.split("\n"):
                    print(f"    {lline}")
            else:
                print(f"    (instrumental)")
        if new_bpm != bpm:
            print(f"  {YELLOW}New BPM:{RESET} {new_bpm}")

        tracks.append((iteration, audio_path, score, feedback, tags, lyrics, bpm))
        history.append({
            "score": score,
            "tags": tags,
            "lyrics": lyrics if lyrics else "",
            "bpm": bpm,
            "feedback": feedback,
        })

        # Voice the feedback
        voice_feedback(feedback)

        # Update parameters for next iteration (gemma drives everything)
        # Always run all iterations — we pick the best at the end.
        tags = new_tags
        lyrics = new_lyrics if new_lyrics and new_lyrics.lower() not in ("instrumental", "empty", "") else ""
        bpm = new_bpm
        info(f"Carrying forward gemma's improved parameters to iteration {iteration + 1}")

    if not tracks:
        fail("No tracks generated.")
        sys.exit(1)

    # ── Pick the best track ────────────────────────────────────────────
    step("Selecting best track")

    best = max(tracks, key=lambda t: t[2])
    best_iter, best_path, best_score, best_feedback, best_tags, best_lyrics, best_bpm = best

    print(f"\n  {BOLD}Best track: Iteration {best_iter}{RESET}")
    print(f"  {BOLD}Score: {best_score}/10{RESET}")
    print(f"  {BOLD}File: {best_path.name}{RESET}")

    best_final = OUTPUT_DIR / "best_track.mp3"
    shutil.copy2(best_path, best_final)
    ok(f"Saved as: {best_final}")

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  Results Summary{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"  {'Iter':<6} {'Score':<8} {'BPM':<6} {'File':<25} {'Feedback'}")
    print(f"  {'─' * 56}")
    for it, path, sc, fb, tg, ly, bp in tracks:
        marker = " ★" if it == best_iter else ""
        fb_short = (fb[:40] + "..") if len(fb) > 40 else fb
        print(f"  {it:<6} {sc:<8} {bp:<6} {path.name:<25}{fb_short}{marker}")

    # ── Play the best ──────────────────────────────────────────────────
    step("Playing best track")
    play_file(best_final)

    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  All outputs in: {OUTPUT_DIR.absolute()}{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}\n")


if __name__ == "__main__":
    main()
