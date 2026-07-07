Music Improvement Demo

Uses **gemma-4-12b-it** as a music critic to iteratively improve a generated
audio track up to 3 times, then picks the best one.

## Pipeline

1. **Generate** — Creates a music track via the router's
   `POST /v1/music/generations` endpoint (Ace Step 1.5 XL Turbo).
2. **Critique** — Sends the audio to gemma (multimodal) which listens and
   returns structured feedback in a 3-backtick command block:

   ```
   score: 7
   tags: lo-fi, chill, ambient, soft piano, warm
   lyrics: instrumental
   feedback: The track is pleasant but lacks energy in the middle section.
   ```

3. **Improve** — Parses gemma's commands, updates the tags/lyrics, and
   regenerates. Repeats up to 3 times or until score ≥ 8.
4. **Select** — Picks the highest-scoring track and saves it as `best_track.mp3`.

## How the Music Model Works

Ace Step 1.5 XL Turbo is a music diffusion model controlled by two text inputs:

- **tags** — 5–8 comma-separated style descriptors (genre, mood, instrumentation).
  This is the primary control. Must be concise (<120 chars).
- **lyrics** — Vocal content. Leave empty for instrumental tracks. For short
  generations (≤30s), keep lyrics brief.

Both fields must be supplied for every generation — empty values produce
meaningless output.

## Usage

```bash
# With defaults (lo-fi chill piano, 2-min tracks):
python demo/iterative_music.py

# Custom router URL and concept:
python demo/iterative_music.py http://127.0.0.1:8000 "a relaxing evening song"

# With a reference MP3 — gemma analyzes it for initial tags and concept:
python demo/iterative_music.py http://127.0.0.1:8000 "inspired by this track" path/to/reference.mp3
```

### Reference MP3 Input

When a third argument (MP3 file path) is provided, gemma listens to the
reference track before any generation occurs and produces:

- **Suggested tags** — 5–8 style descriptors extracted from the reference.
- **Concept paragraph** — A description of the track's vibe and character.

These are used to seed the initial song design, giving the agentic loop a
musical starting point rather than a purely textual one.

## Requirements

- Router running on the specified URL with `gemma-4-12b-it` and
  `ace_step_1.5_xl_turbo` backends available.
- `requests` Python package (auto-installed if missing).

## Output

All generated tracks and the best selection are saved to `demo/output/`:

| File | Description |
|------|-------------|
| `track_iter1.mp3` | First generation |
| `track_iter2.mp3` | Second generation (improved tags) |
| `track_iter3.mp3` | Third generation (further improved) |
| `best_track.mp3` | Highest-scoring track |
| `feedback.wav` | Last gemma feedback voiced via TTS |
