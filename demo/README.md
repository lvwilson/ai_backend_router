Iterative Music Improvement Demo

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

## Usage

```bash
# With defaults (lo-fi chill piano, 30s tracks):
python demo/iterative_music.py

# Custom router URL and tags:
python demo/iterative_music.py http://127.0.0.1:8000 "jazz, upbeat, brass"
```

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
