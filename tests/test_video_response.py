"""
test_video_response.py — Dry-run tests for the video response builder.

Covers build_video_openai_response() including the `augmented_prompt`
echo added by the /v1/videos/generations/augmented endpoint, without
launching ComfyUI or the GPU.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.comfyui import build_video_openai_response

SAMPLE_VIDEO = {
    "path": "/out/video/MiniMax_H3_00001_.mp4",
    "filename": "MiniMax_H3_00001_.mp4",
    "subfolder": "",
    "type": "output",
}


class TestVideoResponseShape:
    def test_base_shape(self):
        resp = build_video_openai_response([SAMPLE_VIDEO], created=123)
        assert resp["created"] == 123
        assert len(resp["data"]) == 1
        item = resp["data"][0]
        assert item["path"] == "/out/video/MiniMax_H3_00001_.mp4"
        assert item["comfyui"] == {
            "filename": "MiniMax_H3_00001_.mp4",
            "subfolder": "",
            "type": "output",
        }

    def test_no_augmented_prompt_field_when_omitted(self):
        # Raw (non-augmented) route — the field must not appear at all.
        resp = build_video_openai_response([SAMPLE_VIDEO], created=123)
        assert "augmented_prompt" not in resp

    def test_explicit_none_omits_field_too(self):
        resp = build_video_openai_response([SAMPLE_VIDEO], created=123, augmented_prompt=None)
        assert "augmented_prompt" not in resp

    def test_augmented_prompt_echoed_verbatim(self):
        desc = "Picture 1 aligns with the 0.00-second mark.\n\nSubject: a lighthouse at dusk."
        resp = build_video_openai_response([SAMPLE_VIDEO], created=123, augmented_prompt=desc)
        assert resp["augmented_prompt"] == desc

    def test_augmented_prompt_does_not_alter_data_items(self):
        plain = build_video_openai_response([SAMPLE_VIDEO], created=123)
        aug = build_video_openai_response(
            [SAMPLE_VIDEO], created=123, augmented_prompt="some description"
        )
        assert aug["data"] == plain["data"]
        assert aug["created"] == plain["created"]

    def test_multiple_videos_still_one_description(self):
        resp = build_video_openai_response([SAMPLE_VIDEO] * 3, created=1, augmented_prompt="x")
        assert len(resp["data"]) == 3
        assert resp["augmented_prompt"] == "x"
