"""
test_image_response.py — Dry-run tests for the image response builder.

Covers build_openai_response() including the `augmented_prompt` echo added
by the /v1/images/generations/augmented endpoint, without launching ComfyUI
or the GPU.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.comfyui import build_openai_response

SAMPLE_IMAGE = {
    "path": "/out/image/krea2_00001_.png",
    "filename": "krea2_00001_.png",
    "subfolder": "",
    "type": "output",
}


class TestImageResponseShape:
    def test_base_shape(self):
        resp = build_openai_response([SAMPLE_IMAGE], created=123)
        assert resp["created"] == 123
        assert len(resp["data"]) == 1
        item = resp["data"][0]
        assert item["path"] == "/out/image/krea2_00001_.png"
        assert item["comfyui"] == {
            "filename": "krea2_00001_.png",
            "subfolder": "",
            "type": "output",
        }

    def test_no_augmented_prompt_field_when_omitted(self):
        # Raw (non-augmented) route — the field must not appear at all.
        resp = build_openai_response([SAMPLE_IMAGE], created=123)
        assert "augmented_prompt" not in resp

    def test_explicit_none_omits_field_too(self):
        resp = build_openai_response(
            [SAMPLE_IMAGE], created=123, augmented_prompt=None
        )
        assert "augmented_prompt" not in resp

    def test_augmented_prompt_echoed_verbatim(self):
        desc = (
            "A lone cyberpunk samurai in weathered dark armor with glowing "
            "violet circuit inlays, katana sheathed, standing in the rain-slick "
            "neon alley of a night market, mid-shot, low dramatic angle, wet "
            "reflections and volumetric neon light, cinematic moody atmosphere, "
            "highly detailed, sharp focus, masterpiece, 8k"
        )
        resp = build_openai_response(
            [SAMPLE_IMAGE], created=123, augmented_prompt=desc
        )
        assert resp["augmented_prompt"] == desc

    def test_augmented_prompt_does_not_alter_data_items(self):
        plain = build_openai_response([SAMPLE_IMAGE], created=123)
        aug = build_openai_response(
            [SAMPLE_IMAGE], created=123, augmented_prompt="some prompt"
        )
        assert aug["data"] == plain["data"]
        assert aug["created"] == plain["created"]

    def test_multiple_images_still_one_prompt(self):
        resp = build_openai_response(
            [SAMPLE_IMAGE] * 4, created=1, augmented_prompt="x"
        )
        assert len(resp["data"]) == 4
        assert resp["augmented_prompt"] == "x"
