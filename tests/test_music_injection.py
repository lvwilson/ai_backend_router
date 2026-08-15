"""
test_music_injection.py — Dry-run tests for music workflow parameter injection.

Loads the real workflow templates from workflows/ and verifies that
inject_music_parameters() targets the right nodes for each model family
(Ace Step 1.5 and MiniMax Music 3) without launching ComfyUI or the GPU.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.comfyui import ComfyUIError, inject_music_parameters

WORKFLOWS_DIR = Path(__file__).parent.parent / "workflows"


def _load(name: str) -> dict:
    return json.loads((WORKFLOWS_DIR / name).read_text())


def _find(wf: dict, class_type: str) -> dict:
    matches = [n for n in wf.values() if n.get("class_type") == class_type]
    assert len(matches) == 1, f"expected exactly one {class_type}, got {len(matches)}"
    return matches[0]


class TestMiniMaxMusic3Injection:
    def test_caption_lyrics_duration_injected(self):
        wf = inject_music_parameters(
            _load("audio_minimax_music_3.json"),
            tags="synthwave, neon, driving bass",
            lyrics="[Verse]\nRunning through the night",
            duration=45,
            seed=12345,
        )
        enc = _find(wf, "MiniMaxMusic3TextEncode")
        assert enc["inputs"]["caption"] == "synthwave, neon, driving bass"
        assert enc["inputs"]["lyrics"] == "[Verse]\nRunning through the night"
        assert enc["inputs"]["max_duration"] == 45
        # seed is a link — must stay a link, value lives in SeedNode
        assert isinstance(enc["inputs"]["seed"], list)
        seed_node = _find(wf, "SeedNode")
        assert seed_node["inputs"]["seed"] == 12345
        # KSampler seed is also a link
        sampler = _find(wf, "KSampler")
        assert isinstance(sampler["inputs"]["seed"], list)

    def test_template_defaults_preserved_when_omitted(self):
        wf = inject_music_parameters(
            _load("audio_minimax_music_3.json"),
            tags="ambient",
        )
        enc = _find(wf, "MiniMaxMusic3TextEncode")
        # Template values must survive untouched.
        assert enc["inputs"]["max_duration"] == 60
        assert enc["inputs"]["cfg_scale"] == 1.7
        assert enc["inputs"]["top_k"] == 50
        assert enc["inputs"]["lyrics"] == ""
        sampler = _find(wf, "KSampler")
        assert sampler["inputs"]["steps"] == 30
        assert sampler["inputs"]["cfg"] == 1.7

    def test_overrides_applied(self):
        wf = inject_music_parameters(
            _load("audio_minimax_music_3.json"),
            tags="jazz",
            cfg_scale=2.5,
            top_k=100,
            steps=40,
            cfg=2.0,
        )
        enc = _find(wf, "MiniMaxMusic3TextEncode")
        assert enc["inputs"]["cfg_scale"] == 2.5
        assert enc["inputs"]["top_k"] == 100
        sampler = _find(wf, "KSampler")
        assert sampler["inputs"]["steps"] == 40
        assert sampler["inputs"]["cfg"] == 2.0

    def test_duration_above_max_rejected(self):
        with pytest.raises(ComfyUIError, match="maximum of 360"):
            inject_music_parameters(
                _load("audio_minimax_music_3.json"),
                tags="ambient",
                duration=400,
            )

    def test_duration_at_max_ok(self):
        wf = inject_music_parameters(
            _load("audio_minimax_music_3.json"),
            tags="ambient",
            duration=360,
        )
        enc = _find(wf, "MiniMaxMusic3TextEncode")
        assert enc["inputs"]["max_duration"] == 360

    def test_seed_zero_is_random(self):
        wf = inject_music_parameters(
            _load("audio_minimax_music_3.json"),
            tags="ambient",
            seed=0,
        )
        seed_node = _find(wf, "SeedNode")
        assert seed_node["inputs"]["seed"] > 0

    def test_original_not_mutated(self):
        wf = _load("audio_minimax_music_3.json")
        before = json.dumps(wf, sort_keys=True)
        inject_music_parameters(wf, tags="test", duration=30)
        assert json.dumps(wf, sort_keys=True) == before


class TestAceStepInjectionStillWorks:
    def test_defaults_preserved(self):
        wf = inject_music_parameters(
            _load("audio_ace_step1_5_xl_turbo.json"),
            tags="lo-fi",
        )
        enc = _find(wf, "TextEncodeAceStepAudio1.5")
        assert enc["inputs"]["tags"] == "lo-fi"
        assert enc["inputs"]["bpm"] == 120
        assert enc["inputs"]["duration"] == 144
        assert enc["inputs"]["timesignature"] == "4"
        assert enc["inputs"]["language"] == "en"
        assert enc["inputs"]["keyscale"] == "E minor"
        assert enc["inputs"]["cfg_scale"] == 2.0
        assert enc["inputs"]["temperature"] == 0.85
        assert enc["inputs"]["top_p"] == 0.9
        assert enc["inputs"]["top_k"] == 0
        assert enc["inputs"]["min_p"] == 0.0
        latent = _find(wf, "EmptyAceStep1.5LatentAudio")
        assert latent["inputs"]["seconds"] == 144
        seed_node = _find(wf, "PrimitiveInt")
        assert isinstance(seed_node["inputs"]["value"], int)

    def test_overrides_applied(self):
        wf = inject_music_parameters(
            _load("audio_ace_step1_5_xl_turbo.json"),
            tags="metal",
            lyrics="[Chorus]\nFire",
            duration=90,
            bpm=150,
            keyscale="D minor",
            seed=777,
        )
        enc = _find(wf, "TextEncodeAceStepAudio1.5")
        assert enc["inputs"]["tags"] == "metal"
        assert enc["inputs"]["lyrics"] == "[Chorus]\nFire"
        assert enc["inputs"]["duration"] == 90
        assert enc["inputs"]["bpm"] == 150
        assert enc["inputs"]["keyscale"] == "D minor"
        latent = _find(wf, "EmptyAceStep1.5LatentAudio")
        assert latent["inputs"]["seconds"] == 90
        assert _find(wf, "PrimitiveInt")["inputs"]["value"] == 777
