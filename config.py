"""
config.py — YAML configuration loader for the smart LLM router.

Parses the router config (see config.example.yaml) into:
  • RouterConfig: host/port/VRAM budget + routing maps
  • One ServiceConfig per backend (command line built per backend type)

Backend types:
  llama    — llama-server, OpenAI + Anthropic passthrough. Routed by `model`
             field matching the backend name.
  crispasr — CrispASR server, OpenAI audio passthrough. One instance serves
             all audio routes.
  comfyui  — ComfyUI, translated via workflow injection. The `model` field
             selects a workflow (per-model VRAM tracked separately since
             ComfyUI loads models on demand, not at startup).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from service_loader import ServiceConfig


def _p(path: Any) -> str:
    """Expand ~ in a path — backends launch without a shell, so we must."""
    return str(Path(str(path)).expanduser())


@dataclass
class ImageModel:
    """A ComfyUI-served image model: workflow template + VRAM budget."""
    name: str
    backend: str            # Name of the comfyui backend that serves it
    workflow: str           # Path to workflow JSON (ComfyUI API format)
    vram_gb: float


@dataclass
class RouterConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    total_vram_gb: float = 48.0
    vram_reserve_gb: float = 0.0
    sysram_reserve_gb: float = 16.0
    cache_dir: str = "~/.cache/llama-router-cache"

    services: list[ServiceConfig] = field(default_factory=list)

    # Routing maps
    llama_backends: list[str] = field(default_factory=list)     # In config order
    audio_backends: list[str] = field(default_factory=list)     # crispasr backend names (in config order)
    image_models: dict[str, ImageModel] = field(default_factory=dict)
    comfyui_output_dirs: dict[str, str] = field(default_factory=dict)  # backend → output dir
    backend_ports: dict[str, int] = field(default_factory=dict)

    def resolve_llama(self, model: str | None) -> str:
        """Map a request's `model` field to a llama backend name (case-insensitive)."""
        if model is not None:
            model_lower = model.lower()
            for backend in self.llama_backends:
                if backend.lower() == model_lower:
                    return backend
        if not self.llama_backends:
            raise KeyError("No llama backends configured")
        if model is not None:
            raise KeyError(f"Unknown model '{model}' — available: {', '.join(self.llama_backends)}")
        return self.llama_backends[0]  # Default when model field is omitted

    def resolve_audio(self, model: str | None) -> str:
        """Map a request's `model` field to a crispasr backend name (case-insensitive)."""
        if model is not None:
            model_lower = model.lower()
            for backend in self.audio_backends:
                if backend.lower() == model_lower:
                    return backend
        if not self.audio_backends:
            raise KeyError("No audio backends configured")
        return self.audio_backends[0]  # Default: first configured

    def resolve_image_model(self, model: str | None) -> ImageModel:
        """Map a request's `model` field to an image model (case-insensitive)."""
        if model is not None:
            model_lower = model.lower()
            for key, img_model in self.image_models.items():
                if key.lower() == model_lower:
                    return img_model
        if not self.image_models:
            raise KeyError("No image models configured")
        return next(iter(self.image_models.values()))  # Default: first configured


def _llama_service(name: str, spec: dict, cache_dir: str | None = None) -> ServiceConfig:
    args = ["-m", _p(spec["model"]), "--port", str(spec["port"])]
    if "context_size" in spec:
        args += ["-c", str(spec["context_size"])]
    if "gpu_layers" in spec:
        args += ["-ngl", str(spec["gpu_layers"])]
    args += [_p(a) if str(a).startswith("~") else str(a) for a in spec.get("extra_args", [])]

    # Slot persistence cache directory (llama.cpp prompt cache).
    # llama-server exits immediately if the directory does not exist, so
    # create it up front. If creation fails for any reason, degrade
    # gracefully: run without the prompt cache rather than blocking startup.
    slot_save_path = _p(cache_dir) if cache_dir else None
    if slot_save_path:
        try:
            os.makedirs(slot_save_path, exist_ok=True)
            args += ["--slot-save-path", slot_save_path]
        except OSError as exc:
            logging.getLogger(__name__).warning(
                "[%s] Cannot create slot cache dir '%s' (%s) — running without prompt cache",
                name, slot_save_path, exc,
            )
            slot_save_path = None

    # Detect multimodal: llama-server returns 501 for slot cache on multimodal models.
    is_multimodal = "--mmproj" in spec.get("extra_args", [])

    return ServiceConfig(
        name=name,
        binary=spec.get("binary", "llama-server"),
        args=args,
        port=spec["port"],
        health_path=spec.get("health_path", "/health"),
        health_timeout=spec.get("health_timeout", 120.0),  # Large models load slowly
        expected_vram_gb=spec.get("vram_usage", 0.0),
        retries=spec.get("retries", 1),
        slot_save_path=slot_save_path,
        is_multimodal=is_multimodal,
    )


def _crispasr_service(name: str, spec: dict) -> ServiceConfig:
    binary = spec.get("binary", "~/CrispASR/build/bin/crispasr")
    if str(binary).startswith("~"):
        binary = _p(binary)
    args = ["--server", "-m", _p(spec["model"]), "--port", str(spec["port"])]
    if "backend" in spec:
        args += ["--backend", str(spec["backend"])]
    if spec.get("cpu"):
        args += ["--no-gpu"]
    if "codec_model" in spec:
        args += ["--codec-model", _p(spec["codec_model"])]
    if "voice_dir" in spec:
        args += ["--voice-dir", _p(spec["voice_dir"])]
    args += [str(a) for a in spec.get("extra_args", [])]
    return ServiceConfig(
        name=name,
        binary=binary,
        args=args,
        port=spec["port"],
        health_path=spec.get("health_path", "/v1/models"),
        health_timeout=spec.get("health_timeout", 60.0),
        expected_vram_gb=spec.get("vram_usage", 0.0),
        expected_ram_gb=spec.get("ram_usage", 0.0),
        retries=spec.get("retries", 1),
    )


def _comfyui_service(name: str, spec: dict) -> ServiceConfig:
    args = [_p(spec.get("main", "main.py")), "--listen", "127.0.0.1", "--port", str(spec["port"])]
    args += [str(a) for a in spec.get("extra_args", [])]
    return ServiceConfig(
        name=name,
        binary=_p(spec.get("venv", "python")),   # venv python interpreter
        args=args,
        working_dir=_p(spec["working_dir"]) if "working_dir" in spec else None,
        port=spec["port"],
        health_path=spec.get("health_path", "/system_stats"),
        health_timeout=spec.get("health_timeout", 120.0),
        # ComfyUI itself is light; models are budgeted per-request.
        expected_vram_gb=spec.get("vram_usage", 1.0),
        retries=spec.get("retries", 1),
    )


_BUILDERS = {
    "llama": _llama_service,
    "crispasr": _crispasr_service,
    "comfyui": _comfyui_service,
}


def load_config(path: str | Path) -> RouterConfig:
    """Load and validate the router YAML config."""
    raw = yaml.safe_load(Path(path).read_text())

    router = raw.get("router", {})
    cfg = RouterConfig(
        host=router.get("host", "0.0.0.0"),
        port=router.get("port", 8000),
        total_vram_gb=router.get("total_vram", 48.0),
        vram_reserve_gb=router.get("vram_reserve", 0.0),
        sysram_reserve_gb=router.get("sysram_reserve", 16.0),
        cache_dir=router.get("cache_dir", "~/.cache/llama-router-cache"),
    )

    for name, spec in raw.get("backends", {}).items():
        btype = spec.get("type")
        if btype not in _BUILDERS:
            raise ValueError(f"Backend '{name}': unknown type '{btype}'")

        if btype == "llama":
            cfg.services.append(_BUILDERS[btype](name, spec, cfg.cache_dir))
        else:
            cfg.services.append(_BUILDERS[btype](name, spec))
        cfg.backend_ports[name] = spec["port"]

        if btype == "llama":
            cfg.llama_backends.append(name)
        elif btype == "crispasr":
            cfg.audio_backends.append(name)
        elif btype == "comfyui":
            if "output_dir" in spec:
                cfg.comfyui_output_dirs[name] = _p(spec["output_dir"])
            for model_name, mspec in spec.get("models", {}).items():
                cfg.image_models[model_name] = ImageModel(
                    name=model_name,
                    backend=name,
                    workflow=_p(mspec["workflow"]),
                    vram_gb=mspec.get("vram_usage", 0.0),
                )

    return cfg
