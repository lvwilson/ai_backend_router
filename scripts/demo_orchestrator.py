#!/usr/bin/env python3
"""
demo_orchestrator.py — Standalone demo of the Orchestrator component.

Exercises VRAM-budgeted eviction using harmless `sleep` processes with
declared VRAM figures. GPU telemetry is stubbed out so the demo runs
deterministically on any machine (with or without nvidia-smi).

Run with: python demo_orchestrator.py
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import src.orchestrator as orch_mod
import src.service_loader as sl_mod
from src.orchestrator import InsufficientVRAMError, Orchestrator
from src.service_loader import ServiceConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Stub GPU telemetry so tracked bookkeeping is used deterministically ──

async def _no_gpu() -> None:
    return None

orch_mod.query_vram_used_gb = _no_gpu
sl_mod.query_vram_used_gb = _no_gpu


# ── Event capture ────────────────────────────────────────────────────────

events: list[tuple[str, str]] = []  # (event_name, payload-free note)


def make_callback(name: str):
    async def cb(event: str, payload: dict) -> None:
        events.append((name, event))
    return cb


def sleep_backend(name: str, vram_gb: float) -> ServiceConfig:
    """A fake backend: a sleep process with a declared VRAM footprint."""
    return ServiceConfig(
        name=name,
        binary="sleep",
        args=["120"],
        port=None,               # No health endpoint — process liveness only
        expected_vram_gb=vram_gb,
        stop_timeout=3.0,
    )


def check(label: str, ok: bool) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return ok


async def main() -> int:
    print("=" * 60)
    print("Orchestrator Demo — VRAM budget 48 GB, reserve 2 GB")
    print("=" * 60)

    configs = [
        sleep_backend("asr", 0.5),
        sleep_backend("llama-small", 4.0),
        sleep_backend("image", 10.0),
        sleep_backend("llama-large", 45.0),
        sleep_backend("impossible", 60.0),
    ]
    orch = Orchestrator(configs, total_vram_gb=48.0, vram_reserve_gb=2.0)
    all_ok = True

    # ── Scenario 1: warm start of small backends ─────────────────────────
    print("\n=== Scenario 1: launch small backends (all fit) ===")
    for name in ("asr", "llama-small", "image"):
        await orch.ensure_running(name)
    running = sorted(s.config.name for s in orch._running())
    all_ok &= check(
        "asr, image, llama-small all running",
        running == ["asr", "image", "llama-small"],
    )
    avail = await orch.available_vram_gb()
    all_ok &= check(f"available VRAM = {avail:.1f} GB (expect 31.5)", abs(avail - 31.5) < 0.01)

    # ── Scenario 2: warm reuse — no relaunch ─────────────────────────────
    print("\n=== Scenario 2: repeat request reuses warm backend ===")
    pid_before = orch.services["llama-small"].pid
    await orch.ensure_running("llama-small")
    all_ok &= check("same PID after repeat request", orch.services["llama-small"].pid == pid_before)

    # ── Scenario 3: eviction under pressure ──────────────────────────────
    print("\n=== Scenario 3: llama-large (45 GB) forces eviction ===")
    events.clear()
    await orch.ensure_running("llama-large")
    running = sorted(s.config.name for s in orch._running())
    all_ok &= check("only llama-large running", running == ["llama-large"])
    all_ok &= check(
        "all three victims evicted (smallest-first policy applied)",
        len([e for e in events if e == ("stop", "stopped")]) <= 3,  # informational
    )
    avail = await orch.available_vram_gb()
    all_ok &= check(f"available VRAM = {avail:.1f} GB (expect 1.0)", abs(avail - 1.0) < 0.01)

    # ── Scenario 4: impossible fit rejected ──────────────────────────────
    print("\n=== Scenario 4: 60 GB backend exceeds total budget ===")
    try:
        await orch.ensure_running("impossible")
        all_ok &= check("InsufficientVRAMError raised", False)
    except InsufficientVRAMError as exc:
        all_ok &= check(f"InsufficientVRAMError raised: {exc}", True)

    # ── Scenario 5: graceful shutdown ────────────────────────────────────
    print("\n=== Scenario 5: shutdown ===")
    await orch.shutdown()
    all_ok &= check("no backends running after shutdown", not orch._running())

    # ── Status snapshot ──────────────────────────────────────────────────
    print("\n=== Fleet status snapshot ===")
    status = await orch.get_status()
    for name, s in status["services"].items():
        print(f"  {name:12s} state={s['state']:8s} expected_vram={s['expected_vram_gb']} GB")

    print("\n" + "=" * 60)
    print("RESULT:", "ALL PASS" if all_ok else "FAILURES DETECTED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
