"""
orchestrator.py — Multi-backend coordinator with VRAM-budgeted eviction.

Owns one ServiceLoader per configured backend and implements the core
router workflow:

  1. ensure_running(name): health-check the target; relaunch if dead.
  2. Before launching, check the VRAM budget:
         available = total_vram - reserve - sum(tracked usage of running backends)
  3. If insufficient, evict running backends smallest-first (confirming
     VRAM is actually freed after each) until the target fits.
  4. Launch with configurable retries (ServiceConfig.retries).

Warm-by-default: backends are never stopped except under VRAM pressure
or on shutdown().

If nvidia-smi is unavailable, budgeting falls back to tracked bookkeeping
(sum of declared/measured values) and eviction confirmation is skipped —
per the plan, we launch anyway and let the OS handle OOM.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from service_loader import (
    EventCallback,
    ServiceConfig,
    ServiceLoader,
    ServiceState,
    query_vram_used_gb,
)

logger = logging.getLogger(__name__)

EVICTION_CONFIRM_TIMEOUT = 15.0   # Seconds to wait for VRAM to drop after eviction
EVICTION_CONFIRM_INTERVAL = 0.5   # Poll interval while confirming


class InsufficientVRAMError(Exception):
    """Raised when a backend cannot fit even after evicting everything else."""


class Orchestrator:
    """
    Coordinates multiple ServiceLoaders under a shared VRAM budget.

    Usage:
        orch = Orchestrator(configs, total_vram_gb=48, vram_reserve_gb=2)
        loader = await orch.ensure_running("llama-large")
        # ... forward request to loader's port ...
        await orch.shutdown()
    """

    def __init__(
        self,
        configs: list[ServiceConfig],
        total_vram_gb: float,
        vram_reserve_gb: float = 2.0,
        event_callback: EventCallback | None = None,
    ):
        self.total_vram_gb = total_vram_gb
        self.vram_reserve_gb = vram_reserve_gb
        self.services: dict[str, ServiceLoader] = {
            c.name: ServiceLoader(c, event_callback=event_callback) for c in configs
        }
        self._lock = asyncio.Lock()  # Serializes ensure_running / eviction decisions
        # Extra VRAM attributed beyond the process itself — e.g. the model a
        # warm ComfyUI instance currently holds resident (loaded on demand).
        self._extra_vram: dict[str, float] = {}

    # ── VRAM accounting ──────────────────────────────────────────────────

    def _tracked_vram(self, loader: ServiceLoader) -> float:
        """VRAM attributed to a service: measured if available, else declared,
        plus any on-demand model VRAM (ComfyUI)."""
        if loader.actual_vram_gb is not None and loader.actual_vram_gb > 0:
            base = loader.actual_vram_gb
        else:
            base = loader.config.expected_vram_gb
        return base + self._extra_vram.get(loader.config.name, 0.0)

    def note_extra_vram(self, name: str, gb: float) -> None:
        """
        Record on-demand model VRAM held by a running backend (ComfyUI keeps
        the most recently used model resident; we track that one).
        """
        self._extra_vram[name] = gb

    def _running(self) -> list[ServiceLoader]:
        return [s for s in self.services.values() if s.is_alive]

    async def available_vram_gb(self) -> float:
        """
        VRAM available for a new backend.

        Prefers a real nvidia-smi reading; falls back to tracked bookkeeping.
        """
        used = await query_vram_used_gb()
        if used is None:
            used = sum(self._tracked_vram(s) for s in self._running())
        return self.total_vram_gb - self.vram_reserve_gb - used

    # ── Core workflow ────────────────────────────────────────────────────

    async def ensure_running(self, name: str, extra_vram_gb: float = 0.0) -> ServiceLoader:
        """
        Guarantee the named backend is running and healthy, evicting others
        if VRAM pressure demands it. Returns its ServiceLoader.

        Args:
            extra_vram_gb: On-demand model VRAM required beyond the process
                itself (ComfyUI per-model budgets). If the backend is already
                warm but the requested model needs more than what's currently
                attributed, room is made for the increase. The caller should
                call note_extra_vram(name, gb) once the model is loaded.

        Raises:
            KeyError: unknown backend name.
            InsufficientVRAMError: backend cannot fit even after evictions.
            RuntimeError: launch failed after retries.
        """
        loader = self.services[name]

        async with self._lock:
            # Fast path: alive and healthy.
            if loader.is_alive:
                if await loader.is_healthy():
                    increase = extra_vram_gb - self._extra_vram.get(name, 0.0)
                    if increase > 0:
                        await self._make_room(increase, exclude=name)
                    return loader
                logger.warning("[%s] Health check failed — killing and relaunching", name)
                await loader.kill()

            # Make room if needed.
            self._extra_vram.pop(name, None)  # Dead process holds no model
            needed = loader.config.expected_vram_gb + extra_vram_gb
            if needed > 0:
                await self._make_room(needed, exclude=name)

            # Launch with retries.
            attempts = 1 + max(0, loader.config.retries)
            for attempt in range(1, attempts + 1):
                if await loader.start():
                    return loader
                logger.warning("[%s] Launch attempt %d/%d failed", name, attempt, attempts)

            raise RuntimeError(f"Backend '{name}' failed to start after {attempts} attempt(s)")

    async def _make_room(self, needed_gb: float, exclude: str) -> None:
        """
        Evict running backends smallest-first until needed_gb fits the budget.

        Raises InsufficientVRAMError if it cannot fit even with all evicted.
        """
        available = await self.available_vram_gb()
        if available >= needed_gb:
            return

        # Sanity: can it ever fit?
        if needed_gb > self.total_vram_gb - self.vram_reserve_gb:
            raise InsufficientVRAMError(
                f"Backend needs {needed_gb:.1f} GB but budget is only "
                f"{self.total_vram_gb - self.vram_reserve_gb:.1f} GB"
            )

        victims = sorted(
            (s for s in self._running() if s.config.name != exclude),
            key=self._tracked_vram,
        )

        for victim in victims:
            freed = self._tracked_vram(victim)
            logger.info(
                "VRAM pressure: evicting '%s' (~%.1f GB) — need %.1f GB, have %.1f GB",
                victim.config.name, freed, needed_gb, available,
            )
            await victim.stop()
            self._extra_vram.pop(victim.config.name, None)
            await self._confirm_vram_freed(available + freed)

            available = await self.available_vram_gb()
            if available >= needed_gb:
                return

        if available < needed_gb:
            raise InsufficientVRAMError(
                f"Only {available:.1f} GB available after evicting all backends; "
                f"need {needed_gb:.1f} GB"
            )

    async def _confirm_vram_freed(self, expected_available: float) -> None:
        """
        Poll nvidia-smi until available VRAM reaches roughly the expected level.

        Best-effort: logs a warning on timeout, skips silently without nvidia-smi.
        """
        if await query_vram_used_gb() is None:
            return  # No GPU telemetry — tracked bookkeeping already updated.

        deadline = time.monotonic() + EVICTION_CONFIRM_TIMEOUT
        while time.monotonic() < deadline:
            available = await self.available_vram_gb()
            if available >= expected_available - 1.0:  # 1 GB tolerance
                return
            await asyncio.sleep(EVICTION_CONFIRM_INTERVAL)
        logger.warning(
            "Eviction VRAM not fully freed after %.0fs (expected ~%.1f GB available)",
            EVICTION_CONFIRM_TIMEOUT, expected_available,
        )

    # ── Status & shutdown ────────────────────────────────────────────────

    async def get_status(self) -> dict[str, Any]:
        """Snapshot of the whole fleet, suitable for a /status endpoint."""
        return {
            "total_vram_gb": self.total_vram_gb,
            "vram_reserve_gb": self.vram_reserve_gb,
            "available_vram_gb": round(await self.available_vram_gb(), 2),
            "services": {
                name: await s.get_status() for name, s in self.services.items()
            },
        }

    async def shutdown(self) -> None:
        """Gracefully stop all running backends (router shutdown hook)."""
        running = self._running()
        if running:
            logger.info("Shutting down %d running backend(s)", len(running))
            await asyncio.gather(*(s.stop() for s in running))
