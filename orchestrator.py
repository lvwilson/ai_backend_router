"""
orchestrator.py — Multi-backend coordinator with VRAM-budgeted eviction.

Owns one ServiceLoader per configured backend and implements the core
router workflow:

  1. ensure_running(name): health-check the target; relaunch if dead.
  2. Before launching, check the VRAM budget using per-process accounting:
         available = total_vram - reserve - sum(per-process VRAM of running backends)
  3. If insufficient, evict running backends smallest-first (confirming
     each process actually freed VRAM) until the target fits.
  4. Launch with configurable retries (ServiceConfig.retries).

Warm-by-default: backends are never stopped except under VRAM pressure
or on shutdown().

VRAM accounting uses nvidia-smi --query-compute-apps for authoritative
per-process readings. If nvidia-smi is unavailable, falls back to
tracked bookkeeping (sum of declared/measured values).
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from pathlib import Path
from typing import Any

from service_loader import (
    EventCallback,
    ServiceConfig,
    ServiceLoader,
    ServiceState,
    query_vram_used_gb,
    query_vram_total_gb,
    query_per_process_vram,
)


async def query_sysram_used_gb() -> float | None:
    """
    Query total system RAM used via /proc/meminfo.

    Returns RAM used in GB, or None if unavailable.
    """
    try:
        text = await asyncio.to_thread(lambda: Path("/proc/meminfo").read_text())
        lines = {k: v for k, v in (
            (line.split(":")[0].strip(), int(line.split(":")[1].strip().split()[0]))
            for line in text.splitlines() if ":" in line
        )}
        total_kb = lines.get("MemTotal", 0)
        available_kb = lines.get("MemAvailable", lines.get("MemFree", 0))
        used_kb = total_kb - available_kb
        return used_kb / (1024.0 * 1024.0)
    except Exception:
        return None

logger = logging.getLogger(__name__)

EVICTION_CONFIRM_TIMEOUT = 15.0   # Seconds to wait for VRAM to drop after eviction
EVICTION_CONFIRM_INTERVAL = 0.5   # Poll interval while confirming


class InsufficientVRAMError(Exception):
    """Raised when a backend cannot fit even after evicting everything else."""


class Orchestrator:
    """
    Coordinates multiple ServiceLoaders under shared VRAM and sysram budgets.

    GPU backends are budgeted against VRAM using per-process accounting
    (nvidia-smi --query-compute-apps); CPU backends are budgeted against
    system RAM (/proc/meminfo).  Each is tracked independently.

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
        sysram_reserve_gb: float = 2.0,
        cache_dir: str | None = None,
        event_callback: EventCallback | None = None,
    ):
        self.total_vram_gb = total_vram_gb
        self.vram_reserve_gb = vram_reserve_gb
        self.sysram_reserve_gb = sysram_reserve_gb
        self.cache_dir = cache_dir
        # Ensure cache directory exists
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self.services: dict[str, ServiceLoader] = {
            c.name: ServiceLoader(c, event_callback=event_callback) for c in configs
        }
        self._lock = asyncio.Lock()  # Serializes ensure_running / eviction decisions
        # Extra VRAM attributed beyond the process itself — e.g. the model a
        # warm ComfyUI instance currently holds resident (loaded on demand).
        self._extra_vram: dict[str, float] = {}

    # ── VRAM accounting ──────────────────────────────────────────────────

    def _tracked_vram(self, loader: ServiceLoader) -> float:
        """VRAM attributed to a service: per-process measurement if available,
        else declared value, plus any on-demand model VRAM (ComfyUI)."""
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

    # ── Sysram accounting (CPU backends) ─────────────────────────────────

    def _tracked_ram(self, loader: ServiceLoader) -> float:
        """RAM attributed to a CPU service."""
        return loader.config.expected_ram_gb

    async def available_sysram_gb(self) -> float:
        """System RAM available for CPU backends."""
        used = await query_sysram_used_gb()
        if used is None:
            used = sum(self._tracked_ram(s) for s in self._running())
        # Total RAM is inferred from /proc/meminfo; reserve is subtracted.
        total = await query_sysram_used_gb()
        if total is not None:
            try:
                text = await asyncio.to_thread(lambda: Path("/proc/meminfo").read_text())
                for line in text.splitlines():
                    if line.startswith("MemTotal"):
                        total_gb = int(line.split(":")[1].strip().split()[0]) / (1024.0 * 1024.0)
                        break
                else:
                    total_gb = None
            except Exception:
                total_gb = None
            if total_gb is not None:
                return total_gb - self.sysram_reserve_gb - used
        # Fallback: tracked bookkeeping with a reasonable total estimate
        return 64.0 - self.sysram_reserve_gb - used

    async def available_vram_gb(self) -> float:
        """
        VRAM available for a new backend.

        Uses per-process accounting: total_vram - reserve - sum(per-process VRAM
        of our managed backends). This is authoritative because it reads exactly
        what each managed process holds, ignoring unmanaged GPU consumers
        (Xorg, Steam, etc.).

        Falls back to nvidia-smi total reading if per-process data is unavailable.
        """
        # Try per-process accounting first.
        per_pid = await query_per_process_vram()
        if per_pid:
            # Sum VRAM of our managed backends by PID.
            our_vram = 0.0
            for loader in self._running():
                pid = loader.pid
                if pid is not None:
                    our_vram += per_pid.get(pid, 0.0)
                else:
                    # Fallback for backends without PID (shouldn't happen for alive processes)
                    our_vram += self._tracked_vram(loader)
            # Add extra VRAM (ComfyUI model VRAM).
            our_vram += sum(self._extra_vram.values())
            return self.total_vram_gb - self.vram_reserve_gb - our_vram

        # Fallback: total nvidia-smi reading minus reserve.
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
        is_cpu = loader.config.expected_ram_gb > 0 and loader.config.expected_vram_gb == 0

        async with self._lock:
            # Fast path: alive and healthy.
            if loader.is_alive:
                if await loader.is_healthy():
                    increase = extra_vram_gb - self._extra_vram.get(name, 0.0)
                    if increase > 0:
                        await self._make_room(increase, exclude=name, cpu=is_cpu)
                    return loader
                logger.warning("[%s] Health check failed — killing and relaunching", name)
                await loader.save_slot_cache()
                await loader.kill()

            # Make room if needed.
            self._extra_vram.pop(name, None)  # Dead process holds no model
            if is_cpu:
                needed = loader.config.expected_ram_gb
            else:
                needed = loader.config.expected_vram_gb + extra_vram_gb
            if needed > 0:
                await self._make_room(needed, exclude=name, cpu=is_cpu)

            # Launch with retries.
            attempts = 1 + max(0, loader.config.retries)
            started = False
            for attempt in range(1, attempts + 1):
                if await loader.start():
                    started = True
                    break
                logger.warning("[%s] Launch attempt %d/%d failed", name, attempt, attempts)

            if not started:
                raise RuntimeError(f"Backend '{name}' failed to start after {attempts} attempt(s)")

            # Restore slot cache after successful launch (llama.cpp only).
            await loader.restore_slot_cache()
            return loader

    async def _make_room(self, needed_gb: float, exclude: str, cpu: bool = False) -> None:
        """
        Evict running backends smallest-first until needed_gb fits the budget.

        Args:
            needed_gb: GB needed (VRAM for GPU backends, RAM for CPU backends).
            exclude: Backend name to skip (the one about to be launched).
            cpu: If True, use sysram budget; otherwise VRAM budget.

        Raises InsufficientVRAMError if it cannot fit even with all evicted.
        """
        if cpu:
            available = await self.available_sysram_gb()
            tracker = self._tracked_ram
            label = "sysram"
        else:
            available = await self.available_vram_gb()
            tracker = self._tracked_vram
            label = "VRAM"

        if available >= needed_gb:
            return

        # Sanity: can it ever fit?
        if cpu:
            try:
                text = await asyncio.to_thread(lambda: Path("/proc/meminfo").read_text())
                for line in text.splitlines():
                    if line.startswith("MemTotal"):
                        budget = int(line.split(":")[1].strip().split()[0]) / (1024.0 * 1024.0) - self.sysram_reserve_gb
                        break
                else:
                    budget = None
            except Exception:
                budget = None
            if budget is not None and needed_gb > budget:
                raise InsufficientVRAMError(
                    f"Backend needs {needed_gb:.1f} GB sysram but budget is only {budget:.1f} GB"
                )
        elif needed_gb > self.total_vram_gb - self.vram_reserve_gb:
            raise InsufficientVRAMError(
                f"Backend needs {needed_gb:.1f} GB but budget is only "
                f"{self.total_vram_gb - self.vram_reserve_gb:.1f} GB"
            )

        victims = sorted(
            (s for s in self._running() if s.config.name != exclude),
            key=tracker,
        )

        for victim in victims:
            freed = tracker(victim)
            logger.info(
                "%s pressure: evicting '%s' (~%.1f GB) — need %.1f GB, have %.1f GB",
                label, victim.config.name, freed, needed_gb, available,
            )
            await victim.save_slot_cache()
            await victim.stop()
            self._extra_vram.pop(victim.config.name, None)

            if not cpu:
                await self._confirm_vram_freed(victim.pid, freed)
                available = await self.available_vram_gb()
            else:
                available = await self.available_sysram_gb()

            if available >= needed_gb:
                return

        if available < needed_gb:
            raise InsufficientVRAMError(
                f"Only {available:.1f} GB {label} available after evicting all backends; "
                f"need {needed_gb:.1f} GB"
            )

    async def _confirm_vram_freed(self, pid: int | None, expected_freed_gb: float) -> None:
        """
        Confirm the evicted process has actually released its VRAM.

        Polls nvidia-smi --query-compute-apps to verify the PID is gone.
        If the PID persists beyond the timeout, it is force-killed.

        Args:
            pid: The PID of the evicted process (may be None if it exited before we checked).
            expected_freed_gb: Approximate VRAM that should have been freed.
        """
        if pid is None:
            return  # Process already exited.

        per_pid = await query_per_process_vram()
        if not per_pid:
            return  # No GPU telemetry — trust the stop.

        # Check if the PID is still consuming VRAM.
        if pid not in per_pid:
            return  # Already freed.

        logger.warning(
            "Evicted process PID %d still holding %.1f GB VRAM — waiting for release",
            pid, per_pid[pid],
        )

        deadline = time.monotonic() + EVICTION_CONFIRM_TIMEOUT
        while time.monotonic() < deadline:
            await asyncio.sleep(EVICTION_CONFIRM_INTERVAL)
            check = await query_per_process_vram()
            if not check or pid not in check:
                logger.info("Evicted process PID %d released VRAM", pid)
                return

        # PID still present after timeout — force kill it.
        logger.error(
            "Evicted process PID %d still holding VRAM after %.0fs — force-killing",
            pid, EVICTION_CONFIRM_TIMEOUT,
        )
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
            # Wait briefly for the kill to take effect.
            await asyncio.sleep(1.0)
            final = await query_per_process_vram()
            if final and pid in final:
                logger.error(
                    "Process PID %d still alive after SIGKILL — VRAM may be leaked",
                    pid,
                )
            else:
                logger.info("Force-killed PID %d, VRAM freed", pid)
        except ProcessLookupError:
            logger.info("PID %d no longer exists, VRAM likely freed", pid)
        except Exception as exc:
            logger.error("Failed to force-kill PID %d: %s", pid, exc)

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
            # Save slot caches before stopping llama backends
            await asyncio.gather(*(s.save_slot_cache() for s in running))
            await asyncio.gather(*(s.stop() for s in running))
