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

Concurrency model (concurrency 1 per backend):
  Each backend has a concurrency-1 "request slot". A request acquires the
  slot for the backend it targets for its full duration (see request_slot),
  which marks the backend "busy". Two consequences:

    • A second request for the same backend *waits its turn* (the slot is a
      semaphore) rather than running concurrently.
    • _make_room never evicts a busy backend — a new request that would
      otherwise need to kill an in-flight one instead *waits for the
      in-flight request to finish* (ensure_running → _BusyBlock) and then
      proceeds.

  This is what makes "a new request" queue behind "a previous one" instead
  of killing it. The wait for a busy backend to drain happens *outside* the
  orchestrator lock, so unrelated backends are not blocked while we wait.

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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from .service_loader import (
    EventCallback,
    ServiceConfig,
    ServiceLoader,
    ServiceState,
    query_vram_used_gb,
    query_vram_total_gb,
    query_per_process_vram,
    query_per_process_vram_with_names,
)


async def query_sysram_gb() -> tuple[float, float] | None:
    """
    Query system RAM via /proc/meminfo in a single read.

    Returns (total_gb, used_gb), or None if unavailable.
    """
    try:
        text = await asyncio.to_thread(lambda: Path("/proc/meminfo").read_text())
        lines = {k: v for k, v in (
            (line.split(":")[0].strip(), int(line.split(":")[1].strip().split()[0]))
            for line in text.splitlines() if ":" in line
        )}
        total_kb = lines.get("MemTotal", 0)
        available_kb = lines.get("MemAvailable", lines.get("MemFree", 0))
        if total_kb <= 0:
            return None
        used_kb = total_kb - available_kb
        gib = 1024.0 * 1024.0
        return total_kb / gib, used_kb / gib
    except Exception:
        return None

logger = logging.getLogger(__name__)

EVICTION_CONFIRM_TIMEOUT = 15.0   # Seconds to wait for VRAM to drop after eviction
EVICTION_CONFIRM_INTERVAL = 0.5   # Poll interval while confirming

# External GPU processes that are considered "default" system consumers
# and should not be killed as VRAM hogs. Matched against the process name
# (from nvidia-smi --query-compute-apps=name) using substring matching.
DEFAULT_GPU_APPS = (
    "Xorg",
    "gnome-shell",
    "kwin",
    "steamwebhelper",
    "steam",
    "mutter",
    "kwin_x11",
    "kwin_wayland",
    "pipewire",
    "wireplumber",
    "firefox",
    "chrome",
    "chromium",
)

# Minimum VRAM usage (GB) for an external process to be considered a "hog".
VRAM_HOG_THRESHOLD_GB = 0.1  # ~100 MiB

# If unmanaged VRAM usage exceeds this on startup, log a warning.
VRAM_UNMANAGED_WARN_GB = 2.0


class InsufficientVRAMError(Exception):
    """Raised when a backend cannot fit even after evicting everything else."""


class _BusyBlock(Exception):
    """
    Internal signal: eviction is blocked because the only backends that could
    be evicted are busy (have in-flight requests). Carries their names so the
    caller can wait for one to drain (outside the lock) and retry.
    """
    def __init__(self, busy_names):
        self.busy_names = list(busy_names)
        super().__init__(f"eviction blocked by busy backends: {self.busy_names}")


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
        # Expand ~ up front — backends launch without a shell, so an
        # unexpanded '~' would create a literal '~' directory in the CWD.
        self.cache_dir = str(Path(cache_dir).expanduser()) if cache_dir else None
        # Ensure cache directory exists
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        self.services: dict[str, ServiceLoader] = {
            c.name: ServiceLoader(c, event_callback=event_callback) for c in configs
        }
        self._lock = asyncio.Lock()  # Serializes ensure_running / eviction decisions
        # ── Request slots (concurrency 1 per backend) ────────────────────
        # A backend with an in-flight request is "busy". A new request for a
        # busy backend waits its turn (the per-backend semaphore), and
        # _make_room never evicts a busy backend — so a new request queues
        # behind a running one instead of killing it. See request_slot and
        # _make_room_or_block.
        self._backend_slots: dict[str, asyncio.Semaphore] = {}
        self._busy: set[str] = set()
        self._busy_cond = asyncio.Condition()
        # Extra VRAM attributed beyond the process itself — e.g. the model a
        # warm ComfyUI instance currently holds resident (loaded on demand).
        self._extra_vram: dict[str, float] = {}
        # VRAM warning state
        self._vram_warned_at: float | None = None
        self._vram_monitor_task: asyncio.Task | None = None

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

    # ── Request slots (concurrency 1 per backend) ────────────────────────

    def _slot_for(self, name: str) -> asyncio.Semaphore:
        """Per-backend concurrency-1 gate (created lazily)."""
        slot = self._backend_slots.get(name)
        if slot is None:
            slot = asyncio.Semaphore(1)
            self._backend_slots[name] = slot
        return slot

    async def acquire_slot(self, name: str) -> None:
        """
        Acquire the concurrency-1 slot for a backend and mark it busy.

        Waits until no other request is in flight on this backend. The busy
        mark is what stops _make_room from evicting this backend mid-request.
        """
        await self._slot_for(name).acquire()
        async with self._busy_cond:
            self._busy.add(name)

    async def release_slot(self, name: str) -> None:
        """Clear the busy mark and release the backend's concurrency slot."""
        async with self._busy_cond:
            self._busy.discard(name)
            self._busy_cond.notify_all()
        self._slot_for(name).release()

    @asynccontextmanager
    async def request_slot(self, name: str):
        """
        Hold the concurrency-1 slot for a backend for the duration of a
        request. Non-streaming routes use this directly; streaming routes
        use acquire_slot/release_slot so the slot survives the response body.
        """
        await self.acquire_slot(name)
        try:
            yield
        finally:
            await self.release_slot(name)

    def busy_backends(self) -> list[str]:
        """Names of backends with an in-flight request (for /status)."""
        return sorted(self._busy)

    # ── Sysram accounting (CPU backends) ─────────────────────────────────

    def _tracked_ram(self, loader: ServiceLoader) -> float:
        """RAM attributed to a CPU service."""
        return loader.config.expected_ram_gb

    async def _mem_total_gb(self) -> float | None:
        """Total system RAM in GB, from a single /proc/meminfo read."""
        res = await query_sysram_gb()
        return res[0] if res else None

    async def available_sysram_gb(self) -> float:
        """System RAM available for CPU backends."""
        res = await query_sysram_gb()
        if res is not None:
            total, used = res
            return total - self.sysram_reserve_gb - used
        # Fallback: tracked bookkeeping with a reasonable total estimate
        used = sum(self._tracked_ram(s) for s in self._running())
        return 64.0 - self.sysram_reserve_gb - used

    async def available_vram_gb(self) -> float:
        """
        VRAM available for a new backend.

        Uses nvidia-smi's actual free VRAM (total - used) minus our reserve.
        This accounts for all GPU consumers (Xorg, Steam, etc.), not just
        our managed backends.

        Falls back to per-process accounting if nvidia-smi is unavailable.
        """
        # Primary: use nvidia-smi's actual free VRAM.
        used = await query_vram_used_gb()
        if used is not None:
            return self.total_vram_gb - self.vram_reserve_gb - used

        # Fallback: per-process accounting of our managed backends.
        per_pid = await query_per_process_vram()
        if per_pid:
            our_vram = 0.0
            for loader in self._running():
                pid = loader.pid
                if pid is not None:
                    our_vram += per_pid.get(pid, 0.0)
                else:
                    our_vram += self._tracked_vram(loader)
            our_vram += sum(self._extra_vram.values())
            return self.total_vram_gb - self.vram_reserve_gb - our_vram

        # Last resort: tracked bookkeeping.
        return self.total_vram_gb - self.vram_reserve_gb - sum(
            self._tracked_vram(s) for s in self._running()
        )

    # ── Core workflow ────────────────────────────────────────────────────

    async def ensure_running(self, name: str, extra_vram_gb: float = 0.0) -> ServiceLoader:
        """
        Guarantee the named backend is running and healthy, evicting others
        if VRAM pressure demands it. Returns its ServiceLoader.

        If the only backends that could be evicted are busy (have in-flight
        requests), this *waits for one to finish* rather than killing it —
        so a new request queues behind a running one. The wait happens
        outside the orchestrator lock, so unrelated backends are not blocked.

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

        while True:
            try:
                return await self._ensure_running_locked(name, extra_vram_gb, is_cpu)
            except _BusyBlock as block:
                logger.info(
                    "VRAM pressure: waiting for in-flight backend(s) %s to finish "
                    "before making room for '%s' (will not evict a busy backend)",
                    ", ".join(sorted(block.busy_names)), name,
                )
                # Wait OUTSIDE the lock for one of the busy backends to drain,
                # then retry. The drained backend is now idle and evictable.
                await self._wait_for_busy_drain(block.busy_names)

    async def _wait_for_busy_drain(self, busy_names) -> None:
        """
        Block (without holding self._lock) until at least one of the named
        busy backends is no longer busy. Called when eviction is blocked by
        in-flight requests.
        """
        names = set(busy_names)
        async with self._busy_cond:
            while names and names.issubset(self._busy):
                await self._busy_cond.wait()

    async def _ensure_running_locked(self, name: str, extra_vram_gb: float, is_cpu: bool) -> ServiceLoader:
        """
        ensure_running's body, run under self._lock. Raises _BusyBlock (which
        the caller handles by waiting outside the lock and retrying) when
        eviction is blocked by busy backends.
        """
        loader = self.services[name]
        async with self._lock:
            # Fast path: alive and healthy.
            if loader.is_alive:
                if await loader.is_healthy():
                    increase = extra_vram_gb - self._extra_vram.get(name, 0.0)
                    if increase > 0:
                        await self._make_room_or_block(increase, exclude=name, cpu=is_cpu)
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
                await self._make_room_or_block(needed, exclude=name, cpu=is_cpu)

            # Launch with retries, re-checking VRAM pressure after each failure.
            attempts = 1 + max(0, loader.config.retries)
            started = False
            for attempt in range(1, attempts + 1):
                if await loader.start():
                    started = True
                    break
                logger.warning("[%s] Launch attempt %d/%d failed", name, attempt, attempts)
                # Re-evaluate VRAM pressure — a failed launch may mean the
                # available budget was optimistic (e.g. unmanaged GPU consumers
                # like Xorg). Try evicting more before retrying.
                if attempt < attempts and not is_cpu:
                    await self._make_room_or_block(needed, exclude=name, cpu=False)

            if not started:
                raise RuntimeError(f"Backend '{name}' failed to start after {attempts} attempt(s)")

            # Restore slot cache after successful launch (llama.cpp only).
            await loader.restore_slot_cache()
            return loader

    async def _make_room_or_block(self, needed_gb: float, exclude: str, cpu: bool = False) -> None:
        """
        Make room for needed_gb by evicting idle backends. If it cannot fit
        and the only evictable backends are busy, raise _BusyBlock (the
        caller waits for one to drain and retries). If it genuinely cannot
        fit (exceeds total budget) or no busy backends are blocking, raise
        InsufficientVRAMError.
        """
        try:
            await self._make_room(needed_gb, exclude=exclude, cpu=cpu)
            return
        except InsufficientVRAMError:
            # If it can never fit even with everything evicted, don't wait.
            if cpu:
                mem_total = await self._mem_total_gb()
                budget = (mem_total - self.sysram_reserve_gb) if mem_total is not None else None
                never = budget is not None and needed_gb > budget
            else:
                never = needed_gb > self.total_vram_gb - self.vram_reserve_gb
            if never:
                raise
            busy_victims = [
                s.config.name for s in self._running()
                if s.config.name != exclude and s.config.name in self._busy
            ]
            if busy_victims:
                raise _BusyBlock(busy_victims)
            raise

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
            mem_total = await self._mem_total_gb()
            budget = mem_total - self.sysram_reserve_gb if mem_total is not None else None
            if budget is not None and needed_gb > budget:
                raise InsufficientVRAMError(
                    f"Backend needs {needed_gb:.1f} GB sysram but budget is only {budget:.1f} GB"
                )
        elif needed_gb > self.total_vram_gb - self.vram_reserve_gb:
            raise InsufficientVRAMError(
                f"Backend needs {needed_gb:.1f} GB but budget is only "
                f"{self.total_vram_gb - self.vram_reserve_gb:.1f} GB"
            )

        # Before evicting our own backends, try killing external VRAM hogs.
        if not cpu:
            freed = await self.kill_vram_hogs(
                target_gb=needed_gb - available,
            )
            if freed > 0:
                available = await self.available_vram_gb()
                if available >= needed_gb:
                    return

        # Never evict a backend with an in-flight request — that would kill a
        # request that is already running. Busy backends are skipped here; the
        # caller (_make_room_or_block) waits for them to drain instead.
        victims = sorted(
            (s for s in self._running()
             if s.config.name != exclude
             and s.config.name not in self._busy),
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
            expected_freed_gb: Approximate GB that should have been freed.
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

    # ── VRAM hog detection & killing ──────────────────────────────────────

    def _managed_pids(self) -> set[int]:
        """Return the set of PIDs belonging to our managed backends."""
        pids: set[int] = set()
        for loader in self._running():
            if loader.pid is not None:
                pids.add(loader.pid)
        return pids

    def _is_default_app(self, process_name: str) -> bool:
        """Check if a process name matches a known default GPU consumer."""
        name_lower = process_name.lower()
        for default in DEFAULT_GPU_APPS:
            if default.lower() in name_lower:
                return True
        return False

    async def detect_vram_hogs(self) -> list[tuple[int, float, str]]:
        """
        Detect external GPU processes that are VRAM hogs.

        Returns a list of (PID, VRAM_GB, process_name) tuples for processes
        that are:
        - Not managed by this orchestrator
        - Not a known default GPU consumer (Xorg, steamwebhelper, etc.)
        - Using more than VRAM_HOG_THRESHOLD_GB

        Sorted by VRAM usage descending (biggest hogs first).
        """
        named = await query_per_process_vram_with_names()
        if not named:
            return []

        managed_pids = self._managed_pids()
        hogs: list[tuple[int, float, str]] = []

        for pid, gb, name in named:
            if pid in managed_pids:
                continue
            if self._is_default_app(name):
                continue
            if gb >= VRAM_HOG_THRESHOLD_GB:
                hogs.append((pid, gb, name))

        # Sort by VRAM usage descending
        hogs.sort(key=lambda t: t[1], reverse=True)
        return hogs

    async def kill_vram_hogs(self, target_gb: float | None = None) -> float:
        """
        Kill external VRAM hog processes to free memory.

        Args:
            target_gb: If set, keep killing hogs until at least this many GB
                are freed. If None, kill all detected hogs.

        Returns the total GB freed.
        """
        hogs = await self.detect_vram_hogs()
        if not hogs:
            logger.debug("No external VRAM hogs detected")
            return 0.0

        freed = 0.0
        for pid, gb, name in hogs:
            if target_gb is not None and freed >= target_gb:
                break

            logger.info(
                "VRAM hog: killing '%s' (PID %d, %.1f GB)",
                name, pid, gb,
            )
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                freed += gb
            except (ProcessLookupError, PermissionError, OSError) as exc:
                logger.warning(
                    "Failed to kill VRAM hog PID %d ('%s'): %s",
                    pid, name, exc,
                )

        if freed > 0:
            logger.info("Killed VRAM hogs, freed ~%.1f GB", freed)
            # Brief pause to let the kernel release the VRAM.
            await asyncio.sleep(0.5)
        return freed

    async def _unmanaged_vram_gb(self) -> float:
        """
        Sum of VRAM used by processes not managed by this orchestrator
        and not in the default-apps list.
        """
        named = await query_per_process_vram_with_names()
        if not named:
            return 0.0
        managed_pids = self._managed_pids()
        total = 0.0
        for pid, gb, name in named:
            if pid not in managed_pids and not self._is_default_app(name):
                total += gb
        return total

    # ── VRAM monitoring ──────────────────────────────────────────────────

    async def _vram_monitor_loop(self) -> None:
        """
        Background loop that warns when unmanaged VRAM usage is high.
        Runs until cancelled.
        """
        while True:
            try:
                await asyncio.sleep(30.0)
                unmanaged = await self._unmanaged_vram_gb()
                if unmanaged > VRAM_UNMANAGED_WARN_GB:
                    now = time.monotonic()
                    if (self._vram_warned_at is None
                            or (now - self._vram_warned_at) >= 30.0):
                        self._vram_warned_at = now
                        logger.warning(
                            "Unmanaged VRAM usage: %.1f GB (threshold %.1f GB) — "
                            "external GPU applications detected: %s",
                            unmanaged,
                            VRAM_UNMANAGED_WARN_GB,
                            ", ".join(
                                f"{name}({gb:.1f}GB)"
                                for _, gb, name in await self.detect_vram_hogs()
                            ),
                        )
                else:
                    self._vram_warned_at = None
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("VRAM monitor error: %s", exc)

    async def check_unmanaged_vram(self) -> float:
        """
        Check and warn about unmanaged VRAM usage. Called on startup.

        Returns the unmanaged VRAM in GB.
        """
        unmanaged = await self._unmanaged_vram_gb()
        if unmanaged > VRAM_UNMANAGED_WARN_GB:
            hogs = await self.detect_vram_hogs()
            logger.warning(
                "Startup VRAM check: %.1f GB unmanaged VRAM "
                "(threshold %.1f GB) — external processes: %s",
                unmanaged,
                VRAM_UNMANAGED_WARN_GB,
                ", ".join(
                    f"{name}({gb:.1f}GB)" for _, gb, name in hogs
                ) if hogs else "none above hog threshold",
            )
        return unmanaged

    def start_vram_monitor(self) -> None:
        """Start the background VRAM monitoring task."""
        if self._vram_monitor_task is not None and not self._vram_monitor_task.done():
            return  # Already running
        self._vram_monitor_task = asyncio.create_task(self._vram_monitor_loop())

    async def stop_vram_monitor(self) -> None:
        """Stop the background VRAM monitoring task."""
        if self._vram_monitor_task is not None:
            self._vram_monitor_task.cancel()
            try:
                await self._vram_monitor_task
            except asyncio.CancelledError:
                pass
            self._vram_monitor_task = None

    async def _emit_monitor_event(self, name: str, payload: dict[str, Any]) -> None:
        """Emit an event through all service callbacks (for monitoring events)."""
        for loader in self.services.values():
            if loader._event_callback is not None:
                try:
                    await loader._event_callback(name, payload)
                except Exception as exc:
                    logger.debug("Monitor event callback error: %s", exc)

    # ── Status & shutdown ────────────────────────────────────────────────

    async def get_status(self) -> dict[str, Any]:
        """Snapshot of the whole fleet, suitable for a /status endpoint."""
        hogs = await self.detect_vram_hogs()
        return {
            "total_vram_gb": self.total_vram_gb,
            "vram_reserve_gb": self.vram_reserve_gb,
            "available_vram_gb": round(await self.available_vram_gb(), 2),
            "busy_backends": self.busy_backends(),
            "vram_hogs": [
                {"pid": pid, "vram_gb": round(gb, 2), "name": name}
                for pid, gb, name in hogs
            ],
            "services": {
                name: await s.get_status() for name, s in self.services.items()
            },
        }

    async def shutdown(self) -> None:
        """Gracefully stop all running backends (router shutdown hook)."""
        await self.stop_vram_monitor()
        running = self._running()
        if running:
            logger.info("Shutting down %d running backend(s)", len(running))
            # Save slot caches before stopping llama backends
            await asyncio.gather(*(s.save_slot_cache() for s in running))
            await asyncio.gather(*(s.stop() for s in running))
