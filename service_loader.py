"""
service_loader.py — Backend process lifecycle manager.

Manages a single backend process (llama.cpp, CrispASR, ComfyUI) with:
  • Start / stop (graceful) / kill (force)
  • State machine (IDLE → STARTING → RUNNING → STOPPING → DEAD)
  • Process group signals (SIGTERM/SIGKILL via os.setsid)
  • Health probing via HealthChecker (HTTP or socket)
  • Async context manager support
  • VRAM delta measurement and drift warnings

Designed to be instantiated per-model by a larger orchestrator.

Usage:
    config = ServiceConfig(
        name="llama-small",
        binary="llama-server",
        args=["-m", "/path/to/model.gguf", "--port", "8080"],
        port=8080,
        expected_vram_gb=4.0,
        expected_ram_gb=2.0,
    )
    loader = ServiceLoader(config)
    await loader.start()
    ...
    await loader.stop()
"""

from __future__ import annotations

import asyncio
import aiohttp
import enum
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from health_checker import HealthChecker

EventCallback = Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]]

logger = logging.getLogger(__name__)


# ── VRAM helpers ───────────────────────────────────────────────────────────

VRAM_DRIFT_THRESHOLD_GB = 2.0       # Warn if actual differs from expected by >2 GB
VRAM_DRIFT_WARN_INTERVAL = 300.0    # Seconds between drift warnings (5 min)


async def query_vram_used_gb(retries: int = 2) -> float | None:
    """
    Query total VRAM used (across all GPUs) via nvidia-smi.

    Retries transient failures (nvidia-smi can fail sporadically under
    load or when subprocess spawning hiccups). Returns VRAM in GB, or
    None if nvidia-smi is unavailable after all attempts.
    """
    for attempt in range(retries + 1):
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.debug(
                    "nvidia-smi failed (attempt %d, rc=%s): %s",
                    attempt, proc.returncode, stderr.decode(errors="replace").strip(),
                )
                if attempt < retries:
                    await asyncio.sleep(0.5)
                    continue
                return None
            total_mi = 0.0
            for line in stdout.decode().strip().splitlines():
                line = line.strip()
                if line:
                    total_mi += float(line)
            return total_mi / 1024.0
        except FileNotFoundError:
            return None
        except Exception as exc:
            logger.debug("nvidia-smi query error (attempt %d): %s", attempt, exc)
            if attempt < retries:
                await asyncio.sleep(0.5)
                continue
            return None
    return None


# ── State machine ──────────────────────────────────────────────────────────

class ServiceState(enum.Enum):
    """Possible states of a managed backend service."""
    IDLE = "idle"            # Not running, not attempting to start
    STARTING = "starting"    # Process launched, waiting for health check
    RUNNING = "running"      # Healthy and accepting requests
    STOPPING = "stopping"    # Graceful shutdown in progress (SIGTERM sent)
    DEAD = "dead"            # Process exited unexpectedly or was killed


# ── Configuration ──────────────────────────────────────────────────────────

@dataclass
class ServiceConfig:
    """Configuration for a single backend service instance."""

    name: str                                    # Unique identifier (e.g. "llama-small")
    binary: str                                  # Executable path or command
    args: list[str] = field(default_factory=list)  # Command-line arguments
    env: dict[str, str] | None = None            # Extra environment variables
    working_dir: str | None = None               # CWD for the process

    # Network / health
    port: int | None = None                      # Listen port (for health checks)
    health_path: str = "/health"                 # HTTP path for health checks
    health_timeout: float = 30.0                 # Seconds to wait for health on start
    health_interval: float = 2.0                 # Seconds between health check polls
    health_host: str = "127.0.0.1"
    health_scheme: str = "http"

    # Resource expectations (used for warnings and eviction planning)
    expected_vram_gb: float = 0.0                # Estimated VRAM usage in GB
    expected_ram_gb: float = 0.0                 # Estimated RAM usage in GB

    # Graceful shutdown
    stop_timeout: float = 10.0                   # Seconds to wait after SIGTERM before SIGKILL

    # Launch retries (total attempts = 1 + retries)
    retries: int = 1                             # Extra launch attempts on failure

    # Slot cache (llama.cpp prompt cache)
    slot_save_path: str | None = None            # Directory for llama-server slot persistence cache


# ── ServiceLoader ──────────────────────────────────────────────────────────

class ServiceLoader:
    """
    Manages the full lifecycle of a single backend process.

    One instance per backend/model. The orchestrator creates several of these
    and coordinates VRAM-based eviction across them.
    """

    def __init__(
        self,
        config: ServiceConfig,
        event_callback: EventCallback | None = None,
    ):
        self.config = config
        self._state = ServiceState.IDLE
        self._process: asyncio.subprocess.Process | None = None
        self._started_at: float | None = None
        self._event_callback = event_callback

        self.health_checker = HealthChecker(
            port=config.port,
            path=config.health_path,
            host=config.health_host,
            scheme=config.health_scheme,
            timeout=config.health_timeout,
            interval=config.health_interval,
        )

        # VRAM tracking state
        self._vram_before_start: float | None = None
        self._actual_vram_gb: float | None = None
        self._vram_drift_warned_at: float | None = None

    # ── Internal helpers ─────────────────────────────────────────────────

    async def _emit(self, name: str, payload: dict[str, Any] | None = None) -> None:
        """Emit a lifecycle event if a callback is registered."""
        if self._event_callback is not None:
            try:
                await self._event_callback(name, payload or {})
            except Exception as exc:
                logger.debug("[%s] Event callback error for '%s': %s", self.config.name, name, exc)

    # ── VRAM tracking ────────────────────────────────────────────────────

    async def record_pre_start_vram(self) -> None:
        """Capture VRAM usage before the service process is launched."""
        self._vram_before_start = await query_vram_used_gb()
        self._actual_vram_gb = None

    async def measure_vram_delta(self) -> float | None:
        """Measure VRAM delta since pre-start snapshot."""
        if self._vram_before_start is None:
            return None
        current = await query_vram_used_gb()
        if current is None:
            return None
        delta = current - self._vram_before_start
        self._actual_vram_gb = round(delta, 2)
        return self._actual_vram_gb

    async def check_vram_drift(self) -> None:
        """Compare measured VRAM against expected and emit a drift warning if diverged."""
        if self._actual_vram_gb is None:
            return
        expected = self.config.expected_vram_gb
        if expected <= 0:
            return
        drift = abs(self._actual_vram_gb - expected)
        if drift <= VRAM_DRIFT_THRESHOLD_GB:
            return
        now = time.monotonic()
        if self._vram_drift_warned_at is not None and (now - self._vram_drift_warned_at) < VRAM_DRIFT_WARN_INTERVAL:
            return
        self._vram_drift_warned_at = now
        await self._emit(
            "resource_warning",
            {
                "type": "vram_drift",
                "expected_gb": expected,
                "actual_gb": self._actual_vram_gb,
                "delta_gb": round(drift, 2),
            },
        )
        logger.warning(
            "[%s] VRAM drift — expected %.1f GB, measured %.2f GB (%.2f GB off)",
            self.config.name,
            expected,
            self._actual_vram_gb,
            drift,
        )

    def reset_vram_tracking(self) -> None:
        """Reset VRAM tracking state for a restart."""
        self._vram_before_start = None
        self._actual_vram_gb = None
        self._vram_drift_warned_at = None

    # ── Public state ─────────────────────────────────────────────────────

    @property
    def state(self) -> ServiceState:
        return self._state

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process else None

    @property
    def is_alive(self) -> bool:
        """True if process is running and healthy."""
        if self._state not in (ServiceState.RUNNING, ServiceState.STARTING):
            return False
        if self._process is None:
            return False
        return self._process.returncode is None

    @property
    def actual_vram_gb(self) -> float | None:
        """Measured VRAM delta attributable to this service, or None."""
        return self._actual_vram_gb

    @property
    def health_url(self) -> str:
        return self.health_checker.url

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> bool:
        """
        Launch the backend process and wait for it to become healthy.

        Returns True if the service is running and healthy, False otherwise.
        """
        if self._state in (ServiceState.RUNNING, ServiceState.STARTING):
            logger.info("[%s] Already %s, skipping start", self.config.name, self._state.value)
            return self._state == ServiceState.RUNNING

        # Kill any stale OS process before launching a new one.
        # This prevents VRAM leaks when a previous stop/kill didn't fully
        # terminate the process (e.g. SIGKILL timeout).
        if self._process is not None and self._process.returncode is None:
            logger.warning(
                "[%s] Stale process detected (PID %d, state=%s) — killing before restart",
                self.config.name, self._process.pid, self._state.value,
            )
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self._process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                pass

        self._state = ServiceState.STARTING
        self._started_at = None
        self.reset_vram_tracking()

        # Measure VRAM before launch
        await self.record_pre_start_vram()

        # Build environment
        env = os.environ.copy()
        if self.config.env:
            env.update(self.config.env)

        try:
            logger.info(
                "[%s] Starting: %s %s",
                self.config.name,
                self.config.binary,
                " ".join(self.config.args),
            )

            self._process = await asyncio.create_subprocess_exec(
                self.config.binary,
                *self.config.args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.config.working_dir,
                preexec_fn=os.setsid,  # Create new process group for clean signal delivery
            )

            self._started_at = time.monotonic()
            logger.info("[%s] Process started, PID=%d", self.config.name, self._process.pid)

            # Wait for health
            healthy = await self.health_checker.wait_for_healthy(self._process)

            if healthy:
                self._state = ServiceState.RUNNING
                await self.measure_vram_delta()
                await self.check_vram_drift()
                await self._emit("started", {"pid": self._process.pid})
                logger.info("[%s] Service is healthy and running", self.config.name)
                return True
            else:
                self._state = ServiceState.DEAD
                await self._cleanup_process()
                await self._emit("unhealthy", {"reason": "health_check_failed_on_start"})
                logger.warning("[%s] Service failed health check on start", self.config.name)
                return False

        except Exception as exc:
            self._state = ServiceState.DEAD
            await self._cleanup_process()
            await self._emit("unhealthy", {"reason": str(exc)})
            logger.error("[%s] Failed to start: %s", self.config.name, exc)
            return False

    async def stop(self) -> bool:
        """
        Gracefully stop the service (SIGTERM, then wait, then SIGKILL).

        Returns True if the process exited cleanly.
        """
        if self._state not in (ServiceState.RUNNING, ServiceState.STARTING):
            logger.info("[%s] Not running (state=%s), skipping stop", self.config.name, self._state.value)
            # Still reset VRAM tracking so next start gets a fresh measurement.
            self.reset_vram_tracking()
            return True

        self._state = ServiceState.STOPPING
        await self._emit("stopping", {})

        proc = self._process
        if proc is None or proc.returncode is not None:
            self._state = ServiceState.DEAD
            self.reset_vram_tracking()
            await self._emit("stopped", {"clean": True})
            return True

        # Send SIGTERM to the process group
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            logger.info("[%s] Sent SIGTERM to PID %d", self.config.name, proc.pid)
        except ProcessLookupError:
            logger.info("[%s] Process %d already exited", self.config.name, proc.pid)
            self._state = ServiceState.DEAD
            self.reset_vram_tracking()
            await self._emit("stopped", {"clean": True})
            return True

        # Wait for graceful exit
        try:
            await asyncio.wait_for(proc.wait(), timeout=self.config.stop_timeout)
            self._state = ServiceState.DEAD
            self.reset_vram_tracking()
            await self._emit("stopped", {"clean": True})
            logger.info("[%s] Stopped gracefully (exit code %d)", self.config.name, proc.returncode)
            return True
        except asyncio.TimeoutError:
            logger.warning("[%s] Graceful stop timed out (%.1fs), sending SIGKILL", self.config.name, self.config.stop_timeout)
            result = await self._force_kill()
            return result

    async def kill(self) -> bool:
        """
        Force-kill the service (SIGKILL).

        Returns True if the process was terminated.
        """
        if self._state == ServiceState.IDLE or self._state == ServiceState.DEAD:
            return True

        self._state = ServiceState.STOPPING
        proc = self._process
        if proc is None or proc.returncode is not None:
            self._state = ServiceState.DEAD
            return True

        return await self._force_kill()

    async def is_healthy(self) -> bool:
        """
        Perform a single health check probe.

        Returns True if the service responds to its health endpoint.
        """
        if not self.is_alive:
            return False
        return await self.health_checker.probe()

    async def get_status(self) -> dict[str, Any]:
        """
        Return a snapshot of the service's current status.

        Suitable for API responses or logging.
        """
        return {
            "name": self.config.name,
            "state": self._state.value,
            "pid": self.pid,
            "is_alive": self.is_alive,
            "health_url": self.health_url,
            "expected_vram_gb": self.config.expected_vram_gb,
            "expected_ram_gb": self.config.expected_ram_gb,
            "actual_vram_gb": self.actual_vram_gb,
            "started_at": self._started_at,
        }

    # ── Slot cache (llama.cpp prompt cache) ──────────────────────────────

    async def save_slot_cache(self) -> None:
        """
        Save the slot prompt cache via llama-server API.

        Only meaningful for llama.cpp backends that have `slot_save_path`
        configured. Uses the backend name as the cache filename so each
        backend gets its own cache file.
        """
        if not self.config.slot_save_path:
            return
        if self._state != ServiceState.RUNNING:
            return
        filename = f"{self.config.name}.bin"
        url = f"{self.config.health_scheme}://{self.config.health_host}:{self.config.port}/slots/0?action=save"
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.post(
                    url,
                    json={"filename": filename},
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    if resp.status == 200:
                        logger.info("[%s] Slot cache saved as '%s'", self.config.name, filename)
                    else:
                        logger.warning(
                            "[%s] Slot cache save returned %d", self.config.name, resp.status
                        )
        except Exception as exc:
            logger.warning("[%s] Failed to save slot cache: %s", self.config.name, exc)

    async def restore_slot_cache(self) -> None:
        """
        Restore the slot prompt cache via llama-server API.

        Only meaningful for llama.cpp backends that have `slot_save_path`
        configured. Uses the backend name as the cache filename.
        """
        if not self.config.slot_save_path:
            return
        if self._state != ServiceState.RUNNING:
            return
        filename = f"{self.config.name}.bin"
        url = f"{self.config.health_scheme}://{self.config.health_host}:{self.config.port}/slots/0?action=restore"
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.post(
                    url,
                    json={"filename": filename},
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    if resp.status == 200:
                        logger.info("[%s] Slot cache restored from '%s'", self.config.name, filename)
                    else:
                        logger.warning(
                            "[%s] Slot cache restore returned %d", self.config.name, resp.status
                        )
        except Exception as exc:
            logger.warning("[%s] Failed to restore slot cache: %s", self.config.name, exc)

    # ── Private helpers ──────────────────────────────────────────────────

    async def _force_kill(self) -> bool:
        """Send SIGKILL to the process group and wait for exit."""
        proc = self._process
        if proc is None or proc.returncode is not None:
            self._state = ServiceState.DEAD
            self.reset_vram_tracking()
            return True

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            logger.info("[%s] Sent SIGKILL to PID %d", self.config.name, proc.pid)
        except ProcessLookupError:
            self._state = ServiceState.DEAD
            self.reset_vram_tracking()
            await self._emit("killed", {"clean": False})
            return True

        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
            self._state = ServiceState.DEAD
            self.reset_vram_tracking()
            await self._emit("killed", {"clean": False})
            logger.info("[%s] Killed (exit code %d)", self.config.name, proc.returncode)
            return True
        except asyncio.TimeoutError:
            self._state = ServiceState.DEAD
            self.reset_vram_tracking()
            await self._emit("killed", {"clean": False, "force": True})
            logger.error(
                "[%s] Process %d did not exit after SIGKILL — VRAM tracking reset, "
                "but process may still hold GPU memory",
                self.config.name, proc.pid,
            )
            return False

    async def _cleanup_process(self) -> None:
        """
        Ensure the subprocess handle is cleaned up.

        If the process is still running in the OS, force-kill it to prevent
        silent VRAM leaks. This is critical: a process that fails its health
        check may still be alive and holding GPU memory.
        """
        if self._process is not None:
            proc = self._process
            if proc.returncode is None:
                # Process still running — kill it to free VRAM
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    logger.info("[%s] Killed stale process PID %d during cleanup",
                                self.config.name, proc.pid)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    logger.warning("[%s] Process %d did not exit after cleanup SIGKILL",
                                   self.config.name, proc.pid)
            self._process = None

    # ── Context manager ──────────────────────────────────────────────────

    async def __aenter__(self) -> "ServiceLoader":
        """Async context manager support."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager cleanup."""
        await self.stop()
