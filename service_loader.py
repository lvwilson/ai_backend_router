"""
service_loader.py — Backend process lifecycle manager.

Manages a single backend process (llama.cpp, CrispASR, ComfyUI) with:
  • Start / stop (graceful) / kill (force)
  • State machine (IDLE → STARTING → RUNNING → STOPPING → DEAD)
  • Process group signals (SIGTERM/SIGKILL via os.setsid)
  • Async context manager support

Delegates to injected subsystems:
  • HealthChecker  — health probing (HTTP or socket)
  • VramTracker    — VRAM delta measurement and drift warnings
  • EventDispatcher — lifecycle event emission

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
import enum
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from typing import Any

from events import EventCallback, EventDispatcher
from health_checker import HealthCheckConfig, HealthChecker
from vram_tracker import VramTracker

logger = logging.getLogger(__name__)


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

    # ── Factory helpers ────────────────────────────────────────────────

    def health_check_config(self) -> HealthCheckConfig:
        """Build a HealthCheckConfig from this service config."""
        return HealthCheckConfig(
            port=self.port,
            health_path=self.health_path,
            health_timeout=self.health_timeout,
            health_interval=self.health_interval,
            health_host=self.health_host,
            health_scheme=self.health_scheme,
        )

    def health_checker(self) -> HealthChecker:
        """Build a HealthChecker from this service config."""
        return HealthChecker(self.health_check_config())

    def vram_tracker(self) -> VramTracker:
        """Build a VramTracker from this service config."""
        return VramTracker(
            service_name=self.name,
            expected_vram_gb=self.expected_vram_gb,
        )

    @property
    def health_url(self) -> str:
        """Build the full health check URL from config."""
        if self.port is None:
            return ""
        return f"{self.health_scheme}://{self.health_host}:{self.port}{self.health_path}"


# ── ServiceLoader ──────────────────────────────────────────────────────────

class ServiceLoader:
    """
    Manages the full lifecycle of a single backend process.

    One instance per backend/model. The orchestrator creates several of these
    and coordinates VRAM-based eviction across them.

    Subsystems are injected (or auto-created from config):
      • health_checker  — probes the health endpoint
      • vram_tracker    — measures VRAM delta and drift
      • event_dispatcher — emits lifecycle events
    """

    def __init__(
        self,
        config: ServiceConfig,
        event_callback: EventCallback | None = None,
        health_checker: HealthChecker | None = None,
        vram_tracker: VramTracker | None = None,
    ):
        self.config = config
        self._state = ServiceState.IDLE
        self._process: asyncio.subprocess.Process | None = None
        self._started_at: float | None = None

        # Injected subsystems (auto-create from config if not provided)
        self.health_checker = health_checker or config.health_checker()
        self.vram_tracker = vram_tracker or config.vram_tracker()
        self.event_dispatcher = EventDispatcher(
            callback=event_callback,
            service_name=config.name,
        )

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
        return self.vram_tracker.actual_vram_gb

    @property
    def health_url(self) -> str:
        return self.config.health_url

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> bool:
        """
        Launch the backend process and wait for it to become healthy.

        Returns True if the service is running and healthy, False otherwise.
        """
        if self._state in (ServiceState.RUNNING, ServiceState.STARTING):
            logger.info("[%s] Already %s, skipping start", self.config.name, self._state.value)
            return self._state == ServiceState.RUNNING

        self._state = ServiceState.STARTING
        self._started_at = None
        self.vram_tracker.reset()

        # Measure VRAM before launch
        await self.vram_tracker.record_pre_start()

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
                await self.vram_tracker.measure_delta()
                await self.vram_tracker.check_drift(self.event_dispatcher.emit)
                await self.event_dispatcher.emit("started", {"pid": self._process.pid})
                logger.info("[%s] Service is healthy and running", self.config.name)
                return True
            else:
                self._state = ServiceState.DEAD
                await self._cleanup_process()
                await self.event_dispatcher.emit("unhealthy", {"reason": "health_check_failed_on_start"})
                logger.warning("[%s] Service failed health check on start", self.config.name)
                return False

        except Exception as exc:
            self._state = ServiceState.DEAD
            await self._cleanup_process()
            await self.event_dispatcher.emit("unhealthy", {"reason": str(exc)})
            logger.error("[%s] Failed to start: %s", self.config.name, exc)
            return False

    async def stop(self) -> bool:
        """
        Gracefully stop the service (SIGTERM, then wait, then SIGKILL).

        Returns True if the process exited cleanly.
        """
        if self._state not in (ServiceState.RUNNING, ServiceState.STARTING):
            logger.info("[%s] Not running (state=%s), skipping stop", self.config.name, self._state.value)
            return True

        self._state = ServiceState.STOPPING
        await self.event_dispatcher.emit("stopping", {})

        proc = self._process
        if proc is None or proc.returncode is not None:
            self._state = ServiceState.DEAD
            await self.event_dispatcher.emit("stopped", {"clean": True})
            return True

        # Send SIGTERM to the process group
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            logger.info("[%s] Sent SIGTERM to PID %d", self.config.name, proc.pid)
        except ProcessLookupError:
            logger.info("[%s] Process %d already exited", self.config.name, proc.pid)
            self._state = ServiceState.DEAD
            await self.event_dispatcher.emit("stopped", {"clean": True})
            return True

        # Wait for graceful exit
        try:
            await asyncio.wait_for(proc.wait(), timeout=self.config.stop_timeout)
            self._state = ServiceState.DEAD
            await self.event_dispatcher.emit("stopped", {"clean": True})
            logger.info("[%s] Stopped gracefully (exit code %d)", self.config.name, proc.returncode)
            return True
        except asyncio.TimeoutError:
            logger.warning("[%s] Graceful stop timed out (%.1fs), sending SIGKILL", self.config.name, self.config.stop_timeout)
            return await self._force_kill()

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
            "actual_vram_gb": self.vram_tracker.actual_vram_gb,
            "started_at": self._started_at,
        }

    # ── Internal helpers ─────────────────────────────────────────────────

    async def _force_kill(self) -> bool:
        """Send SIGKILL to the process group and wait for exit."""
        proc = self._process
        if proc is None or proc.returncode is not None:
            self._state = ServiceState.DEAD
            return True

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            logger.info("[%s] Sent SIGKILL to PID %d", self.config.name, proc.pid)
        except ProcessLookupError:
            self._state = ServiceState.DEAD
            await self.event_dispatcher.emit("killed", {"clean": False})
            return True

        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
            self._state = ServiceState.DEAD
            await self.event_dispatcher.emit("killed", {"clean": False})
            logger.info("[%s] Killed (exit code %d)", self.config.name, proc.returncode)
            return True
        except asyncio.TimeoutError:
            self._state = ServiceState.DEAD
            await self.event_dispatcher.emit("killed", {"clean": False, "force": True})
            logger.error("[%s] Process %d did not exit after SIGKILL", self.config.name, proc.pid)
            return False

    async def _cleanup_process(self) -> None:
        """Ensure the subprocess handle is cleaned up."""
        if self._process is not None:
            try:
                # If still running, wait briefly for it to finish
                if self._process.returncode is None:
                    try:
                        await asyncio.wait_for(self._process.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        pass
            except Exception:
                pass
            self._process = None

    # ── Context manager ──────────────────────────────────────────────────

    async def __aenter__(self) -> "ServiceLoader":
        """Async context manager support."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager cleanup."""
        await self.stop()
