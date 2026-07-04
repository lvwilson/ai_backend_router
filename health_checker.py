"""
health_checker.py — Health probing strategies for backend services.

Supports HTTP-based health checks (via aiohttp) with a socket-based
TCP fallback when aiohttp is unavailable. Configurable polling interval
and timeout for startup health waits.

Designed to be injected into ServiceLoader so the lifecycle code stays
focused on process management.
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class HealthCheckConfig:
    """Configuration for health checking a backend service."""

    def __init__(
        self,
        port: int | None = None,
        health_path: str = "/health",
        health_timeout: float = 30.0,
        health_interval: float = 2.0,
        health_host: str = "127.0.0.1",
        health_scheme: str = "http",
    ):
        self.port = port
        self.health_path = health_path
        self.health_timeout = health_timeout
        self.health_interval = health_interval
        self.health_host = health_host
        self.health_scheme = health_scheme

    @property
    def health_url(self) -> str:
        """Build the full health check URL from config."""
        if self.port is None:
            return ""
        return f"{self.health_scheme}://{self.health_host}:{self.port}{self.health_path}"

    @property
    def is_configured(self) -> bool:
        """True if a health endpoint is configured."""
        return self.port is not None


class HealthChecker:
    """
    Probes a backend service's health endpoint.

    Supports HTTP probing (via aiohttp) with a TCP socket fallback.
    Used by ServiceLoader during startup and for periodic liveness checks.

    Usage:
        checker = HealthChecker(HealthCheckConfig(port=8080))
        healthy = await checker.wait_for_healthy(process)
        is_ok = await checker.probe()
    """

    def __init__(self, config: HealthCheckConfig):
        self._config = config
        self._use_aiohttp: bool = False
        self._probe_method: callable = self._probe_socket

        # Determine available probe strategy
        try:
            import aiohttp  # noqa: F401
            self._use_aiohttp = True
            self._probe_method = self._probe_http
        except ImportError:
            pass

    @property
    def health_url(self) -> str:
        return self._config.health_url

    @property
    def is_configured(self) -> bool:
        return self._config.is_configured

    async def probe(self) -> bool:
        """
        Perform a single health check probe.

        Returns True if the service responds healthily.
        """
        if not self.is_configured:
            return True  # No endpoint = assume healthy if process alive
        return await self._probe_method()

    async def wait_for_healthy(
        self,
        process: asyncio.subprocess.Process | None,
    ) -> bool:
        """
        Poll the health endpoint until it succeeds or timeout is reached.

        Args:
            process: The subprocess handle to check for unexpected exit.

        Returns True if the service became healthy before timeout.
        """
        if not self.is_configured:
            # No health endpoint — consider it healthy if process is alive
            if process is not None and process.returncode is None:
                return True
            return False

        deadline = time.monotonic() + self._config.health_timeout

        while time.monotonic() < deadline:
            if process is not None and process.returncode is not None:
                # Process exited during startup
                return False

            if await self._probe_method():
                return True

            await asyncio.sleep(self._config.health_interval)

        return False

    async def _probe_http(self) -> bool:
        """Execute a single HTTP GET against the health endpoint via aiohttp."""
        try:
            import aiohttp
        except ImportError:
            return await self._probe_socket()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self._config.health_url,
                    timeout=aiohttp.ClientTimeout(total=3.0),
                ) as resp:
                    return resp.status < 500
        except Exception:
            return False

    async def _probe_socket(self) -> bool:
        """Fallback health probe using raw TCP connection."""
        if not self._config.port:
            return True
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(
                    self._config.health_host,
                    self._config.port,
                ),
                timeout=3.0,
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False
