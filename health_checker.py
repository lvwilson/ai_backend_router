"""
health_checker.py — Health probing for backend services.

Uses HTTP health checks when aiohttp is available, falling back to a raw
TCP connection probe otherwise. Configurable polling interval and timeout
for startup health waits.
"""

from __future__ import annotations

import asyncio
import logging
import time

try:
    import aiohttp
except ImportError:
    aiohttp = None

logger = logging.getLogger(__name__)


class HealthChecker:
    """
    Probes a backend service's health endpoint.

    Used by ServiceLoader during startup and for periodic liveness checks.
    If no port is configured, probes always succeed (process liveness is
    the caller's responsibility).

    Usage:
        checker = HealthChecker(port=8080)
        healthy = await checker.wait_for_healthy(process)
        is_ok = await checker.probe()
    """

    def __init__(
        self,
        port: int | None = None,
        path: str = "/health",
        host: str = "127.0.0.1",
        scheme: str = "http",
        timeout: float = 30.0,
        interval: float = 2.0,
    ):
        self.port = port
        self.timeout = timeout
        self.interval = interval
        self.url = f"{scheme}://{host}:{port}{path}" if port is not None else ""
        self._host = host
        self._probe = self._probe_http if aiohttp is not None else self._probe_socket

    @property
    def is_configured(self) -> bool:
        """True if a health endpoint is configured."""
        return self.port is not None

    async def probe(self) -> bool:
        """
        Perform a single health check probe.

        Returns True if the service responds healthily.
        """
        if not self.is_configured:
            return True  # No endpoint = assume healthy if process alive
        return await self._probe()

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
            return process is not None and process.returncode is None

        deadline = time.monotonic() + self.timeout

        while time.monotonic() < deadline:
            if process is not None and process.returncode is not None:
                # Process exited during startup
                return False

            if await self._probe():
                return True

            await asyncio.sleep(self.interval)

        return False

    async def _probe_http(self) -> bool:
        """Execute a single HTTP GET against the health endpoint via aiohttp."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.url,
                    timeout=aiohttp.ClientTimeout(total=3.0),
                ) as resp:
                    return resp.status < 500
        except Exception:
            return False

    async def _probe_socket(self) -> bool:
        """Fallback health probe using raw TCP connection."""
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(self._host, self.port),
                timeout=3.0,
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False
