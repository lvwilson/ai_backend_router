"""
vram_tracker.py — VRAM measurement and drift tracking for backend services.

Measures actual VRAM delta on service start and compares against expected
values from config, emitting drift warnings with spam control.

Designed to be used per-service by ServiceLoader and aggregated by the
orchestrator for eviction decisions.
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


async def query_vram_used_gb() -> float | None:
    """
    Query total VRAM used (across all GPUs) via nvidia-smi.

    Returns VRAM in GB, or None if nvidia-smi is unavailable.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return None
        total_mi = 0.0
        for line in stdout.decode().strip().splitlines():
            line = line.strip()
            if line:
                total_mi += float(line)
        return total_mi / 1024.0
    except FileNotFoundError:
        return None


class VramTracker:
    """
    Tracks VRAM usage for a single service.

    Measures the VRAM delta when the service starts and compares it
    against the expected value, emitting drift warnings when they
    diverge significantly.

    Usage:
        tracker = VramTracker(service_name, expected_vram_gb=4.0)
        await tracker.record_pre_start()
        # ... start the process ...
        actual = await tracker.measure_delta()
    """

    DRIFT_THRESHOLD_GB = 2.0       # Warn if actual differs from expected by >2 GB
    DRIFT_WARN_INTERVAL = 300.0    # Seconds between drift warnings (5 min spam control)

    def __init__(
        self,
        service_name: str,
        expected_vram_gb: float = 0.0,
    ):
        self._service_name = service_name
        self._expected_vram_gb = expected_vram_gb
        self._vram_before_start: float | None = None
        self._actual_vram_gb: float | None = None
        self._drift_warned_at: float | None = None

    @property
    def actual_vram_gb(self) -> float | None:
        """Measured VRAM delta attributable to this service, or None."""
        return self._actual_vram_gb

    @property
    def expected_vram_gb(self) -> float:
        return self._expected_vram_gb

    async def record_pre_start(self) -> None:
        """Capture VRAM usage before the service process is launched."""
        self._vram_before_start = await query_vram_used_gb()
        self._actual_vram_gb = None

    async def measure_delta(self) -> float | None:
        """
        Measure VRAM delta since pre-start snapshot.

        Returns the measured delta in GB, or None if measurement failed.
        Sets self._actual_vram_gb as a side effect.
        """
        if self._vram_before_start is None:
            return None

        current = await query_vram_used_gb()
        if current is None:
            return None

        delta = current - self._vram_before_start
        self._actual_vram_gb = round(delta, 2)
        return self._actual_vram_gb

    async def check_drift(self, emit_fn: callable) -> None:
        """
        Compare measured VRAM against expected and emit a drift warning
        if they diverge significantly.

        Args:
            emit_fn: Async callable(event_name, payload) for emitting events.
        """
        if self._actual_vram_gb is None:
            return

        expected = self._expected_vram_gb
        if expected <= 0:
            return

        drift = abs(self._actual_vram_gb - expected)
        if drift <= self.DRIFT_THRESHOLD_GB:
            return

        now = time.monotonic()
        if self._drift_warned_at is not None and (now - self._drift_warned_at) < self.DRIFT_WARN_INTERVAL:
            return

        self._drift_warned_at = now
        await emit_fn(
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
            self._service_name,
            expected,
            self._actual_vram_gb,
            drift,
        )

    def reset(self) -> None:
        """Reset tracking state for a restart."""
        self._vram_before_start = None
        self._actual_vram_gb = None
        self._drift_warned_at = None
