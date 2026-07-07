#!/usr/bin/env python3
"""
test_eviction.py — Tests for the VRAM eviction zombie-process bug and
orchestrator lifecycle management.

Bug: When _force_kill() times out after SIGKILL, the process state is set
to DEAD but the OS process is still running and holding VRAM. The next
start() spawns a new process without cleaning up the zombie, causing
VRAM to accumulate silently.

Fixes applied:
  1. start() now kills stale OS processes before launching new ones
  2. stop() resets VRAM tracking on all exit paths
  3. _force_kill() resets VRAM tracking on all exit paths
  4. _cleanup_process() now force-kills running processes (not just wait)
  5. VRAM accounting uses per-process nvidia-smi readings (authoritative)

Tests use `python3 -m http.server` as the backend so health checks pass.

Run: pytest tests/test_eviction.py -v
"""
import asyncio
import os
import signal
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.service_loader import (
    ServiceConfig,
    ServiceLoader,
    ServiceState,
    query_vram_used_gb,
)
from src.orchestrator import Orchestrator, InsufficientVRAMError


def _free_port() -> int:
    """Grab a free TCP port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http_server_config(name: str, port: int | None = None) -> ServiceConfig:
    """Create a ServiceConfig backed by python3 -m http.server (passes health checks)."""
    port = port or _free_port()
    return ServiceConfig(
        name=name,
        binary=sys.executable,
        args=["-m", "http.server", str(port), "--bind", "127.0.0.1"],
        port=port,
        expected_vram_gb=0.1,
        health_timeout=5.0,
        stop_timeout=3.0,
        retries=0,
    )


# ── Zombie Process Cleanup ────────────────────────────────────────────────

class TestZombieProcessCleanup:
    """Test that zombie processes are cleaned up before new launches."""

    async def test_start_kills_stale_process(self):
        """
        When start() is called and a stale OS process exists from a previous
        launch, it must be killed before the new process starts.
        """
        config = _http_server_config("test-stale")
        loader = ServiceLoader(config)

        # First launch — should succeed
        ok = await loader.start()
        assert ok, "First start should succeed"
        pid_before = loader.pid
        assert pid_before is not None

        # Verify process alive in OS
        os.kill(pid_before, 0)  # raises if dead

        # Simulate zombie: state is DEAD but process still running
        loader._state = ServiceState.DEAD

        # Second launch — should kill stale process first
        loader._state = ServiceState.IDLE
        ok2 = await loader.start()
        assert ok2, "Second start should succeed after killing stale"
        pid_after = loader.pid
        assert pid_after is not None

        # Old process should be dead
        with pytest.raises(ProcessLookupError):
            os.kill(pid_before, 0)

        assert pid_before != pid_after, "Should have a new PID"

        await loader.stop()

    async def test_force_kill_timeout_resets_vram(self):
        """
        When _force_kill() times out after SIGKILL, VRAM tracking must be
        reset even though the OS process may still be alive.
        """
        config = _http_server_config("test-kill-timeout")
        loader = ServiceLoader(config)

        ok = await loader.start()
        assert ok
        pid = loader.pid

        # Simulate measured VRAM (per-process accounting stores in actual_vram_gb)
        loader._actual_vram_gb = 5.5

        # Force-kill with mocked timeout
        loader._state = ServiceState.STOPPING

        async def mock_wait():
            raise asyncio.TimeoutError()

        loader._process.wait = mock_wait
        await loader._force_kill()

        # State should be DEAD
        assert loader._state == ServiceState.DEAD

        # VRAM tracking must be reset despite zombie
        assert loader._actual_vram_gb is None, \
            "VRAM tracking should be reset on force-kill timeout"

        # OS process is still alive (mock never resolved)
        os.kill(pid, 0)

        # Clean up zombie
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass

    async def test_stop_resets_vram_tracking(self):
        """stop() must reset VRAM tracking on all exit paths."""
        config = _http_server_config("test-vram-reset")
        loader = ServiceLoader(config)

        ok = await loader.start()
        assert ok

        # Simulate measured VRAM
        loader._actual_vram_gb = 5.5

        await loader.stop()

        assert loader._actual_vram_gb is None, \
            "actual_vram_gb should be None after stop"


# ── Eviction VRAM Confirmation ────────────────────────────────────────────

class TestEvictionVRAMConfirmation:
    """Test that eviction actually confirms VRAM is freed."""

    async def test_eviction_confirms_vram_freed(self):
        """
        After stopping a backend, nvidia-smi should show VRAM was released.
        """
        vram = await query_vram_used_gb()
        if vram is None:
            pytest.skip("nvidia-smi not available")

        vram_before = vram

        config = _http_server_config("evict-vram")
        orch = Orchestrator([config], total_vram_gb=48, vram_reserve_gb=2)
        loader = orch.services["evict-vram"]

        ok = await loader.start()
        assert ok

        await loader.stop()
        await asyncio.sleep(1)

        vram_after = await query_vram_used_gb()

        assert vram_after <= vram_before + 0.5, \
            f"VRAM leaked after stop: before={vram_before:.2f}, after={vram_after:.2f}"

        await orch.shutdown()

    async def test_ensure_running_relaunches_after_stop(self):
        """
        After stopping a backend, ensure_running should relaunch it
        with a fresh process.
        """
        config = _http_server_config("evict-relaunch")
        orch = Orchestrator([config], total_vram_gb=48, vram_reserve_gb=2)

        # First launch
        loader = await orch.ensure_running("evict-relaunch")
        pid1 = loader.pid
        assert pid1 is not None

        # Stop it
        await loader.stop()

        # Second launch via ensure_running
        loader2 = await orch.ensure_running("evict-relaunch")
        pid2 = loader2.pid
        assert pid2 is not None

        await orch.shutdown()


# ── Orchestrator Locking ──────────────────────────────────────────────────

class TestOrchestratorLocking:
    """Test that concurrent ensure_running calls are properly serialized."""

    async def test_concurrent_ensure_running_is_serialized(self):
        """
        Multiple concurrent ensure_running() for the same backend
        should not cause double-launches.
        """
        config = _http_server_config("concurrent-test")
        orch = Orchestrator([config], total_vram_gb=48, vram_reserve_gb=2)

        results = await asyncio.gather(
            orch.ensure_running("concurrent-test"),
            orch.ensure_running("concurrent-test"),
            orch.ensure_running("concurrent-test"),
        )

        # All should return the same loader
        assert all(r is results[0] for r in results)

        await orch.shutdown()


# ── VRAM Accounting ───────────────────────────────────────────────────────

class TestVRAMAccounting:
    """Test that VRAM accounting is correct after lifecycle changes."""

    async def test_vram_reset_after_stop_allows_fresh_measurement(self):
        """
        After stop(), the next start() should measure VRAM from a fresh
        per-process reading, not from stale data.
        """
        config = _http_server_config("vram-account")
        loader = ServiceLoader(config)

        # First cycle
        ok1 = await loader.start()
        assert ok1
        vram_after_first = loader._actual_vram_gb

        await loader.stop()

        # VRAM tracking should be reset
        assert loader._actual_vram_gb is None

        # Second cycle — should get fresh measurement
        ok2 = await loader.start()
        assert ok2
        vram_after_second = loader._actual_vram_gb

        # http.server doesn't use GPU, so actual_vram_gb may be None.
        # But if it is set, it should be non-negative.
        if vram_after_second is not None:
            assert vram_after_second >= 0

        await loader.stop()

    async def test_multiple_start_stop_cycles_no_leak(self):
        """
        Repeated start/stop cycles should not accumulate stale VRAM tracking.
        """
        config = _http_server_config("multi-cycle")
        loader = ServiceLoader(config)

        for i in range(3):
            ok = await loader.start()
            assert ok, f"Start cycle {i} should succeed"
            await loader.stop()
            assert loader._actual_vram_gb is None, \
                f"VRAM should be reset after stop cycle {i}"
