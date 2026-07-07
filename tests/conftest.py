"""
conftest.py — Shared pytest fixtures for the router test suite.

Provides:
  - GPU telemetry stubbing (session-scoped) so tests use deterministic
    tracked bookkeeping regardless of physical GPU availability.
  - --base-url CLI option for live tests.
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.orchestrator as orch_mod
import src.service_loader as sl_mod


def pytest_addoption(parser):
    """Register --base-url CLI option for live tests."""
    parser.addoption(
        "--base-url",
        action="store",
        default=None,
        help="Base URL for live router tests (default: http://127.0.0.1:8000)",
    )


# ── GPU telemetry stubs ───────────────────────────────────────────────────

async def _no_gpu(*args, **kwargs):
    return None

async def _no_per_process_vram(*args, **kwargs):
    return {}


@pytest.fixture(scope="session", autouse=True)
def stub_gpu_telemetry():
    """
    Stub nvidia-smi GPU telemetry for the entire test session.

    This ensures all tests use deterministic tracked bookkeeping
    (per-process VRAM from ServiceLoader) rather than live GPU readings,
    making tests pass on any machine.
    """
    orig_orch_query = orch_mod.query_vram_used_gb
    orig_sl_query = sl_mod.query_vram_used_gb
    orig_orch_per_pid = orch_mod.query_per_process_vram
    orig_sl_per_pid = sl_mod.query_per_process_vram

    orch_mod.query_vram_used_gb = _no_gpu
    sl_mod.query_vram_used_gb = _no_gpu
    orch_mod.query_per_process_vram = _no_per_process_vram
    sl_mod.query_per_process_vram = _no_per_process_vram

    yield

    orch_mod.query_vram_used_gb = orig_orch_query
    sl_mod.query_vram_used_gb = orig_sl_query
    orch_mod.query_per_process_vram = orig_orch_per_pid
    sl_mod.query_per_process_vram = orig_sl_per_pid
