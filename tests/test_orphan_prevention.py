#!/usr/bin/env python3
"""
test_orphan_prevention.py — Backend processes must never outlive the router.

Covers the two independent backstops added so a router exit can never leave
an orphaned backend holding VRAM:

  1. atexit orphan sweeper (router.py): on interpreter exit, any backend
     process group still alive is SIGKILLed. Verified by running a throwaway
     child process that registers the sweeper, attaches a live `sleep`
     backend, and exits the interpreter — the backend must be killed.

  2. PR_SET_PDEATHSIG (service_loader.py): the kernel SIGKILLs a backend if
     its parent (the router) dies abruptly (SIGKILL/OOM/segfault), with no
     chance to run cleanup. Simulated with a throwaway parent that spawns a
     ServiceLoader-managed `sleep` and then SIGKILLs itself.

The pdeathsig test skips gracefully where it can't be observed (e.g. a
pid-1 container, where orphaned children are reaped by init too quickly).
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import router  # noqa: F401  (import proves the sweeper module loads)
from src.orchestrator import Orchestrator  # noqa: F401
from src.service_loader import ServiceConfig  # noqa: F401


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # Exists but owned by someone else


def _wait_dead(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.1)
    return False


def _zombies_in_env() -> bool:
    """True if some process is parented to pid 1 (pid-1 container)."""
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                for line in (p / "status").read_text().splitlines():
                    if line.startswith("PPid:") and line.split()[1] == "1":
                        return True
            except Exception:
                continue
    return False


# Child script: register the sweeper on a real orchestrator, attach a live
# `sleep` backend, print its PID, then exit the interpreter via SystemExit
# (which runs atexit handlers). The sweeper must SIGKILL the backend.
_SWEEP_CHILD_SCRIPT = (
    "import subprocess, sys\n"
    f"sys.path.insert(0, {str(ROOT)!r})\n"
    "import router\n"
    "from src.orchestrator import Orchestrator\n"
    "from src.service_loader import ServiceConfig\n"
    "cfg = ServiceConfig(name='sleeper', binary='sleep', args=['300'])\n"
    "orch = Orchestrator([cfg], total_vram_gb=10, vram_reserve_gb=1)\n"
    "router._install_orphan_sweeper(orch)\n"
    "proc = subprocess.Popen(['sleep', '300'], start_new_session=True)\n"
    "orch.services['sleeper']._process = proc\n"
    "print(proc.pid, flush=True)\n"
    "raise SystemExit(0)\n"
)


class TestOrphanSweeper:
    def test_atexit_sweep_kills_running_backends(self):
        """
        A backend still alive when the router interpreter exits must be
        SIGKILLed by the registered atexit sweeper.

        Runs in a throwaway child process: atexit handlers are per-process,
        so the sweeper must be registered and triggered in the same child
        that owns the backend.
        """
        child = subprocess.Popen(
            [sys.executable, "-c", _SWEEP_CHILD_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            backend_pid = int(child.stdout.readline().strip())
        finally:
            child.wait(timeout=15)
        assert child.returncode == 0, f"sweeper child exited rc={child.returncode}"

        assert _wait_dead(backend_pid, timeout=10), (
            f"atexit sweeper did not kill backend PID {backend_pid}"
        )


@pytest.mark.skipif(
    _zombies_in_env(),
    reason="pid-1 container: orphaned children are reaped by init before we can observe pdeathsig",
)
class TestPdeathsig:
    def test_backend_dies_when_router_is_sigkilled(self):
        """
        If the router process is SIGKILLed (no cleanup possible), a backend
        spawned through ServiceLoader must die via PR_SET_PDEATHSIG rather
        than leak as an orphan.
        """
        parent_script = (
            "import asyncio, os, signal, sys, time\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from src.orchestrator import Orchestrator\n"
            "from src.service_loader import ServiceConfig, _child_preexec\n"
            "cfg = ServiceConfig(name='sleeper', binary='sleep', args=['300'])\n"
            "orch = Orchestrator([cfg], total_vram_gb=10, vram_reserve_gb=1)\n"
            "async def go():\n"
            "    loader = orch.services['sleeper']\n"
            "    loader._process = await asyncio.create_subprocess_exec(\n"
            "        'sleep', '300',\n"
            "        stdout=asyncio.subprocess.DEVNULL,\n"
            "        stderr=asyncio.subprocess.DEVNULL,\n"
            "        preexec_fn=_child_preexec,\n"
            "    )\n"
            "    print(loader._process.pid, flush=True)\n"
            "    time.sleep(0.3)\n"
            "    os.kill(os.getpid(), signal.SIGKILL)\n"
            "asyncio.run(go())\n"
        )
        parent = subprocess.Popen(
            [sys.executable, "-c", parent_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            child_pid = int(parent.stdout.readline().strip())
        finally:
            parent.wait(timeout=15)
        assert parent.returncode == -signal.SIGKILL, (
            f"parent should have SIGKILLed itself, rc={parent.returncode}"
        )

        assert _wait_dead(child_pid, timeout=10), (
            f"backend PID {child_pid} survived router SIGKILL — orphaned process leak"
        )
