"""Linux PID guards for external memory users; never signal a numeric PID.

Registrations are in-memory leases. Timeout releases admission blocking, not
ownership: the process remains listed/protected until exit or unregister.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import select
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class ProcessGuardError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class _Guard:
    pid: int
    fd: int
    name: str
    registered_at: float
    timeout: float | None
    deadline: float | None
    state: str = "active"
    done: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None

    def snapshot(self) -> dict:
        return {
            "pid": self.pid,
            "name": self.name,
            "state": self.state,
            "blocking": self.state in ("active", "killing"),
            "registered_at": self.registered_at,
            "timeout": self.timeout,
            "expires_at": (
                self.registered_at + self.timeout if self.timeout is not None else None
            ),
        }


class ProcessGuards:
    POLL_INTERVAL = 0.1

    def __init__(self, condition: asyncio.Condition, managed_pids: Callable[[], set[int]]):
        self._condition = condition
        self._managed_pids = managed_pids
        self._guards: dict[int, _Guard] = {}

    @property
    def blocking(self) -> bool:
        return any(g.state in ("active", "killing") for g in self._guards.values())

    @property
    def pids(self) -> set[int]:
        return set(self._guards)

    def snapshots(self) -> list[dict]:
        return [g.snapshot() for _, g in sorted(self._guards.items())]

    @staticmethod
    def _exited(guard: _Guard) -> bool:
        # pidfds become readable on exit, including unreaped zombies.
        return bool(select.select([guard.fd], [], [], 0)[0])

    @staticmethod
    def _ancestors(pid: int) -> set[int]:
        ancestors = set()
        while pid > 1 and pid not in ancestors:
            ancestors.add(pid)
            try:
                stat = Path(f"/proc/{pid}/stat").read_text()
                pid = int(stat.rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                break
        return ancestors

    def register(self, pid: int, timeout: float | None = None) -> dict:
        if type(pid) is not int or pid <= 1:
            raise ProcessGuardError(400, "pid must be an integer greater than 1")
        if timeout is not None and (
            isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0
        ):
            raise ProcessGuardError(400, "timeout must be a finite positive number of seconds")
        if pid in self._guards:
            raise ProcessGuardError(409, f"PID {pid} is already registered")
        if pid in self._ancestors(os.getpid()):
            raise ProcessGuardError(400, "Cannot guard the router or its ancestors")
        if self._ancestors(pid) & self._managed_pids():
            raise ProcessGuardError(400, "Cannot guard a router-managed backend")
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise ProcessGuardError(503, "Process guards require Linux pidfd support")

        fd = None
        try:
            fd = os.pidfd_open(pid)
            proc = Path(f"/proc/{pid}")
            if proc.stat().st_uid != os.geteuid():
                raise ProcessGuardError(403, "Process must belong to the router's OS user")
            name = (proc / "comm").read_text().strip()
            guard = _Guard(
                pid, fd, name, time.time(), timeout,
                time.monotonic() + timeout if timeout is not None else None,
            )
            if self._exited(guard):
                raise ProcessGuardError(404, f"PID {pid} has already exited")
            # Verify signal permission without changing the process.
            signal.pidfd_send_signal(fd, 0)
        except (ProcessLookupError, FileNotFoundError):
            if fd is not None:
                os.close(fd)
            raise ProcessGuardError(404, f"PID {pid} is not running") from None
        except PermissionError:
            if fd is not None:
                os.close(fd)
            raise ProcessGuardError(403, f"No permission to guard PID {pid}") from None
        except BaseException as exc:
            if fd is not None:
                os.close(fd)
            if isinstance(exc, OSError):
                raise ProcessGuardError(503, f"Cannot open process handle: {exc}") from exc
            raise

        # No await between validation and installation: admission sees this
        # atomically, including when an unload is currently draining work.
        self._guards[pid] = guard
        guard.task = asyncio.create_task(self._watch(guard))
        logger.info("Registered process guard PID %d (%s), timeout=%s", pid, name, timeout)
        return guard.snapshot()

    async def _notify(self) -> None:
        async with self._condition:
            self._condition.notify_all()

    def _remove(self, guard: _Guard) -> None:
        if self._guards.get(guard.pid) is guard:
            del self._guards[guard.pid]
            os.close(guard.fd)
            guard.done.set()

    async def _watch(self, guard: _Guard) -> None:
        while True:
            if self._exited(guard):
                self._remove(guard)
                await self._notify()
                logger.info("Process guard PID %d exited; registration removed", guard.pid)
                return
            if guard.state == "active" and guard.deadline is not None:
                if time.monotonic() >= guard.deadline:
                    guard.state = "timed_out"
                    await self._notify()
                    logger.info("Process guard PID %d timed out (not killed)", guard.pid)
            await asyncio.sleep(self.POLL_INTERVAL)

    async def unregister(self, pid: int) -> dict:
        guard = self._guards.get(pid)
        if guard is None:
            raise ProcessGuardError(404, f"PID {pid} is not registered")
        if guard.state == "killing":
            raise ProcessGuardError(409, f"PID {pid} is being killed; wait for exit")
        # Remove before the first await: cancellation or a concurrent kill
        # must not leave a registered process with a cancelled watcher.
        guard.task.cancel()
        self._remove(guard)
        await self._notify()
        await asyncio.gather(guard.task, return_exceptions=True)
        return {"pid": pid, "unregistered": True, "killed": False}

    async def kill(self, pid: int) -> dict:
        guard = self._guards.get(pid)
        if guard is None:
            raise ProcessGuardError(404, f"PID {pid} is not registered")
        if self._exited(guard):
            raise ProcessGuardError(404, f"Registered PID {pid} has already exited")
        try:
            # Signal ONLY the pinned process, never its group or a reused PID.
            signal.pidfd_send_signal(guard.fd, signal.SIGKILL)
        except ProcessLookupError:
            raise ProcessGuardError(404, f"Registered PID {pid} has already exited") from None
        except PermissionError:
            raise ProcessGuardError(403, f"No permission to kill PID {pid}") from None
        guard.state = "killing"
        # Keep admission blocked until exit is confirmed, even if timeout was
        # reached before this explicit kill operation.
        try:
            await asyncio.wait_for(guard.done.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            raise ProcessGuardError(504, f"PID {pid} has not exited; guard remains blocking") from None
        return {"pid": pid, "killed": True}

    def protects_group(self, pid: int) -> bool:
        """Prevent the legacy hog killer from killing a registered group member."""
        if pid in self._guards:
            return True
        try:
            group = os.getpgid(pid)
        except ProcessLookupError:
            return False
        for protected_pid in self._guards:
            try:
                if os.getpgid(protected_pid) == group:
                    return True
            except ProcessLookupError:
                pass
        return False

    async def close(self) -> None:
        """Router shutdown drops registrations; it never kills external users."""
        guards = list(self._guards.values())
        for guard in guards:
            guard.task.cancel()
        await asyncio.gather(*(g.task for g in guards), return_exceptions=True)
        for guard in guards:
            self._remove(guard)
        await self._notify()
