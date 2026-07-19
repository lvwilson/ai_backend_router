#!/usr/bin/env python3
"""
watchdog.py — Keep the router alive.

Watches the router process and restarts it if it exits.
Also watches for config changes and triggers a graceful restart.

Usage:
    python watchdog.py [config.yaml]

The watchdog runs as a simple loop — suitable for running under tmux/screen
or as a systemd service.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

logger = logging.getLogger("watchdog")

SHUTDOWN_DELAY = 1.0       # Seconds between restart attempts
CONFIG_POLL_INTERVAL = 5.0 # Seconds between config change checks


async def run_router(config_path: str, stop: asyncio.Future) -> int:
    """
    Launch the router as a subprocess and wait for it to exit.

    If `stop` is set while the router is running, sends SIGTERM to the
    router's process group and waits for it to exit.

    Returns the exit code.
    """
    logger.info("Starting router with config=%s", config_path)
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "router.py", config_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        preexec_fn=os.setsid,
    )
    logger.info("Router PID=%d", proc.pid)

    # Forward stderr lines in real-time (router logs to stderr + file).
    # stdout must also be drained — an unread pipe eventually fills its
    # ~64 KB OS buffer and blocks the router process on write().
    async def tail_stream(stream, label):
        if stream:
            async for line in stream:
                text = line.decode(errors="replace").rstrip()
                if text:
                    logger.debug("router %s: %s", label, text)

    try:
        await asyncio.gather(proc.wait(), tail_stream(proc.stderr, "stderr"), tail_stream(proc.stdout, "stdout"))
    except asyncio.CancelledError:
        # Shutdown requested while waiting — terminate the router process group.
        logger.info("Shutting down router process group (PGID=%d)...", proc.pid)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        # Give it a moment to exit gracefully.
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.info("Router didn't exit in 5s, sending SIGKILL")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()

    logger.info("Router exited with code %d", proc.returncode)
    return proc.returncode or 0


async def wait_config_change(config: Path, mtime: float, stop: asyncio.Future) -> float | None:
    """
    Poll the config file until its mtime changes (returns the new mtime)
    or shutdown is requested (returns None).
    """
    while not stop.done():
        try:
            new_mtime = config.stat().st_mtime
            if new_mtime != mtime:
                return new_mtime
        except OSError:
            pass
        try:
            await asyncio.wait_for(asyncio.shield(asyncio.sleep(CONFIG_POLL_INTERVAL)), timeout=CONFIG_POLL_INTERVAL + 1.0)
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            return None
    return None


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = Path(config_path)
    if not config.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    # Track config mtime for change detection.
    config_mtime = config.stat().st_mtime

    loop = asyncio.get_event_loop()
    stop = loop.create_future()
    current_task: asyncio.Task | None = None

    def handle_signal(sig, frame):
        nonlocal current_task
        if not stop.done():
            stop.set_result(None)
            # Cancel the currently running router task so it can clean up.
            if current_task is not None and not current_task.done():
                current_task.cancel()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info("Watchdog started — watching %s", config_path)

    while not stop.done():
        # Run the router and a config-change watcher concurrently; whichever
        # finishes first ends the cycle.
        current_task = asyncio.ensure_future(run_router(config_path, stop))
        watcher = asyncio.ensure_future(wait_config_change(config, config_mtime, stop))
        try:
            done, pending = await asyncio.wait(
                {current_task, watcher}, return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            watcher.cancel()
            current_task.cancel()
            try:
                await current_task
            except asyncio.CancelledError:
                pass
            break

        if watcher in done and not watcher.cancelled():
            new_mtime = watcher.result()
            if new_mtime is not None:
                logger.info("Config file changed — restarting router")
                config_mtime = new_mtime
                # Terminate the running router; run_router's CancelledError
                # handler SIGTERMs the process group.
                current_task.cancel()
                try:
                    await current_task
                except asyncio.CancelledError:
                    pass
                current_task = None
                continue  # Relaunch immediately with the new config
        else:
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass

        current_task = None

        if stop.done():
            break

        # Router exited on its own — restart after a brief delay.
        logger.info("Restarting router in %.1fs...", SHUTDOWN_DELAY)
        try:
            await asyncio.sleep(SHUTDOWN_DELAY)
        except asyncio.CancelledError:
            break

    logger.info("Watchdog shutting down")


if __name__ == "__main__":
    asyncio.run(main())
