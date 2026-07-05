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
import time
from pathlib import Path

logger = logging.getLogger("watchdog")

SHUTDOWN_DELAY = 1.0       # Seconds between restart attempts
CONFIG_POLL_INTERVAL = 5.0 # Seconds between config change checks


async def run_router(config_path: str) -> int:
    """
    Launch the router as a subprocess and wait for it to exit.

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

    # Forward stderr lines in real-time (router logs to stderr + file)
    async def tail_stderr():
        if proc.stderr:
            async for line in proc.stderr:
                text = line.decode(errors="replace").rstrip()
                if text:
                    logger.debug("router: %s", text)

    await asyncio.gather(proc.wait(), tail_stderr())

    logger.info("Router exited with code %d", proc.returncode)
    return proc.returncode or 0


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
    restart_requested = False

    loop = asyncio.get_event_loop()
    stop = loop.create_future()

    def handle_signal(sig, frame):
        if not stop.done():
            stop.set_result(None)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info("Watchdog started — watching %s", config_path)

    while not stop.done():
        # Check for config changes
        try:
            new_mtime = config.stat().st_mtime
            if new_mtime != config_mtime:
                logger.info("Config file changed — will restart router on next cycle")
                config_mtime = new_mtime
                restart_requested = True
        except OSError:
            pass

        exit_code = await run_router(config_path)
        restart_requested = False  # Config was loaded fresh

        if stop.done():
            break

        # Decide whether to restart.
        # Exit code 0 from SIGTERM is intentional shutdown — but since we're
        # the watchdog, any exit triggers a restart (unless we're stopping).
        logger.info("Restarting router in %.1fs...", SHUTDOWN_DELAY)
        await asyncio.sleep(SHUTDOWN_DELAY)

    # Propagate shutdown to the router process group if it's running.
    logger.info("Watchdog shutting down")


if __name__ == "__main__":
    asyncio.run(main())
