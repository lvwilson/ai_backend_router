#!/usr/bin/env python3
"""
demo_service_loader.py — Standalone demo of the ServiceLoader component.

Shows how to configure and manage a backend process lifecycle.
Run with: python demo_service_loader.py
"""

import asyncio
import json
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.service_loader import ServiceConfig, ServiceLoader, ServiceState, query_vram_used_gb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def on_event(name: str, payload: dict) -> None:
    """Example event callback."""
    logger.info("  → Event '%s': %s", name, json.dumps(payload, default=str))


async def demo_config():
    """Show a sample configuration for a llama.cpp backend."""
    config = ServiceConfig(
        name="llama-demo",
        binary="echo",        # Use 'echo' as a simple test binary
        args=["Hello from llama.cpp"],
        port=None,            # No health endpoint for this demo
        expected_vram_gb=4.0,
        expected_ram_gb=2.0,
        health_timeout=5.0,
    )
    print("\n=== ServiceConfig ===")
    print(f"  name:           {config.name}")
    print(f"  binary:         {config.binary}")
    print(f"  args:           {config.args}")
    print(f"  expected_vram:  {config.expected_vram_gb} GB")
    print(f"  expected_ram:   {config.expected_ram_gb} GB")
    return config


async def demo_lifecycle():
    """Demonstrate start → status → stop lifecycle with a real process."""
    config = ServiceConfig(
        name="sleep-demo",
        binary="sleep",
        args=["60"],          # Sleep for 60 seconds — enough for demo
        port=None,
        expected_vram_gb=0.0,
        expected_ram_gb=0.1,
        stop_timeout=3.0,
    )

    loader = ServiceLoader(config, event_callback=on_event)

    print("\n=== Starting service ===")
    ok = await loader.start()
    print(f"  start() returned: {ok}")

    print("\n=== Status ===")
    status = await loader.get_status()
    print(f"  {json.dumps(status, indent=2, default=str)}")

    print(f"\n  is_alive: {loader.is_alive}")
    print(f"  state:    {loader.state.value}")
    print(f"  pid:      {loader.pid}")

    print("\n=== Stopping service (graceful) ===")
    stopped = await loader.stop()
    print(f"  stop() returned: {stopped}")

    print(f"\n  is_alive: {loader.is_alive}")
    print(f"  state:    {loader.state.value}")


async def demo_vram_query():
    """Show VRAM query helper."""
    print("\n=== VRAM Query ===")
    vram = await query_vram_used_gb()
    if vram is not None:
        print(f"  VRAM used: {vram:.2f} GB")
    else:
        print("  nvidia-smi not available (or no NVIDIA GPU)")


async def demo_context_manager():
    """Show async context manager usage."""
    config = ServiceConfig(
        name="ctx-demo",
        binary="sleep",
        args=["30"],
        port=None,
        expected_vram_gb=0.0,
        stop_timeout=2.0,
    )

    print("\n=== Context Manager Demo ===")
    async with ServiceLoader(config, event_callback=on_event) as loader:
        print(f"  Inside context: state={loader.state.value}, pid={loader.pid}")
        await asyncio.sleep(0.5)
    print(f"  After context:  state={loader.state.value}")


async def main():
    print("=" * 60)
    print("ServiceLoader Component Demo")
    print("=" * 60)

    await demo_config()
    await demo_lifecycle()
    await demo_vram_query()
    await demo_context_manager()

    print("\n" + "=" * 60)
    print("Demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
