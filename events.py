"""
events.py — Lifecycle event system for backend services.

Provides a minimal pub/sub mechanism for service lifecycle events.
Designed to be extended (multiple subscribers, filtering) without
changing the ServiceLoader contract.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


EventCallback = Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]]


@dataclass
class ServiceEvent:
    """Event emitted during a service lifecycle transition."""
    name: str            # "started", "stopped", "killed", "unhealthy", "resource_warning"
    payload: dict[str, Any] = field(default_factory=dict)


class EventDispatcher:
    """
    Dispatches lifecycle events to registered callbacks.

    Usage:
        dispatcher = EventDispatcher(on_event)
        await dispatcher.emit("started", {"pid": 1234})
    """

    def __init__(
        self,
        callback: EventCallback | None = None,
        service_name: str = "",
    ):
        self._callback = callback
        self._service_name = service_name

    async def emit(self, name: str, payload: dict[str, Any] | None = None) -> None:
        """Emit a lifecycle event to the registered callback."""
        if self._callback:
            try:
                await self._callback(name, payload or {})
            except Exception as exc:
                logger.debug(
                    "[%s] Event callback error for '%s': %s",
                    self._service_name,
                    name,
                    exc,
                )
