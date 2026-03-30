"""Internal utility for safe sync-from-async bridging.

Library code should be async-first. These helpers exist ONLY for
``_sync()`` convenience wrappers. The CLI is the only place where
``asyncio.run()`` should appear directly.
"""

from __future__ import annotations

import asyncio
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from synchronous code.

    Uses ``asyncio.run()`` when no event loop is running.
    Falls back to ``anyio.from_thread.run()`` when called from
    within an existing loop (e.g. Jupyter, nested sync wrappers).

    Library internals should NEVER call this — use ``await`` instead.
    This is only for public ``_sync()`` convenience methods.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop running — safe to use asyncio.run()
        return asyncio.run(coro)

    # Already inside an async context — use anyio thread bridge
    import anyio.from_thread

    return anyio.from_thread.run(coro)  # type: ignore[arg-type]
