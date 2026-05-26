"""Axon Future implementation for high-performance active message completion.

This module provides `AxonFuture`, a specialized awaitable that wraps `asyncio.Future`.
It is designed to bypass the standard `epoll` / `eventfd` wake-up mechanism of `asyncio`
when the underlying C++ background task completes extremely fast.

By synchronously polling the lock-free completion queue (`_process_wake_queue`)
on the main Python thread before yielding to the asyncio event loop, this future
can achieve "Zero-Yield" completions. This drastically improves throughput and
reduces latency by bypassing context switches into the kernel for eventfd wakeups.
"""

import asyncio
import os
from typing import Any, Generic, TypeVar

# We will import the C++ extension module inside to avoid circular imports
_process_wake_queue_func = None


def _get_process_wake_queue():
    global _process_wake_queue_func
    if _process_wake_queue_func is None:
        try:
            from ._axon import _process_wake_queue  # type: ignore

            _process_wake_queue_func = _process_wake_queue
        except ImportError:
            # Fallback for testing/mocking if C++ module is unavailable
            _process_wake_queue_func = lambda: 0
    return _process_wake_queue_func


T = TypeVar("T")

AXON_DISABLE_FAST_PATH = os.environ.get("AXON_DISABLE_FAST_PATH", "0") == "1"


class AxonFuture(Generic[T]):
    """
    A high-performance awaitable that bypasses eventfd wakeups when possible.
    It synchronouly probes the C++ completion queue before yielding to the asyncio event loop.
    """

    def __init__(self, asyncio_future: asyncio.Future | None = None):
        if asyncio_future is None:
            self._future = asyncio.get_running_loop().create_future()
        else:
            self._future = asyncio_future

    def __await__(self):
        # Fast Path: drain the lock-free queue synchronously
        if not AXON_DISABLE_FAST_PATH:
            _process_wake_queue = _get_process_wake_queue()
            _process_wake_queue()

            # If the C++ worker already finished our task, _process_wake_queue
            # will have executed the callback that sets self._future.result()!
            if self._future.done():
                # Zero-yield fast path!
                return self._future.result()

        # Slow Path: yield to asyncio event loop to wait for eventfd wakeup
        return (yield from self._future.__await__())

    def get_asyncio_future(self) -> asyncio.Future:
        return self._future

    def done(self) -> bool:
        return self._future.done()

    def result(self) -> Any:
        return self._future.result()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._future, name)
