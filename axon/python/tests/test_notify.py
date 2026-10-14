"""One-way RPC coverage matching the invoke argument and NumPy tests."""

import asyncio
import gc
import sys
from typing import List

import numpy as np
import pytest
import pytest_asyncio

import axon


@pytest_asyncio.fixture
async def runtimes():
    server = axon.AxonRuntime("notify_server", timeout=5000)
    client = axon.AxonRuntime("notify_client", timeout=5000)
    try:
        server.start_server()
        client.start_client()
        await asyncio.wait_for(
            client.connect_endpoint_async(server.get_local_address(), "notify_server"),
            timeout=5,
        )
        yield server, client
    finally:
        client.stop()
        server.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("by_name", [False, True], ids=["function-id", "function-name"])
async def test_notify_returns_none_before_handler_completes(runtimes, by_name):
    server, client = runtimes
    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    received = []
    completed = []

    async def record(x: int, y: int, label: str) -> int:
        received.append((x, y, label))
        entered.set()
        await release.wait()
        completed.append(x + y)
        finished.set()
        return x + y

    async def read_total() -> int:
        return sum(completed)

    server.register_function(
        record, function_id=None if by_name else 1, function_name="record"
    )
    server.register_function(read_total, function_id=2)

    try:
        # Unlike invoke(), notify() returns no awaitable or remote return value.
        result = client.notify(
            10,
            20,
            "hello",
            worker_name="notify_server",
            session_id=1,
            function="record" if by_name else 1,
            workflow_id=7,
        )
        assert result is None
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert received == [(10, 20, "hello")]
        assert not finished.is_set()

        release.set()
        await asyncio.wait_for(finished.wait(), timeout=5)

        # A normal RPC on the same connection must still receive its response.
        total = await asyncio.wait_for(
            client.invoke(worker_name="notify_server", session_id=1, function=2),
            timeout=5,
        )
        assert total == 30
    finally:
        # Do not leave a suspended server handler behind if an assertion fails.
        release.set()
        if entered.is_set() and not finished.is_set():
            await asyncio.wait_for(finished.wait(), timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("tensor_count", [1, 2], ids=["single-tensor", "tensor-list"])
@pytest.mark.parametrize("elements", [8, 32768], ids=["small", "large"])
async def test_notify_tensor_payload_survives_caller_release(
    runtimes, tensor_count, elements
):
    server, client = runtimes
    received = asyncio.get_running_loop().create_future()

    async def record_tensor(value: np.ndarray) -> int:
        received.set_result([value.copy()])
        return 1

    async def record_tensors(values: List[np.ndarray]) -> int:
        received.set_result([value.copy() for value in values])
        return len(values)

    server.register_function(
        record_tensor if tensor_count == 1 else record_tensors,
        function_id=1,
        from_dlpack_fn=np.from_dlpack,
    )
    expected = [np.arange(elements, dtype=np.int32) + i for i in range(tensor_count)]
    payload = expected[0].copy() if tensor_count == 1 else [a.copy() for a in expected]

    result = client.notify(
        payload, worker_name="notify_server", session_id=0, function=1
    )
    assert result is None
    # There is no response future to keep the caller's input arrays alive.
    del payload
    gc.collect()

    actual = await asyncio.wait_for(received, timeout=5)
    assert len(actual) == tensor_count
    for actual_array, expected_array in zip(actual, expected):
        np.testing.assert_array_equal(actual_array, expected_array)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", *sys.argv[1:]]))
