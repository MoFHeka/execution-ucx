"""RDT fetch futures carry results and retain storage until native completion."""

import asyncio
from contextlib import asynccontextmanager
import gc
import threading
from typing import List
import weakref

import axon
import numpy as np
import pytest


_TIMEOUT = 10
_FUNCTION_ID = 19


def _launch(client, object_id="object", tensor_index=0, **kwargs):
    future = client._async_fetch(
        object_id,
        tensor_index,
        "fetch_server",
        kwargs.pop("function_id", _FUNCTION_ID),
        27,
        from_dlpack=kwargs.pop("from_dlpack", np.from_dlpack),
        **kwargs,
    )
    assert isinstance(future, asyncio.Future)
    assert future.get_loop() is asyncio.get_running_loop()
    return future


async def _finish(future):
    # A timeout must not cancel our only native-completion signal.
    return await asyncio.wait_for(asyncio.shield(future), _TIMEOUT)


@asynccontextmanager
async def _connected(fetch):
    server = axon.AxonRuntime("fetch_server")
    client = axon.AxonRuntime("fetch_client")
    server.start()
    client.start_client()
    try:
        server.register_function(fetch, _FUNCTION_ID)
        await client.connect_endpoint_async(server.get_local_address(), "fetch_server")
        yield client
    finally:
        client.stop()
        server.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shape,as_list",
    [
        ((0,), False),
        ((1,), False),
        ((32768,), False),
        ((), False),
        ((0,), True),
        ((32768,), True),
        ((), True),
    ],
)
async def test_async_fetch_result_and_owned_conversion_lifetime(shape, as_list):
    expected = np.arange(max(1, int(np.prod(shape))), dtype=np.int32)
    expected = expected[: int(np.prod(shape))].reshape(shape)
    extra = np.array([19], dtype=np.uint8)

    async def fetch(object_id: str, tensor_index: int) -> List[np.ndarray]:
        assert object_id == "object"
        assert tensor_index == 4
        return [expected, extra] if as_list else expected

    if not as_list:
        fetch.__annotations__["return"] = np.ndarray

    def owned_from_dlpack(tensor):
        # Eager replies can borrow UCX storage. Make the conversion own its
        # storage when retaining the result after the runtime is stopped.
        return np.from_dlpack(tensor).copy()

    async with _connected(fetch) as client:
        future = client._async_fetch(
            object_id="object",
            tensor_index=4,
            worker_name="fetch_server",
            function_id=_FUNCTION_ID,
            workflow_id=27,
            from_dlpack=owned_from_dlpack,
        )
        result = await _finish(future)
        assert await future is result
        tensors = result if as_list else [result]
        assert len(tensors) == (2 if as_list else 1)
        assert tensors[0].shape == shape
        assert tensors[0].dtype == expected.dtype
        np.testing.assert_array_equal(tensors[0], expected)
        if as_list:
            np.testing.assert_array_equal(tensors[1], extra)

    # Framework-owned conversion results outlive both runtimes.
    np.testing.assert_array_equal(tensors[0], expected)
    if as_list:
        np.testing.assert_array_equal(tensors[1], extra)


@pytest.mark.asyncio
async def test_async_fetch_launches_all_tensors_before_first_completion():
    release = asyncio.Event()
    all_entered = asyncio.Event()
    entered = set()
    count = 4

    async def fetch(object_id: str, tensor_index: int) -> np.ndarray:
        assert object_id == "batch"
        entered.add(tensor_index)
        if len(entered) == count:
            all_entered.set()
        await release.wait()
        return np.full(32768, tensor_index, dtype=np.int32)

    async with _connected(fetch) as client:
        pending = []
        try:
            for index in range(count):
                # These futures retain their results after runtime shutdown.
                # Eager replies can borrow UCX storage, so take an owned copy.
                pending.append(
                    _launch(
                        client,
                        "batch",
                        index,
                        from_dlpack=lambda tensor: np.from_dlpack(tensor).copy(),
                    )
                )
            await asyncio.wait_for(all_entered.wait(), _TIMEOUT)
            assert entered == set(range(count))
            assert all(not future.done() for future in pending)
        finally:
            release.set()
            results = await asyncio.gather(*(_finish(future) for future in pending))
        for index, result in enumerate(results):
            np.testing.assert_array_equal(result, np.full(32768, index, dtype=np.int32))


@pytest.mark.asyncio
@pytest.mark.parametrize("function_id", [_FUNCTION_ID, 999999])
async def test_async_fetch_remote_error_signals_completion(function_id):
    async def fetch(object_id: str, tensor_index: int) -> np.ndarray:
        raise KeyError("missing tensor")

    async with _connected(fetch) as client:
        future = _launch(client, function_id=function_id)
        with pytest.raises(RuntimeError):
            await _finish(future)
        assert future.done() and not future.cancelled()


@pytest.mark.asyncio
async def test_async_fetch_conversion_error_signals_completion():
    async def fetch(object_id: str, tensor_index: int) -> np.ndarray:
        return np.arange(4, dtype=np.int32)

    def fail_conversion(_tensor):
        raise ValueError("injected DLPack conversion failure")

    async with _connected(fetch) as client:
        future = _launch(client, from_dlpack=fail_conversion)
        with pytest.raises(
            (ValueError, RuntimeError), match="injected DLPack conversion failure"
        ):
            await _finish(future)
        assert future.done() and not future.cancelled()


@pytest.mark.asyncio
async def test_async_fetch_uses_preallocated_large_response():
    expected = np.arange(1024 * 1024, dtype=np.float32)
    destination = np.empty_like(expected)
    policy_calls = []

    async def fetch(object_id: str, tensor_index: int) -> np.ndarray:
        return expected

    def memory_policy(metadata):
        assert len(metadata) == 1
        assert tuple(metadata[0].shape) == expected.shape
        policy_calls.append(len(metadata))
        return destination

    async with _connected(fetch) as client:
        future = _launch(client, memory_policy=memory_policy)
        result = await _finish(future)
        assert policy_calls == [1]
        assert result is destination
        np.testing.assert_array_equal(destination, expected)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", ["cancel", "timeout"])
async def test_async_fetch_shielded_wait_preserves_native_completion(interrupt):
    entered = asyncio.Event()
    release = asyncio.Event()
    expected = np.arange(1024 * 1024, dtype=np.float32)
    destination = np.full_like(expected, -1)
    policy_calls = []

    async def fetch(object_id: str, tensor_index: int) -> np.ndarray:
        entered.set()
        await release.wait()
        return expected

    def memory_policy(metadata):
        policy_calls.append(len(metadata))
        return destination

    async with _connected(fetch) as client:
        future = _launch(client, memory_policy=memory_policy)
        try:
            await asyncio.wait_for(entered.wait(), _TIMEOUT)
            if interrupt == "cancel":
                waiter = asyncio.create_task(_finish(future))
                await asyncio.sleep(0)
                waiter.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await waiter
            else:
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(future), timeout=0)
            assert not future.done()
            assert not policy_calls
            assert np.all(destination == -1)
        finally:
            release.set()
            result = await _finish(future)
        assert not future.cancelled()
        assert result is destination
        assert policy_calls == [1]
        np.testing.assert_array_equal(destination, expected)


@pytest.mark.asyncio
async def test_async_fetch_conversion_and_completion_run_on_shared_loop():
    loop = asyncio.get_running_loop()
    thread = threading.get_ident()
    conversions = []
    completions = []

    async def fetch(object_id: str, tensor_index: int) -> np.ndarray:
        return np.array([1], dtype=np.int32)

    def convert(tensor):
        conversions.append((asyncio.get_running_loop(), threading.get_ident()))
        return np.from_dlpack(tensor).copy()

    def completed(future):
        completions.append(
            (asyncio.get_running_loop(), threading.get_ident(), future.done())
        )

    async with _connected(fetch) as client:
        future = _launch(client, from_dlpack=convert)
        future.add_done_callback(completed)
        result = await _finish(future)
        np.testing.assert_array_equal(result, [1])
        assert conversions == [(loop, thread)]
        assert completions == [(loop, thread, True)]


@pytest.mark.asyncio
async def test_async_fetch_requires_running_loop():
    async def fetch(object_id: str, tensor_index: int) -> np.ndarray:
        return np.array([1], dtype=np.int32)

    async with _connected(fetch) as client:

        def launch_without_running_loop():
            with pytest.raises(RuntimeError, match="loop"):
                client._async_fetch(
                    "object",
                    0,
                    "fetch_server",
                    _FUNCTION_ID,
                    27,
                    from_dlpack=np.from_dlpack,
                )

        await asyncio.to_thread(launch_without_running_loop)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["never_started", "stopped", "client_stopped"])
async def test_async_fetch_rejects_inactive_client_with_live_loop(state):
    async def fetch(object_id: str, tensor_index: int) -> np.ndarray:
        return np.array([1], dtype=np.int32)

    # The connected pair keeps the shared native reader active throughout.
    async with _connected(fetch):
        inactive = axon.AxonRuntime("inactive_fetch_client")
        if state != "never_started":
            inactive.start_client()
            if state == "client_stopped":
                inactive.stop_client()
            else:
                inactive.stop()
        with pytest.raises(RuntimeError, match="running|started|stopped|client"):
            inactive._async_fetch(
                "object",
                0,
                "fetch_server",
                _FUNCTION_ID,
                27,
                from_dlpack=np.from_dlpack,
            )


@pytest.mark.asyncio
async def test_async_fetch_retains_runtime_until_native_completion():
    entered = asyncio.Event()
    release = asyncio.Event()
    # Match the large-response tests: small eager replies bypass memory_policy.
    expected = np.arange(1024 * 1024, dtype=np.int32)
    destination = np.empty_like(expected)

    async def fetch(object_id: str, tensor_index: int) -> np.ndarray:
        entered.set()
        await release.wait()
        return expected

    server = axon.AxonRuntime("fetch_server")
    client = axon.AxonRuntime("fetch_client")
    client_ref = weakref.ref(client)
    server.start()
    client.start_client()
    future = None
    try:
        server.register_function(fetch, _FUNCTION_ID)
        await client.connect_endpoint_async(server.get_local_address(), "fetch_server")
        future = _launch(client, memory_policy=lambda _: destination)
        await asyncio.wait_for(entered.wait(), _TIMEOUT)
        del client
        gc.collect()
        assert client_ref() is not None
    finally:
        release.set()
        if future is not None:
            result = await _finish(future)
        remaining_client = client_ref()
        if remaining_client is not None:
            remaining_client.stop()
        server.stop()
    assert result is destination
    np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
