import asyncio
import pytest

from axon.future import AxonFuture


@pytest.mark.asyncio
async def test_axon_future_fast_path(monkeypatch):
    # Mock the fast path queue processing
    queue_processed = False

    def mock_process_wake_queue():
        nonlocal queue_processed
        queue_processed = True

    monkeypatch.setattr(
        "axon.future._get_process_wake_queue", lambda: mock_process_wake_queue
    )
    monkeypatch.setattr("axon.future.AXON_DISABLE_FAST_PATH", False)

    # Create an asyncio future and immediately set its result
    # This simulates a C++ task that finished instantly.
    loop = asyncio.get_running_loop()
    inner_fut = loop.create_future()
    inner_fut.set_result("fast_result")

    # Create the AxonFuture wrapping it
    axon_fut = AxonFuture(inner_fut)

    # Await it. Since it's done, it should bypass yielding.
    result = await axon_fut

    assert queue_processed, "Fast path should process the queue synchronously"
    assert result == "fast_result"


@pytest.mark.asyncio
async def test_axon_future_slow_path(monkeypatch):
    queue_processed = False

    def mock_process_wake_queue():
        nonlocal queue_processed
        queue_processed = True

    monkeypatch.setattr(
        "axon.future._get_process_wake_queue", lambda: mock_process_wake_queue
    )
    monkeypatch.setattr("axon.future.AXON_DISABLE_FAST_PATH", False)

    loop = asyncio.get_running_loop()
    inner_fut = loop.create_future()

    # Simulate an asynchronous completion
    async def complete_later():
        await asyncio.sleep(0.01)
        inner_fut.set_result("slow_result")

    asyncio.create_task(complete_later())

    axon_fut = AxonFuture(inner_fut)
    result = await axon_fut

    assert queue_processed, "Fast path should still probe the queue first"
    assert result == "slow_result"


@pytest.mark.asyncio
async def test_axon_future_disable_fast_path(monkeypatch):
    queue_processed = False

    def mock_process_wake_queue():
        nonlocal queue_processed
        queue_processed = True

    monkeypatch.setattr(
        "axon.future._get_process_wake_queue", lambda: mock_process_wake_queue
    )
    monkeypatch.setattr("axon.future.AXON_DISABLE_FAST_PATH", True)

    loop = asyncio.get_running_loop()
    inner_fut = loop.create_future()
    inner_fut.set_result("disabled_result")

    axon_fut = AxonFuture(inner_fut)
    result = await axon_fut

    assert (
        not queue_processed
    ), "Queue should not be processed when fast path is disabled"
    assert result == "disabled_result"
