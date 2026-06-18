"""Private synchronous fetch preserves response ownership and completion errors."""
import asyncio
from typing import List

import axon
from axon import _axon
import numpy as np
import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize('elements,scalar', [(0,False), (1,False), (32768,False), (1,True), (32768,True)])
async def test_sync_fetch_completion_and_lifetime(elements, scalar):
    server, client = axon.AxonRuntime('fetch_server'), axon.AxonRuntime('fetch_client')
    entered, release = asyncio.Event(), asyncio.Event()
    expected = np.arange(elements, dtype=np.int32)

    async def fetch(object_id: str, transfer_id: str) -> List[np.ndarray]:
        assert transfer_id == 'transfer'
        if object_id == 'missing':
            raise KeyError(object_id)
        entered.set()
        await release.wait()
        return expected if scalar else [expected, np.array([19], dtype=np.uint8)]

    if scalar:
        fetch.__annotations__["return"] = np.ndarray

    server.start()
    client.start_client()
    task = None
    try:
        server.register_function(fetch, 19)
        await client.connect_endpoint_async(server.get_local_address(), 'fetch_server')
        task = asyncio.create_task(asyncio.to_thread(
            _axon._sync_fetch, client, 'object', 'transfer', 'fetch_server', 19, 27,
            np.from_dlpack))
        await asyncio.wait_for(entered.wait(), 5)
        assert not task.done(), 'fetch must await the response'
        release.set()
        result = await asyncio.wait_for(task, 5)
        if scalar:
            result = [result]
        assert len(result) == (1 if scalar else 2)
        np.testing.assert_array_equal(result[0], expected)
        if not scalar:
            np.testing.assert_array_equal(result[1], [19])
        for obj_id, function in [('missing', 19), ('object', 999999)]:
            with pytest.raises(RuntimeError):
                await asyncio.to_thread(_axon._sync_fetch, client, obj_id, 'transfer',
                                        'fetch_server', function, 27, np.from_dlpack)
    finally:
        release.set()
        if task is not None and not task.done():
            await asyncio.wait_for(task, 5)
        client.stop()
        server.stop()
    np.testing.assert_array_equal(result[0], expected)
    if not scalar:
        np.testing.assert_array_equal(result[1], [19])
