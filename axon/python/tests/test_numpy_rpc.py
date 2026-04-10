import pytest
import numpy as np
import asyncio
import gc
import weakref
from typing import List, Tuple
import test_utils  # noqa: F401
import axon


# Define test functions
async def sum_int_func(a: np.ndarray, b: np.ndarray) -> int:
    result = np.sum(a + b)
    return int(result)


async def sum_arr_func(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    result = np.sum(a + b) + np.sum(c)
    result_arr = np.array([result], dtype=np.int64)
    return result_arr


async def list_func(a: np.ndarray, b: np.ndarray) -> List[np.ndarray]:
    c = a + b
    d = a * b
    return [c, d]


async def tuple_func(
    x: np.ndarray, y: np.ndarray
) -> Tuple[int, np.ndarray, np.ndarray]:
    result_int = int(np.sum(x) + np.sum(y))
    return (result_int, x.copy(), y.copy())


async def mixed_args_func(a: np.ndarray, b: int, c: str) -> str:
    return f"{np.sum(a)}_{b}_{c}"


async def receive_shards_func(
    op_name: str,
    names: List[str],
    step: int,
    sender_rank: int,
    payload_groups: List[List[np.ndarray]],
    counts_groups: List[List[int]],
    is_ragged_flags: List[bool],
    has_indices_flags: List[bool],
) -> int:
    # Basic verification of received data
    assert isinstance(op_name, str)
    assert isinstance(names, list)
    assert isinstance(payload_groups, list)
    return 0


class RpcTestContext:
    def __init__(self):
        self.server = None
        self.client = None

    async def __aenter__(self):
        # Start server
        self.server = axon.AxonRuntime("test_worker")
        self.server.start()

        self.server.register_function(sum_int_func, 0, from_dlpack_fn=np.from_dlpack)
        self.server.register_function(sum_arr_func, 1, from_dlpack_fn=np.from_dlpack)
        self.server.register_function(list_func, 2, from_dlpack_fn=np.from_dlpack)
        self.server.register_function(tuple_func, 3, from_dlpack_fn=np.from_dlpack)
        self.server.register_function(mixed_args_func, 4, from_dlpack_fn=np.from_dlpack)
        self.server.register_function(
            receive_shards_func, 5, from_dlpack_fn=np.from_dlpack
        )

        server_addr = self.server.get_local_address()

        # Start client
        self.client = axon.AxonRuntime("client_worker")
        self.client.start_client()

        await self.client.connect_endpoint_async(server_addr, "test_worker")

        return self.client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.stop()
        if self.server:
            self.server.stop()


@pytest.mark.asyncio
async def test_sum_int_func():
    async with RpcTestContext() as client:
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        b = np.array([9, 10, 11, 12, 13, 14, 15, 16], dtype=np.int64)

        # Expected: sum(a+b) = sum([10, 12, ..., 24]) = 136
        result = await client.invoke(
            a, b, worker_name="test_worker", session_id=0, function=0
        )

        # invoke returns a int
        assert isinstance(result, int)
        assert result == 136


@pytest.mark.asyncio
async def test_sum_arr_func():
    """Test single tensor return with automatic from_dlpack conversion."""
    async with RpcTestContext() as client:
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        b = np.array([9, 10, 11, 12, 13, 14, 15, 16], dtype=np.int64)
        c = np.array([17, 18, 19, 20], dtype=np.int64)

        # Expected: sum(a+b) + sum(c) = 136 + 74 = 210
        # Use from_dlpack to convert to numpy array
        result = await client.invoke(
            a,
            b,
            c,
            worker_name="test_worker",
            session_id=0,
            function=1,
            from_dlpack_fn=np.from_dlpack,
        )

        # With from_dlpack, result should be a numpy array
        assert isinstance(result, np.ndarray)
        assert result[0] == 210


@pytest.mark.asyncio
async def test_list_func():
    """Test List[Tensor] return with automatic from_dlpack conversion."""
    async with RpcTestContext() as client:
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        b = np.array([9, 10, 11, 12, 13, 14, 15, 16], dtype=np.int64)

        # Use from_dlpack to convert to numpy arrays
        result = await client.invoke(
            a,
            b,
            worker_name="test_worker",
            session_id=0,
            function=2,
            from_dlpack_fn=np.from_dlpack,
        )

        assert isinstance(result, list)
        assert len(result) == 2

        # With from_dlpack, results should be numpy arrays
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)
        np.testing.assert_array_equal(result[0], a + b)
        np.testing.assert_array_equal(result[1], a * b)


@pytest.mark.asyncio
async def test_tuple_func():
    """Test Tuple[int, Tensor, Tensor] return with automatic from_dlpack conversion."""
    async with RpcTestContext() as client:
        x = np.array([1, 2, 3], dtype=np.int64)
        y = np.array([4, 5, 6], dtype=np.int64)

        # Expected int: sum(x) + sum(y) = 6 + 15 = 21
        # Use from_dlpack to convert to numpy arrays
        result = await client.invoke(
            x,
            y,
            worker_name="test_worker",
            session_id=0,
            function=3,
            from_dlpack_fn=np.from_dlpack,
        )

        assert isinstance(result, tuple)
        assert len(result) == 3

        res_int = result[0]
        assert res_int == 21

        # With from_dlpack, results should be numpy arrays
        assert isinstance(result[1], np.ndarray)
        assert isinstance(result[2], np.ndarray)
        np.testing.assert_array_equal(result[1], x)
        np.testing.assert_array_equal(result[2], y)


@pytest.mark.asyncio
async def test_mixed_args_func():
    """Test mixed arguments (Tensor, int, str) -> str."""
    async with RpcTestContext() as client:
        a = np.array([1, 2, 3], dtype=np.int64)
        b = 100
        c = "hello"

        # Expected: "6_100_hello"
        result = await client.invoke(
            a,
            b,
            c,
            worker_name="test_worker",
            session_id=0,
            function=4,
            from_dlpack_fn=np.from_dlpack,
        )

        assert isinstance(result, str)
        assert result == "6_100_hello"


@pytest.mark.asyncio
async def test_receive_shards():
    """Test complex receive_shards function signature."""
    async with RpcTestContext() as client:
        op_name = "test_op"
        names = ["tensor_1", "tensor_2"]
        step = 10
        sender_rank = 0
        payload_groups = [
            [np.random.rand(2, 2), np.random.rand(3, 3)],
            [np.random.rand(4, 4)],
        ]
        counts_groups = [[2, 2], [1]]
        is_ragged_flags = [False, True]
        has_indices_flags = [True, False]

        result = await client.invoke(
            op_name,
            names,
            step,
            sender_rank,
            payload_groups,
            counts_groups,
            is_ragged_flags,
            has_indices_flags,
            worker_name="test_worker",
            session_id=0,
            function=5,
            from_dlpack_fn=np.from_dlpack,
        )

        assert result == 0


@pytest.mark.asyncio
async def test_invoke_args_classification():
    """Test argument classification fixes for tuple, nested list, and empty list."""
    server = axon.AxonRuntime("test_classify_server")
    server.start()

    async def dummy_rpc(tensors: List[np.ndarray]) -> bool:
        return True

    server.register_function(
        dummy_rpc,
        99,
        from_dlpack_fn=np.from_dlpack,
    )

    client = axon.AxonRuntime("test_classify_client")
    client.start_client()

    await client.connect_endpoint_async(
        server.get_local_address(), "test_classify_server"
    )

    try:
        tensor = np.ones((10,), dtype=np.float32)

        # tuple of tensors - ClassifyArg now treats tuple as a valid sequence
        result = await client.invoke(
            (tensor, tensor),
            worker_name="test_classify_server",
            session_id=0,
            function=99,
        )
        assert result is True

        # nested list starting with empty inner list - ClassifyArg now scans
        # past empty inner lists to find the first non-empty one
        result = await client.invoke(
            [[], [tensor]],
            worker_name="test_classify_server",
            session_id=0,
            function=99,
        )
        assert result is True

        # empty outer list - ClassifyArg now returns FlatTensorList for empty
        # sequences, serializing as empty TENSOR_META_VEC rather than VOID
        result = await client.invoke(
            [], worker_name="test_classify_server", session_id=0, function=99
        )
        assert result is True

    finally:
        try:
            client.stop()
            server.stop()
        except:
            pass


@pytest.mark.asyncio
async def test_nested_list():
    """Test List[List[int]] and List[str] RPC serialization/deserialization."""
    server = axon.AxonRuntime("test_nested_list_server")
    server.start()

    async def _test_nested(
        nested: List[List[int]], strings: List[str]
    ) -> List[List[int]]:
        return [[x + 1 for x in group] for group in nested]

    server.register_function(_test_nested, 100, from_dlpack_fn=np.from_dlpack)
    server_addr = server.get_local_address()

    client = axon.AxonRuntime("test_nested_list_client")
    client.start_client()
    await client.connect_endpoint_async(server_addr, "test_nested_list_server")

    try:
        nested_in = [[1, 2], [], [3, 4, 5]]
        strings_in = ["hello", "axon", "rpc"]
        result = await client.invoke(
            nested_in,
            strings_in,
            worker_name="test_nested_list_server",
            session_id=0,
            function=100,
        )
        assert result == [[2, 3], [], [4, 5, 6]]

        try:
            # Test unsupported
            await client.invoke(
                [["a"]],
                ["b"],
                worker_name="test_nested_list_server",
                session_id=0,
                function=100,
            )
            assert False, "Should have thrown error"
        except Exception as e:
            assert (
                "unsupported element type" in str(e)
                or "ParamType" in str(e)
                or "could not dispatch" in str(e)
                or "Axon failed" in str(e)
                or "RPC invocation failed" in str(e)
            )
    finally:
        client.stop()
        server.stop()


@pytest.mark.asyncio
async def test_empty_nested_list_bug():
    """Test the bug related to empty nested lists type inference."""
    server = axon.AxonRuntime("test_empty_nested_server")
    server.start()

    async def _test_nested(nested: List[List[int]]) -> List[List[int]]:
        return nested

    server.register_function(_test_nested, 101, from_dlpack_fn=np.from_dlpack)
    server_addr = server.get_local_address()

    client = axon.AxonRuntime("test_empty_nested_client")
    client.start_client()
    await client.connect_endpoint_async(server_addr, "test_empty_nested_server")

    try:
        nested_in = [[], [], []]

        try:
            result = await client.invoke(
                nested_in,
                worker_name="test_empty_nested_server",
                session_id=0,
                function=101,
            )
        except Exception as e:
            pass

    finally:
        try:
            client.stop()
            server.stop()
        except:
            pass


@pytest.mark.asyncio
async def test_nested_bool():
    server = axon.AxonRuntime("test_nested_bool_server")
    server.start()

    async def _test_bool(nested: List[List[bool]]) -> List[List[bool]]:
        return nested

    server.register_function(_test_bool, 102, from_dlpack_fn=np.from_dlpack)
    server_addr = server.get_local_address()

    client = axon.AxonRuntime("test_nested_bool_client")
    client.start_client()
    await client.connect_endpoint_async(server_addr, "test_nested_bool_server")

    try:
        nested_in = [[True, False], [], [False]]
        result = await client.invoke(
            nested_in,
            worker_name="test_nested_bool_server",
            session_id=0,
            function=102,
        )
        assert result == [
            [True, False],
            [],
            [False],
        ], f"Expected [[True, False], [], [False]], got {result}"
        for row in result:
            for item in row:
                assert isinstance(item, bool), f"Expected bool, got {type(item)}"
    finally:
        try:
            client.stop()
            server.stop()
        except:
            pass


_stored_input_tensor = None
_released_input_tensor = None
_released_input_tensor_ref = None


async def _store_input_func(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    global _stored_input_tensor
    _stored_input_tensor = a
    return a + b


async def _store_then_release_input_func(a: np.ndarray) -> int:
    global _released_input_tensor, _released_input_tensor_ref

    _released_input_tensor = a
    _released_input_tensor_ref = weakref.ref(a)
    del a

    await asyncio.sleep(0)

    result = int(np.sum(_released_input_tensor))
    _released_input_tensor = None
    return result


async def _wait_for_object_release(obj_ref, retries: int = 10):
    for _ in range(retries):
        if obj_ref is None or obj_ref() is None:
            return True
        gc.collect()
        await asyncio.sleep(0)
    return obj_ref() is None


@pytest.mark.asyncio
async def test_input_tensor_lifetime_safety():
    """Input tensors must remain valid when stored beyond the coroutine lifetime.

    The server function captures an input tensor into a global. With the old
    non-owning UcxBuffer design, the backing UCX buffer was freed when the
    payload_keeper callback fired (after coroutine completion), causing a
    use-after-free. With the shared_ptr<void> keeper design the tensor keeps
    the buffer alive for as long as any Python object references it.
    """
    global _stored_input_tensor
    _stored_input_tensor = None

    server = axon.AxonRuntime("test_lifetime_server")
    server.start()
    server.register_function(_store_input_func, 200, from_dlpack_fn=np.from_dlpack)

    client = axon.AxonRuntime("test_lifetime_client")
    client.start_client()
    await client.connect_endpoint_async(
        server.get_local_address(), "test_lifetime_server"
    )

    try:
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)

        result = await client.invoke(
            a,
            b,
            worker_name="test_lifetime_server",
            session_id=0,
            function=200,
            from_dlpack_fn=np.from_dlpack,
        )

        # Yield to the event loop so the done_callback fires and
        # payload_keeper in the callback lambda is released.
        await asyncio.sleep(0)

        # _stored_input_tensor must still point to valid memory even after
        # the callback released payload_keeper.
        assert _stored_input_tensor is not None
        np.testing.assert_array_equal(_stored_input_tensor, a)
        np.testing.assert_array_equal(result, a + b)
    finally:
        _stored_input_tensor = None
        try:
            client.stop()
            server.stop()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_input_tensor_is_released_after_rpc():
    """Stored server-side tensor references should be released after RPC."""
    global _released_input_tensor, _released_input_tensor_ref
    _released_input_tensor = None
    _released_input_tensor_ref = None

    server = axon.AxonRuntime("test_release_server")
    server.start()
    server.register_function(
        _store_then_release_input_func,
        201,
        from_dlpack_fn=np.from_dlpack,
    )

    client = axon.AxonRuntime("test_release_client")
    client.start_client()
    await client.connect_endpoint_async(server.get_local_address(), "test_release_server")

    try:
        a = np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float64)

        result = await client.invoke(
            a,
            worker_name="test_release_server",
            session_id=0,
            function=201,
            from_dlpack_fn=np.from_dlpack,
        )

        assert result == int(np.sum(a))
        assert _released_input_tensor is None
        assert _released_input_tensor_ref is not None
        assert await _wait_for_object_release(_released_input_tensor_ref)
    finally:
        _released_input_tensor = None
        _released_input_tensor_ref = None
        try:
            client.stop()
            server.stop()
        except Exception:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
