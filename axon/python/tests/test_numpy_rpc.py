import pytest
import numpy as np
import asyncio
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


class RpcTestContext:
    def __init__(self):
        self.server = None
        self.client = None

    async def __aenter__(self):
        # Start server
        self.server = axon.AxonRuntime("test_worker")
        self.server.start()

        self.server.register_function(0, sum_int_func, from_dlpack_fn=np.from_dlpack)
        self.server.register_function(1, sum_arr_func, from_dlpack_fn=np.from_dlpack)
        self.server.register_function(2, list_func, from_dlpack_fn=np.from_dlpack)
        self.server.register_function(3, tuple_func, from_dlpack_fn=np.from_dlpack)
        self.server.register_function(4, mixed_args_func, from_dlpack_fn=np.from_dlpack)

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
            a, b, worker_name="test_worker", session_id=0, function_id=0
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
            function_id=1,
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
            function_id=2,
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
            function_id=3,
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
            function_id=4,
            from_dlpack_fn=np.from_dlpack,
        )

        assert isinstance(result, str)
        assert result == "6_100_hello"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
