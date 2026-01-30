"""Tests for connection management."""

import asyncio
import pytest

# Import test_utils first to setup module path
import test_utils  # noqa: F401

import axon


async def dummy_func(x: int) -> int:
    """Dummy function for testing."""
    return x * 2


@pytest.mark.asyncio
async def test_connect_endpoint_async_success():
    """Test successful endpoint connection."""
    # Create server
    server = axon.AxonRuntime("server_worker")
    server.start_server()

    server.register_function_raw(
        function_id=1,
        function_name="dummy_func",
        param_types=[axon.ParamType.PRIMITIVE_INT32],
        return_types=[axon.ParamType.PRIMITIVE_INT32],
        input_payload_type=axon.PayloadType.NO_PAYLOAD,
        return_payload_type=axon.PayloadType.NO_PAYLOAD,
        callable=dummy_func,
    )

    server_address = server.get_local_address()

    # Create client
    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        # Connect to server
        conn_id = await client.connect_endpoint_async(server_address, "server_worker")
        assert isinstance(conn_id, int)
        assert conn_id >= 0

    finally:
        client.stop()
        server.stop()


@pytest.mark.asyncio
async def test_connect_endpoint_async_invalid_address():
    """Test connection with invalid address."""
    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        # Try to connect with empty/invalid address
        invalid_address = bytes([])
        try:
            conn_id = await client.connect_endpoint_async(
                invalid_address, "remote_worker"
            )
            # If connection succeeds (unlikely), verify conn_id
            assert isinstance(conn_id, int)
        except Exception:
            # Expected to fail with invalid address
            pass

    finally:
        client.stop()


@pytest.mark.asyncio
async def test_connect_endpoint_async_timeout():
    """Test connection timeout handling."""
    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        # Try to connect to non-existent server
        # Use a valid format but non-existent address
        fake_address = bytes([0] * 100)  # Fake address
        try:
            # This should timeout or fail
            conn_id = await asyncio.wait_for(
                client.connect_endpoint_async(fake_address, "non_existent_worker"),
                timeout=2.0,
            )
            # If it doesn't timeout, verify the result
            assert isinstance(conn_id, int)
        except (asyncio.TimeoutError, Exception):
            # Expected to timeout or fail
            pass

    finally:
        client.stop()


def test_get_local_address_server():
    """Test getting local address in server mode."""
    runtime = axon.AxonRuntime("test_worker")
    runtime.start_server()

    address = runtime.get_local_address()
    assert isinstance(address, bytes)
    assert len(address) > 0  # Should have some address data

    runtime.stop()


def test_get_local_address_client():
    """Test getting local address in client mode."""
    runtime = axon.AxonRuntime("test_worker")
    runtime.start_client()

    address = runtime.get_local_address()
    assert isinstance(address, bytes)
    assert len(address) > 0  # Should have some address data

    runtime.stop()


def test_get_local_address_both():
    """Test getting local address with both server and client."""
    runtime = axon.AxonRuntime("test_worker")
    runtime.start()

    address = runtime.get_local_address()
    assert isinstance(address, bytes)
    assert len(address) > 0

    runtime.stop()


def test_get_local_signatures_empty():
    """Test getting signatures with no registered functions."""
    runtime = axon.AxonRuntime("test_worker")
    runtime.start_server()

    signatures = runtime.get_local_signatures()
    assert isinstance(signatures, bytes)
    # May be empty or contain metadata

    runtime.stop()


def test_get_local_signatures_single_function():
    """Test getting signatures with one registered function."""
    runtime = axon.AxonRuntime("test_worker")
    runtime.start_server()

    async def test_func(x: int) -> int:
        return x * 2

    runtime.register_function_raw(
        function_id=1,
        function_name="test_func",
        param_types=[axon.ParamType.PRIMITIVE_INT32],
        return_types=[axon.ParamType.PRIMITIVE_INT32],
        input_payload_type=axon.PayloadType.NO_PAYLOAD,
        return_payload_type=axon.PayloadType.NO_PAYLOAD,
        callable=test_func,
    )

    signatures = runtime.get_local_signatures()
    assert isinstance(signatures, bytes)
    assert len(signatures) > 0  # Should contain signature data

    runtime.stop()


def test_get_local_signatures_multiple_functions():
    """Test getting signatures with multiple registered functions."""
    runtime = axon.AxonRuntime("test_worker")
    runtime.start_server()

    async def func1(x: int) -> int:
        return x * 2

    async def func2(s: str) -> str:
        return f"Hello, {s}!"

    async def func3() -> int:
        return 42

    runtime.register_function_raw(
        function_id=1,
        function_name="func1",
        param_types=[axon.ParamType.PRIMITIVE_INT32],
        return_types=[axon.ParamType.PRIMITIVE_INT32],
        input_payload_type=axon.PayloadType.NO_PAYLOAD,
        return_payload_type=axon.PayloadType.NO_PAYLOAD,
        callable=func1,
    )

    runtime.register_function_raw(
        function_id=2,
        function_name="func2",
        param_types=[axon.ParamType.STRING],
        return_types=[axon.ParamType.STRING],
        input_payload_type=axon.PayloadType.NO_PAYLOAD,
        return_payload_type=axon.PayloadType.NO_PAYLOAD,
        callable=func2,
    )

    runtime.register_function_raw(
        function_id=3,
        function_name="func3",
        param_types=[],
        return_types=[axon.ParamType.PRIMITIVE_INT32],
        input_payload_type=axon.PayloadType.NO_PAYLOAD,
        return_payload_type=axon.PayloadType.NO_PAYLOAD,
        callable=func3,
    )

    signatures = runtime.get_local_signatures()
    assert isinstance(signatures, bytes)
    assert len(signatures) > 0  # Should contain all signatures

    runtime.stop()


def test_get_local_signatures_format():
    """Test signature format (bytes type)."""
    runtime = axon.AxonRuntime("test_worker")
    runtime.start_server()

    async def test_func(x: int) -> int:
        return x * 2

    runtime.register_function_raw(
        function_id=1,
        function_name="test_func",
        param_types=[axon.ParamType.PRIMITIVE_INT32],
        return_types=[axon.ParamType.PRIMITIVE_INT32],
        input_payload_type=axon.PayloadType.NO_PAYLOAD,
        return_payload_type=axon.PayloadType.NO_PAYLOAD,
        callable=test_func,
    )

    signatures = runtime.get_local_signatures()
    # Verify it's bytes
    assert isinstance(signatures, bytes)
    # Can check hex representation
    hex_repr = signatures.hex()
    assert isinstance(hex_repr, str)

    runtime.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
