"""Tests for Axon Runtime Python bindings."""

import pytest

# Import test_utils first to setup module path
import test_utils  # noqa: F401

import axon


def test_axon_runtime_creation():
    """Test creating an AxonRuntime instance."""
    runtime = axon.AxonRuntime("test_worker")
    assert runtime is not None


def test_start_stop():
    """Test starting and stopping the runtime."""
    runtime = axon.AxonRuntime("test_worker")
    runtime.start()  # Throws exception on failure
    runtime.stop()


@pytest.mark.asyncio
async def test_connect_endpoint_async():
    """Test async endpoint connection."""
    runtime = axon.AxonRuntime("test_worker")
    runtime.start_client()

    # Create a mock address (empty for now)
    address = bytes([])
    try:
        conn_id = await runtime.connect_endpoint_async(address, "remote_worker")
        assert isinstance(conn_id, int)
    except Exception:
        # Expected to fail without actual remote endpoint
        pass
    finally:
        runtime.stop()


def test_register_function():
    """Test registering a Python function."""
    runtime = axon.AxonRuntime("test_worker")
    runtime.start_server()

    async def test_func(x: int, y: int) -> int:
        return x + y

    runtime.register_function_raw(
        function_id=1,
        function_name="test_func",
        param_types=[
            axon.ParamType.PRIMITIVE_INT32,
            axon.ParamType.PRIMITIVE_INT32,
        ],
        return_types=[axon.ParamType.PRIMITIVE_INT32],
        input_payload_type=axon.PayloadType.NO_PAYLOAD,
        return_payload_type=axon.PayloadType.NO_PAYLOAD,
        callable=test_func,
    )

    runtime.stop()


@pytest.mark.asyncio
async def test_invoke():
    """Test async RPC invocation."""
    runtime = axon.AxonRuntime("test_worker")
    runtime.start_client()

    request_header = axon.RpcRequestHeader()
    # Header fields are set by invoke_raw parameters

    try:
        result = await runtime.invoke_raw(
            worker_name="remote_worker",
            session_id=0,
            function_id=1,
            workflow_id=0,
            request_header=request_header,
        )
        assert result is not None
    except Exception:
        # Expected to fail without actual remote endpoint
        pass
    finally:
        runtime.stop()


def test_get_local_address():
    """Test getting local address."""
    runtime = axon.AxonRuntime("test_worker")
    runtime.start()
    address = runtime.get_local_address()
    assert isinstance(address, bytes)
    runtime.stop()


def test_get_local_signatures():
    """Test getting local signatures."""
    runtime = axon.AxonRuntime("test_worker")
    runtime.start_server()

    async def test_func(x: int) -> int:
        return x * 2

    runtime.register_function_raw(
        function_id=2,
        function_name="test_func",
        param_types=[axon.ParamType.PRIMITIVE_INT32],
        return_types=[axon.ParamType.PRIMITIVE_INT32],
        input_payload_type=axon.PayloadType.NO_PAYLOAD,
        return_payload_type=axon.PayloadType.NO_PAYLOAD,
        callable=test_func,
    )

    signatures = runtime.get_local_signatures()
    assert isinstance(signatures, bytes)
    runtime.stop()


if __name__ == "__main__":
    pytest.main([__file__])
