"""Tests for RPC invocation."""

import asyncio
import pytest

# Import test_utils first to setup module path
import test_utils  # noqa: F401

import axon


async def server_add(x: int, y: int) -> int:
    """Server add function."""
    return x + y


async def server_greet(name: str) -> str:
    """Server greet function."""
    return f"Hello, {name}!"


@pytest.mark.asyncio
async def test_basic_rpc_call():
    """Test basic RPC call flow."""
    # Create server
    server = axon.AxonRuntime("server_worker")
    server.start_server()

    # Register function
    server.register_function_raw(
        function_id=1,
        function_name="server_add",
        param_types=[
            axon.ParamType.PRIMITIVE_INT32,
            axon.ParamType.PRIMITIVE_INT32,
        ],
        return_types=[axon.ParamType.PRIMITIVE_INT32],
        input_payload_type=axon.PayloadType.NO_PAYLOAD,
        return_payload_type=axon.PayloadType.NO_PAYLOAD,
        callable=server_add,
    )

    # Get server address
    server_address = server.get_local_address()

    # Create client
    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        # Connect to server
        conn_id = await client.connect_endpoint_async(server_address, "server_worker")
        assert isinstance(conn_id, int)

        # Create request header
        request_header = axon.RpcRequestHeader()
        request_header.function_id = 1
        request_header.session_id = 0
        request_header.request_id = 1
        request_header.workflow_id = 0
        request_header.AddParam(10)
        request_header.AddParam(20)

        # Invoke RPC
        result = await client.invoke_raw(
            worker_name="server_worker",
            request_header=request_header,
        )

        # Verify response
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == 30

    finally:
        client.stop()
        server.stop()


@pytest.mark.asyncio
async def test_rpc_call_with_string():
    """Test RPC call with string parameter."""
    # Create server
    server = axon.AxonRuntime("server_worker")
    server.start_server()

    # Register function
    server.register_function_raw(
        function_id=2,
        function_name="server_greet",
        param_types=[axon.ParamType.STRING],
        return_types=[axon.ParamType.STRING],
        input_payload_type=axon.PayloadType.NO_PAYLOAD,
        return_payload_type=axon.PayloadType.NO_PAYLOAD,
        callable=server_greet,
    )

    server_address = server.get_local_address()

    # Create client
    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        # Connect
        await client.connect_endpoint_async(server_address, "server_worker")

        # Create request
        request_header = axon.RpcRequestHeader()
        request_header.function_id = 2
        request_header.session_id = 0
        request_header.request_id = 1
        request_header.workflow_id = 0
        request_header.AddParam("World")

        # Invoke
        result = await client.invoke_raw(
            worker_name="server_worker",
            request_header=request_header,
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == "Hello, World!"

    finally:
        client.stop()
        server.stop()


@pytest.mark.asyncio
async def test_rpc_call_with_payload_none():
    """Test RPC call with no payload (monostate)."""
    server = axon.AxonRuntime("server_worker")
    server.start_server()

    server.register_function_raw(
        function_id=3,
        function_name="server_add",
        param_types=[
            axon.ParamType.PRIMITIVE_INT32,
            axon.ParamType.PRIMITIVE_INT32,
        ],
        return_types=[axon.ParamType.PRIMITIVE_INT32],
        input_payload_type=axon.PayloadType.NO_PAYLOAD,
        return_payload_type=axon.PayloadType.NO_PAYLOAD,
        callable=server_add,
    )

    server_address = server.get_local_address()
    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        await client.connect_endpoint_async(server_address, "server_worker")

        request_header = axon.RpcRequestHeader()
        request_header.function_id = 3
        request_header.session_id = 0
        request_header.request_id = 1
        request_header.workflow_id = 0
        request_header.AddParam(10)
        request_header.AddParam(20)

        # Call with payload=None (default)
        result = await client.invoke_raw(
            worker_name="server_worker",
            request_header=request_header,
            payload=None,
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == 30

    finally:
        client.stop()
        server.stop()


@pytest.mark.asyncio
async def test_rpc_call_request_header_fields():
    """Test RPC call with different request header fields."""
    server = axon.AxonRuntime("server_worker")
    server.start_server()

    server.register_function_raw(
        function_id=4,
        function_name="server_add",
        param_types=[
            axon.ParamType.PRIMITIVE_INT32,
            axon.ParamType.PRIMITIVE_INT32,
        ],
        return_types=[axon.ParamType.PRIMITIVE_INT32],
        input_payload_type=axon.PayloadType.NO_PAYLOAD,
        return_payload_type=axon.PayloadType.NO_PAYLOAD,
        callable=server_add,
    )

    server_address = server.get_local_address()
    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        await client.connect_endpoint_async(server_address, "server_worker")

        # Test different session_id, request_id, workflow_id
        request_header = axon.RpcRequestHeader()
        request_header.function_id = 4
        request_header.session_id = 100
        request_header.request_id = 200
        request_header.workflow_id = 300
        request_header.AddParam(10)
        request_header.AddParam(20)

        result = await client.invoke_raw(
            worker_name="server_worker",
            request_header=request_header,
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == 30

    finally:
        client.stop()
        server.stop()


@pytest.mark.asyncio
async def test_rpc_call_error_handling():
    """Test RPC call error handling."""
    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        # Try to call non-existent server
        request_header = axon.RpcRequestHeader()
        request_header.function_id = 999
        request_header.session_id = 0
        request_header.request_id = 1
        request_header.workflow_id = 0

        # Should fail
        try:
            result = await client.invoke_raw(
                worker_name="non_existent_worker",
                request_header=request_header,
            )
        except Exception:
            # Expected to fail
            pass

    finally:
        client.stop()


@pytest.mark.asyncio
async def test_rpc_call_memory_policy():
    """Test RPC call with memory policy (default None)."""
    server = axon.AxonRuntime("server_worker")
    server.start_server()

    server.register_function_raw(
        function_id=5,
        function_name="server_add",
        param_types=[
            axon.ParamType.PRIMITIVE_INT32,
            axon.ParamType.PRIMITIVE_INT32,
        ],
        return_types=[axon.ParamType.PRIMITIVE_INT32],
        input_payload_type=axon.PayloadType.NO_PAYLOAD,
        return_payload_type=axon.PayloadType.NO_PAYLOAD,
        callable=server_add,
    )

    server_address = server.get_local_address()
    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        await client.connect_endpoint_async(server_address, "server_worker")

        request_header = axon.RpcRequestHeader()
        request_header.function_id = 5
        request_header.session_id = 0
        request_header.request_id = 1
        request_header.workflow_id = 0
        request_header.AddParam(10)
        request_header.AddParam(20)

        # Call with memory_policy=None (default)
        result = await client.invoke_raw(
            worker_name="server_worker",
            request_header=request_header,
            memory_policy=None,
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == 30

    finally:
        client.stop()
        server.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
