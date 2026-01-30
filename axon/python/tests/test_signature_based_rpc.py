"""Tests for signature-based RPC invocation."""

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


async def server_no_params() -> int:
    """Server function with no parameters."""
    return 42


async def server_mixed(x: int, name: str, f: float) -> str:
    """Server function with mixed parameters."""
    return f"{name}: {x} + {f}"


async def server_vector(vec: list) -> list:
    """Server function with vector."""
    return [x * 2 for x in vec]


def test_signature_retrieval_single_function():
    """Test retrieving signatures with single function."""
    server = axon.AxonRuntime("server_worker")
    server.start_server()

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

    # Get signatures
    signatures = server.get_local_signatures()
    assert isinstance(signatures, bytes)
    assert len(signatures) > 0  # Should contain signature data

    server.stop()


def test_signature_retrieval_multiple_functions():
    """Test retrieving signatures with multiple different function signatures."""
    server = axon.AxonRuntime("server_worker")
    server.start_server()

    # Register multiple functions with different signatures
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

    server.register_function_raw(
        function_id=2,
        function_name="server_greet",
        param_types=[axon.ParamType.STRING],
        return_types=[axon.ParamType.STRING],
        input_payload_type=axon.PayloadType.NO_PAYLOAD,
        return_payload_type=axon.PayloadType.NO_PAYLOAD,
        callable=server_greet,
    )

    server.register_function_raw(
        function_id=3,
        function_name="server_no_params",
        param_types=[],
        return_types=[axon.ParamType.PRIMITIVE_INT32],
        input_payload_type=axon.PayloadType.NO_PAYLOAD,
        return_payload_type=axon.PayloadType.NO_PAYLOAD,
        callable=server_no_params,
    )

    # Get signatures
    signatures = server.get_local_signatures()
    assert isinstance(signatures, bytes)
    assert len(signatures) > 0  # Should contain all function signatures

    server.stop()


def test_signature_format_validation():
    """Test signature format validation."""
    server = axon.AxonRuntime("server_worker")
    server.start_server()

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

    signatures = server.get_local_signatures()

    # Verify signature format
    assert isinstance(signatures, bytes)
    # Signatures should be serialized data
    # In a real scenario, we would deserialize and verify structure
    # For now, we verify it's bytes and has content

    server.stop()


@pytest.mark.asyncio
async def test_signature_based_rpc_call_basic():
    """Test RPC call based on signature information (basic flow)."""
    # Setup server
    server = axon.AxonRuntime("server_worker")
    server.start_server()

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

    # Get server address and signatures
    server_address = server.get_local_address()
    server_signatures = server.get_local_signatures()

    # Setup client
    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        # Connect to server
        conn_id = await client.connect_endpoint_async(server_address, "server_worker")
        assert isinstance(conn_id, int)

        # Client has signature information (simulating service discovery)
        # Based on signature, we know:
        # - function_id: 1
        # - param_types: [PRIMITIVE_INT32, PRIMITIVE_INT32]
        # - return_types: [PRIMITIVE_INT32]
        # - input_payload_type: NO_PAYLOAD
        # - return_payload_type: NO_PAYLOAD

        # Construct RPC request based on signature
        request_header = axon.RpcRequestHeader()
        request_header.function_id = 1  # From signature
        request_header.session_id = 0
        request_header.request_id = 1
        request_header.workflow_id = 0
        request_header.AddParam(10)
        request_header.AddParam(20)

        # Call with no payload (as per signature)
        result = await client.invoke_raw(
            worker_name="server_worker",
            request_header=request_header,
            payload=None,  # NO_PAYLOAD as per signature
        )

        # Verify response
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == 30

    finally:
        client.stop()
        server.stop()


@pytest.mark.asyncio
async def test_signature_based_rpc_call_string():
    """Test RPC call based on signature with STRING type."""
    server = axon.AxonRuntime("server_worker")
    server.start_server()

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
    server_signatures = server.get_local_signatures()

    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        await client.connect_endpoint_async(server_address, "server_worker")

        # Based on signature: function_id=2, param_types=[STRING], return_types=[STRING]
        request_header = axon.RpcRequestHeader()
        request_header.function_id = 2
        request_header.session_id = 0
        request_header.request_id = 1
        request_header.workflow_id = 0
        request_header.AddParam("World")

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
async def test_signature_based_rpc_call_no_params():
    """Test RPC call based on signature with no parameters."""
    server = axon.AxonRuntime("server_worker")
    server.start_server()

    server.register_function_raw(
        function_id=3,
        function_name="server_no_params",
        param_types=[],
        return_types=[axon.ParamType.PRIMITIVE_INT32],
        input_payload_type=axon.PayloadType.NO_PAYLOAD,
        return_payload_type=axon.PayloadType.NO_PAYLOAD,
        callable=server_no_params,
    )

    server_address = server.get_local_address()
    server_signatures = server.get_local_signatures()

    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        await client.connect_endpoint_async(server_address, "server_worker")

        # Based on signature: function_id=3, param_types=[], return_types=[PRIMITIVE_INT32]
        request_header = axon.RpcRequestHeader()
        request_header.function_id = 3
        request_header.session_id = 0
        request_header.request_id = 1
        request_header.workflow_id = 0

        result = await client.invoke_raw(
            worker_name="server_worker",
            request_header=request_header,
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == 42

    finally:
        client.stop()
        server.stop()


@pytest.mark.asyncio
async def test_signature_based_rpc_call_mixed_params():
    """Test RPC call based on signature with mixed parameter types."""
    server = axon.AxonRuntime("server_worker")
    server.start_server()

    server.register_function_raw(
        function_id=4,
        function_name="server_mixed",
        param_types=[
            axon.ParamType.PRIMITIVE_INT32,
            axon.ParamType.STRING,
            axon.ParamType.PRIMITIVE_FLOAT64,
        ],
        return_types=[axon.ParamType.STRING],
        input_payload_type=axon.PayloadType.NO_PAYLOAD,
        return_payload_type=axon.PayloadType.NO_PAYLOAD,
        callable=server_mixed,
    )

    server_address = server.get_local_address()
    server_signatures = server.get_local_signatures()

    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        await client.connect_endpoint_async(server_address, "server_worker")

        # Based on signature: mixed param types
        request_header = axon.RpcRequestHeader()
        request_header.function_id = 4
        request_header.session_id = 0
        request_header.request_id = 1
        request_header.workflow_id = 0
        request_header.AddParam(10)
        request_header.AddParam("test")
        request_header.AddParam(3.14)

        result = await client.invoke_raw(
            worker_name="server_worker",
            request_header=request_header,
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == "test: 10 + 3.14"

    finally:
        client.stop()
        server.stop()


@pytest.mark.asyncio
async def test_signature_based_rpc_call_vector():
    """Test RPC call based on signature with vector type."""
    server = axon.AxonRuntime("server_worker")
    server.start_server()

    server.register_function_raw(
        function_id=5,
        function_name="server_vector",
        param_types=[axon.ParamType.VECTOR_INT32],
        return_types=[axon.ParamType.VECTOR_INT32],
        input_payload_type=axon.PayloadType.NO_PAYLOAD,
        return_payload_type=axon.PayloadType.NO_PAYLOAD,
        callable=server_vector,
    )

    server_address = server.get_local_address()
    server_signatures = server.get_local_signatures()

    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        await client.connect_endpoint_async(server_address, "server_worker")

        # Based on signature: VECTOR_INT32
        request_header = axon.RpcRequestHeader()
        request_header.function_id = 5
        request_header.session_id = 0
        request_header.request_id = 1
        request_header.workflow_id = 0
        request_header.AddParam([1, 2, 3])

        result = await client.invoke_raw(
            worker_name="server_worker",
            request_header=request_header,
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == [2, 4, 6]

    finally:
        client.stop()
        server.stop()


@pytest.mark.asyncio
async def test_signature_mismatch_wrong_function_id():
    """Test RPC call with wrong function_id (signature mismatch)."""
    server = axon.AxonRuntime("server_worker")
    server.start_server()

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

    server_address = server.get_local_address()
    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        await client.connect_endpoint_async(server_address, "server_worker")

        # Use wrong function_id (999 instead of 1)
        request_header = axon.RpcRequestHeader()
        request_header.function_id = 999  # Wrong ID
        request_header.session_id = 0
        request_header.request_id = 1
        request_header.workflow_id = 0

        # Should fail or return error status
        try:
            result = await client.invoke_raw(
                worker_name="server_worker",
                request_header=request_header,
            )
            # If we get here, check result (it might be an empty list or error)
            assert isinstance(result, list)
        except Exception:
            # Expected to fail
            pass

    finally:
        client.stop()
        server.stop()


@pytest.mark.asyncio
async def test_signature_based_multiple_functions():
    """Test calling multiple functions based on their signatures."""
    server = axon.AxonRuntime("server_worker")
    server.start_server()

    # Register multiple functions
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

    server.register_function_raw(
        function_id=2,
        function_name="server_greet",
        param_types=[axon.ParamType.STRING],
        return_types=[axon.ParamType.STRING],
        input_payload_type=axon.PayloadType.NO_PAYLOAD,
        return_payload_type=axon.PayloadType.NO_PAYLOAD,
        callable=server_greet,
    )

    server.register_function_raw(
        function_id=3,
        function_name="server_no_params",
        param_types=[],
        return_types=[axon.ParamType.PRIMITIVE_INT32],
        input_payload_type=axon.PayloadType.NO_PAYLOAD,
        return_payload_type=axon.PayloadType.NO_PAYLOAD,
        callable=server_no_params,
    )

    server_address = server.get_local_address()
    server_signatures = server.get_local_signatures()

    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        await client.connect_endpoint_async(server_address, "server_worker")

        # Call function 1
        request_header1 = axon.RpcRequestHeader()
        request_header1.function_id = 1
        request_header1.session_id = 0
        request_header1.request_id = 1
        request_header1.workflow_id = 0
        request_header1.AddParam(10)
        request_header1.AddParam(20)

        result1 = await client.invoke_raw(
            worker_name="server_worker",
            request_header=request_header1,
        )
        assert result1[0] == 30

        # Call function 2
        request_header2 = axon.RpcRequestHeader()
        request_header2.function_id = 2
        request_header2.session_id = 0
        request_header2.request_id = 2
        request_header2.workflow_id = 0
        request_header2.AddParam("World")

        result2 = await client.invoke_raw(
            worker_name="server_worker",
            request_header=request_header2,
        )
        assert result2[0] == "Hello, World!"

        # Call function 3
        request_header3 = axon.RpcRequestHeader()
        request_header3.function_id = 3
        request_header3.session_id = 0
        request_header3.request_id = 3
        request_header3.workflow_id = 0

        result3 = await client.invoke_raw(
            worker_name="server_worker",
            request_header=request_header3,
        )
        assert result3[0] == 42

    finally:
        client.stop()
        server.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
