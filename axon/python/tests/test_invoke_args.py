"""Tests for invoking RPC with arguments list."""

import pytest

# Import test_utils first to setup module path
import test_utils  # noqa: F401

import axon
from test_utils import server_add, server_mixed


@pytest.mark.asyncio
async def test_invoke_with_request_header_allowed_params():
    """Test that request_header mode only allows worker_name, memory_policy, payload."""
    # Create server and register function
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

    # Create client
    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        # Connect to server
        await client.connect_endpoint_async(server_address, "server_worker")

        # Create request header
        request_header = axon.RpcRequestHeader()
        request_header.function_id = 1
        request_header.session_id = 0
        request_header.request_id = 1
        request_header.workflow_id = 0
        # For server_add(x: int, y: int), we need to add two int params
        request_header.AddParam(10)  # arg1
        request_header.AddParam(20)  # arg2

        # Test: Allowed parameters with request_header
        # This should not raise an error about disallowed arguments
        result = await client.invoke_raw(
            worker_name="server_worker",
            request_header=request_header,
            memory_policy=None,  # Allowed
            payload=None,  # Allowed
        )

        # Verify response - result is a list of return values
        assert isinstance(result, list)
        assert len(result) > 0
        # The result contains the function's return value(s)
        # For server_add(10, 20), result should be [30]
        assert result[0] == 30

    finally:
        client.stop()
        server.stop()


@pytest.mark.asyncio
async def test_invoke_with_request_header_disallowed_params():
    """Test that request_header mode rejects disallowed parameters like session_id, function_id, workflow_id."""
    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        request_header = axon.RpcRequestHeader()
        request_header.function_id = 1
        request_header.session_id = 0
        request_header.workflow_id = 0

        # Test: session_id should be rejected when using request_header
        with pytest.raises(TypeError):
            await client.invoke_raw(
                worker_name="server_worker",
                request_header=request_header,
                session_id=0,  # Should be rejected
            )

        # Test: function_id should be rejected when using request_header
        with pytest.raises(TypeError):
            await client.invoke_raw(
                worker_name="server_worker",
                request_header=request_header,
                function_id=1,  # Should be rejected
            )

        # Test: workflow_id should be rejected when using request_header
        with pytest.raises(TypeError):
            await client.invoke_raw(
                worker_name="server_worker",
                request_header=request_header,
                workflow_id=100,  # Should be rejected
            )

    finally:
        client.stop()


@pytest.mark.asyncio
async def test_invoke_with_request_header_and_payload():
    """Test that request_header mode allows payload to be passed separately."""
    # Create server and register function
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

    # Create client
    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        # Connect to server
        await client.connect_endpoint_async(server_address, "server_worker")

        # Create request header with TensorMeta params
        request_header = axon.RpcRequestHeader()
        request_header.function_id = 1
        request_header.session_id = 0
        request_header.request_id = 1
        request_header.workflow_id = 0
        # Add function parameters as TensorMeta
        request_header.AddParam(10)  # arg1
        request_header.AddParam(20)  # arg2
        # Note: params in request_header are TensorMeta

        # Test: payload can be passed separately when using request_header
        # This should not raise an error about disallowed arguments
        result = await client.invoke_raw(
            worker_name="server_worker",
            request_header=request_header,
            payload=None,  # Allowed and can be passed separately
        )

        # Verify response - result is a list of return values
        assert isinstance(result, list)
        assert len(result) > 0
        # The result contains the function's return value(s)
        # For server_add(10, 20), result should be [30]
        assert result[0] == 30

    finally:
        client.stop()
        server.stop()


@pytest.mark.asyncio
async def test_invoke_with_request_header_positional_args():
    """Test that request_header mode only allows worker_name as positional argument."""
    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        request_header = axon.RpcRequestHeader()
        request_header.function_id = 1
        request_header.session_id = 0
        request_header.workflow_id = 0

        # Test: Only worker_name allowed as positional argument
        # This should work
        try:
            await client.invoke_raw(
                "server_worker",  # worker_name as positional arg
                request_header=request_header,
            )
        except RuntimeError as e:
            # Connection errors are OK, we're just testing argument parsing
            if "disallowed argument" in str(e) or "positional argument" in str(e):
                pytest.fail(f"Unexpected argument error: {e}")

        # Test: Additional positional arguments should be rejected
        with pytest.raises(TypeError):
            await client.invoke_raw(
                "server_worker",
                0,  # Additional positional arg - should be rejected
                request_header=request_header,
            )

    finally:
        client.stop()


@pytest.mark.asyncio
async def test_invoke_raw_missing_header():
    """Test that invoke_raw fails if request_header is missing."""
    client = axon.AxonRuntime("client_worker")
    client.start_client()
    try:
        # invoke_raw now strictly requires request_header
        with pytest.raises(TypeError):
            await client.invoke_raw(worker_name="server_worker")
    finally:
        client.stop()


@pytest.mark.asyncio
async def test_invoke_without_request_header_positional_args():
    """Test invoking RPC using invoke() with positional arguments."""
    # Create server and register function
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

    # Create client
    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        # Connect to server
        await client.connect_endpoint_async(server_address, "server_worker")

        # Test: invoke(worker_name, function_id, *args, **kwargs)
        # worker_name=server_worker, function_id=1, args=(10, 20)
        result = await client.invoke(
            10,
            20,
            worker_name="server_worker",
            session_id=0,
            function_id=1,
            workflow_id=0,
        )

        # Verify response - result is a list of return values
        assert isinstance(result, list)
        assert len(result) > 0
        # The result contains the function's return value(s)
        # For server_add(10, 20), result should be [30]
        assert result[0] == 30

    finally:
        client.stop()
        server.stop()


@pytest.mark.asyncio
async def test_invoke_with_keyword_function_args_fails():
    """Test invoking RPC using invoke() with keyword function args fails."""
    # Create server and register function
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

    # Create client
    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        # Connect to server
        await client.connect_endpoint_async(server_address, "server_worker")

        # Test: invoke(args..., worker_name=..., function_id=...)
        # Note: function args must be positional based on current binding implementation
        # Passing them as kwargs (arg1, arg2) should fail with TypeError
        with pytest.raises(TypeError):
            await client.invoke(
                arg1=10,
                arg2=20,
                worker_name="server_worker",
                session_id=0,
                function_id=1,
                workflow_id=0,
            )

    finally:
        client.stop()
        server.stop()


@pytest.mark.asyncio
async def test_invoke_without_request_header_mixed_system_and_function_args():
    """Test invoking RPC using invoke(): system params as kwargs + function args as positional."""
    # Create server and register function
    server = axon.AxonRuntime("server_worker")
    server.start_server()

    server.register_function_raw(
        function_id=1,
        function_name="server_mixed",
        param_types=[
            axon.ParamType.PRIMITIVE_INT32,
            axon.ParamType.PRIMITIVE_FLOAT64,
            axon.ParamType.STRING,
        ],
        return_types=[axon.ParamType.STRING],
        input_payload_type=axon.PayloadType.NO_PAYLOAD,
        return_payload_type=axon.PayloadType.NO_PAYLOAD,
        callable=server_mixed,
    )

    server_address = server.get_local_address()

    # Create client
    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        # Connect to server
        await client.connect_endpoint_async(server_address, "server_worker")

        # Test: invoke(worker_name, function_id, *args, **kwargs)
        result = await client.invoke(
            # Function args as positional
            42,  # arg1: int
            3.14,  # arg2: float
            "hello",  # arg3: str
            worker_name="server_worker",
            session_id=0,
            function_id=1,
            workflow_id=100,  # workflow_id as kwargs
        )

        # Verify response
        assert isinstance(result, list)
        assert len(result) > 0
        # The result contains the function's return value(s)
        # For server_mixed(42, "hello", 3.14), result should be ["hello: 42 + 3.14"]
        assert isinstance(result[0], str)
        assert "hello" in result[0]
        assert "42" in result[0]

    finally:
        client.stop()
        server.stop()


@pytest.mark.asyncio
async def test_invoke_without_request_header_with_memory_policy():
    """Test invoking RPC using invoke() with memory_policy."""
    # Create server and register function
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

    # Create client
    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        # Connect to server
        await client.connect_endpoint_async(server_address, "server_worker")

        # Test: invoke(worker, function, *args, memory_policy=...)
        result = await client.invoke(
            10,  # arg1
            20,  # arg2
            worker_name="server_worker",
            session_id=0,
            function_id=1,
            workflow_id=0,
            memory_policy=None,
        )

        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0] == 30

    finally:
        client.stop()
        server.stop()


@pytest.mark.asyncio
async def test_invoke_with_explicit_payload_error():
    """Test that invoke() raises error if payload is passed explicitly."""
    client = axon.AxonRuntime("client_worker")
    client.start_client()
    try:
        with pytest.raises(TypeError):
            await client.invoke(
                worker_name="server_worker",
                session_id=0,
                function_id=1,
                payload="some_payload",
            )
    finally:
        client.stop()
