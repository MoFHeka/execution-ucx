"""Integration tests for Axon Runtime."""

import asyncio
import json
import pytest

# Import test_utils first to setup module path
import test_utils  # noqa: F401

import axon

# #region agent log
LOG_PATH = "/home/hejia/Documents/execution-ucx/.cursor/debug.log"


def debug_log(location, message, data=None, hypothesis_id=None):
    """Write debug log entry."""
    try:
        with open(LOG_PATH, "a") as f:
            entry = {
                "location": location,
                "message": message,
                "data": data or {},
                "timestamp": (
                    asyncio.get_event_loop().time()
                    if hasattr(asyncio, "get_event_loop")
                    else 0
                ),
                "sessionId": "debug-session",
                "runId": "run1",
            }
            if hypothesis_id:
                entry["hypothesisId"] = hypothesis_id
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


# #endregion


async def server_add(x: int, y: int) -> int:
    """Server add function."""
    # #region agent log
    debug_log(
        "test_integration.py:server_add",
        "Function started",
        {"x": x, "y": y},
        "PYTHON_TASK",
    )
    # #endregion
    result = x + y
    # #region agent log
    debug_log(
        "test_integration.py:server_add",
        "Function completed",
        {"result": result},
        "PYTHON_TASK",
    )
    # #endregion
    return result


async def server_multiply(x: int, y: int) -> int:
    """Server multiply function."""
    # #region agent log
    debug_log(
        "test_integration.py:server_multiply",
        "Function started",
        {"x": x, "y": y},
        "PYTHON_TASK",
    )
    # #endregion
    result = x * y
    # #region agent log
    debug_log(
        "test_integration.py:server_multiply",
        "Function completed",
        {"result": result},
        "PYTHON_TASK",
    )
    # #endregion
    return result


async def server_greet(name: str) -> str:
    """Server greet function."""
    # #region agent log
    debug_log(
        "test_integration.py:server_greet",
        "Function started",
        {"name": name},
        "PYTHON_TASK",
    )
    # #endregion
    result = f"Hello, {name}!"
    # #region agent log
    debug_log(
        "test_integration.py:server_greet",
        "Function completed",
        {"result": result},
        "PYTHON_TASK",
    )
    # #endregion
    return result


@pytest.mark.asyncio
async def test_complete_server_client_interaction():
    """Test complete server-client interaction with multiple functions."""
    # #region agent log
    debug_log("test_integration.py:31", "Test started", {}, "A")
    # #endregion
    # Setup server
    server = axon.AxonRuntime("server_worker")
    # #region agent log
    debug_log("test_integration.py:34", "Server created", {}, "A")
    # #endregion
    server.start_server()
    # #region agent log
    debug_log("test_integration.py:36", "Server started", {}, "A")
    # #endregion

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
        function_name="server_multiply",
        param_types=[
            axon.ParamType.PRIMITIVE_INT32,
            axon.ParamType.PRIMITIVE_INT32,
        ],
        return_types=[axon.ParamType.PRIMITIVE_INT32],
        input_payload_type=axon.PayloadType.NO_PAYLOAD,
        return_payload_type=axon.PayloadType.NO_PAYLOAD,
        callable=server_multiply,
    )

    server.register_function_raw(
        function_id=3,
        function_name="server_greet",
        param_types=[axon.ParamType.STRING],
        return_types=[axon.ParamType.STRING],
        input_payload_type=axon.PayloadType.NO_PAYLOAD,
        return_payload_type=axon.PayloadType.NO_PAYLOAD,
        callable=server_greet,
    )

    server_address = server.get_local_address()

    # Setup client
    client = axon.AxonRuntime("client_worker")
    # #region agent log
    debug_log("test_integration.py:76", "Client created", {}, "A")
    # #endregion
    client.start_client()
    # #region agent log
    debug_log("test_integration.py:78", "Client started", {}, "A")
    # #endregion

    try:
        # Connect to server
        # #region agent log
        debug_log("test_integration.py:82", "Before connect_endpoint_async", {}, "A")
        # #endregion
        conn_id = await client.connect_endpoint_async(server_address, "server_worker")
        # #region agent log
        debug_log(
            "test_integration.py:85",
            "After connect_endpoint_async",
            {"conn_id": conn_id},
            "A",
        )
        # #endregion
        assert isinstance(conn_id, int)

        # Call all registered functions
        # Function 1: add
        request_header1 = axon.RpcRequestHeader()
        request_header1.function_id = 1
        request_header1.session_id = 0
        request_header1.request_id = 1
        request_header1.workflow_id = 0
        request_header1.AddParam(10)  # x parameter
        request_header1.AddParam(20)  # y parameter

        # #region agent log
        debug_log("test_integration.py:93", "Before invoke_raw 1", {}, "B")
        # #endregion
        result1 = await client.invoke_raw(
            "server_worker",
            request_header=request_header1,
        )
        # #region agent log
        debug_log(
            "test_integration.py:101",
            "After invoke_raw 1",
            {"result": result1},
            "B",
        )
        # #endregion
        # Result is a list, get first element
        assert isinstance(result1, list)
        assert len(result1) == 1
        assert result1[0] == 30  # 10 + 20 = 30

        # Function 2: multiply
        request_header2 = axon.RpcRequestHeader()
        request_header2.function_id = 2
        request_header2.session_id = 0
        request_header2.request_id = 2
        request_header2.workflow_id = 0
        request_header2.AddParam(5)  # x parameter
        request_header2.AddParam(6)  # y parameter

        result2 = await client.invoke_raw(
            worker_name="server_worker",
            request_header=request_header2,
        )
        # Result is a list, get first element
        assert isinstance(result2, list)
        assert len(result2) == 1
        assert result2[0] == 30  # 5 * 6 = 30

        # Function 3: greet
        request_header3 = axon.RpcRequestHeader()
        request_header3.function_id = 3
        request_header3.session_id = 0
        request_header3.request_id = 3
        request_header3.workflow_id = 0
        request_header3.AddParam("World")  # name parameter

        result3 = await client.invoke_raw(
            worker_name="server_worker",
            request_header=request_header3,
        )
        # Result is a list, get first element
        assert isinstance(result3, list)
        assert len(result3) == 1
        assert result3[0] == "Hello, World!"
        # #region agent log
        debug_log(
            "test_integration.py:132",
            "All RPC calls completed, entering finally",
            {},
            "D",
        )
        # #endregion

    finally:
        # #region agent log
        debug_log("test_integration.py:136", "Before client.stop()", {}, "D")
        # #endregion
        client.stop()
        # #region agent log
        debug_log("test_integration.py:139", "After client.stop()", {}, "D")
        # #endregion
        # #region agent log
        debug_log("test_integration.py:141", "Before server.stop()", {}, "D")
        # #endregion
        server.stop()
        # #region agent log
        debug_log("test_integration.py:144", "After server.stop()", {}, "D")
        # #endregion


@pytest.mark.asyncio
async def test_signature_based_complete_workflow():
    """Test complete workflow based on signatures (service discovery simulation)."""
    # Setup server
    server = axon.AxonRuntime("server_worker")
    server.start_server()

    # Register functions
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

    # Server gets signatures (simulating service discovery)
    server_address = server.get_local_address()
    server_signatures = server.get_local_signatures()
    assert isinstance(server_signatures, bytes)
    assert len(server_signatures) > 0

    # Setup client
    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        # Client connects to server
        conn_id = await client.connect_endpoint_async(server_address, "server_worker")
        assert isinstance(conn_id, int)

        # Client receives signatures (simulating service discovery)
        # Deserialize signatures to get function info
        signatures = axon.deserialize_signatures(server_signatures)
        assert len(signatures) == 2

        # Find function signatures by function_id
        sig1 = None
        sig2 = None
        for sig in signatures:
            if sig.id == 1:
                sig1 = sig
            elif sig.id == 2:
                sig2 = sig

        assert sig1 is not None
        assert sig2 is not None
        assert sig1.function_name == "server_add"
        assert sig2.function_name == "server_greet"

        # Based on signatures, client calls all registered functions
        # Function 1: add (INT32, INT32) -> INT32
        request_header1 = axon.RpcRequestHeader()
        request_header1.function_id = sig1.id
        request_header1.session_id = 0
        request_header1.request_id = 1
        request_header1.workflow_id = 0
        # Add parameters based on signature
        request_header1.AddParam(10)  # x parameter
        request_header1.AddParam(20)  # y parameter

        result1 = await client.invoke_raw(
            worker_name="server_worker",
            request_header=request_header1,
        )
        # Result is a list, get first element
        assert isinstance(result1, list)
        assert len(result1) == 1
        assert result1[0] == 30  # 10 + 20 = 30

        # Function 2: greet (STRING) -> STRING
        request_header2 = axon.RpcRequestHeader()
        request_header2.function_id = sig2.id
        request_header2.session_id = 0
        request_header2.request_id = 2
        request_header2.workflow_id = 0
        # Add parameter based on signature
        request_header2.AddParam("World")  # name parameter

        result2 = await client.invoke_raw(
            worker_name="server_worker",
            request_header=request_header2,
        )
        # Result is a list, get first element
        assert isinstance(result2, list)
        assert len(result2) == 1
        assert result2[0] == "Hello, World!"

    finally:
        client.stop()
        server.stop()


@pytest.mark.asyncio
async def test_concurrent_rpc_calls():
    """Test multiple concurrent RPC calls."""
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

        # Create multiple concurrent calls
        async def make_call(request_id):
            request_header = axon.RpcRequestHeader()
            request_header.function_id = 1
            request_header.session_id = 0
            request_header.request_id = request_id
            request_header.workflow_id = 0
            # Add parameters for server_add (needs 2 INT32 parameters)
            request_header.AddParam(10)  # x parameter
            request_header.AddParam(20)  # y parameter

            result = await client.invoke_raw(
                worker_name="server_worker",
                request_header=request_header,
            )
            return result

        # Make 5 concurrent calls
        tasks = [make_call(i) for i in range(1, 6)]
        results = await asyncio.gather(*tasks)

        # Verify all calls completed
        assert len(results) == 5
        for result in results:
            # Result is a list, get first element
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0] == 30  # 10 + 20 = 30

    finally:
        client.stop()
        server.stop()


@pytest.mark.asyncio
async def test_error_unregistered_function():
    """Test calling unregistered function."""
    server = axon.AxonRuntime("server_worker")
    server.start_server()

    # Don't register any function with ID 999
    server_address = server.get_local_address()
    client = axon.AxonRuntime("client_worker")
    client.start_client()

    try:
        await client.connect_endpoint_async(server_address, "server_worker")

        # Try to call unregistered function
        request_header = axon.RpcRequestHeader()
        request_header.function_id = 999  # Not registered
        request_header.session_id = 0
        request_header.request_id = 1
        request_header.workflow_id = 0

        try:
            result = await client.invoke_raw(
                worker_name="server_worker",
                request_header=request_header,
            )
            # If we get here, result should be a list (even if empty or with error)
            # Payload has been converted to dltensor and is part of the result list
            assert isinstance(result, list)
        except Exception:
            # Expected to fail
            pass

    finally:
        client.stop()
        server.stop()


@pytest.mark.asyncio
async def test_error_server_stopped():
    """Test calling after server is stopped."""
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
        # #region agent log
        debug_log("test_integration.py:436", "After connect_endpoint_async", {}, "H1")
        # #endregion

        # Stop server
        # #region agent log
        debug_log("test_integration.py:439", "Before server.stop()", {}, "H1")
        # #endregion
        server.stop()
        # #region agent log
        debug_log("test_integration.py:441", "After server.stop()", {}, "H1")
        # #endregion

        # Try to call after server stopped
        request_header = axon.RpcRequestHeader()
        request_header.function_id = 1
        request_header.session_id = 0
        request_header.request_id = 1
        request_header.workflow_id = 0
        # Add parameters for server_add (needs 2 INT32 parameters)
        request_header.AddParam(10)  # x parameter
        request_header.AddParam(20)  # y parameter

        try:
            # #region agent log
            debug_log(
                "test_integration.py:454",
                "Before invoke_raw (server stopped)",
                {},
                "H1",
            )
            # #endregion
            # Store the Future object so we can clean it up explicitly
            future = client.invoke_raw(
                worker_name="server_worker",
                request_header=request_header,
            )
            # #region agent log
            debug_log(
                "test_integration.py:after_invoke",
                "Got future from invoke_raw",
                {
                    "future_type": str(type(future)),
                    "future_done": future.done() if hasattr(future, "done") else None,
                },
                "H1",
            )
            # #endregion
            # Await the future - this will raise an exception
            result = await future
            # #region agent log
            debug_log(
                "test_integration.py:460",
                "After invoke_raw (unexpected success)",
                {},
                "H1",
            )
            # #endregion
            # If we get here, result should be a list
            # Payload has been converted to dltensor and is part of the result list
            assert isinstance(result, list)
        except Exception as e:
            # #region agent log
            debug_log(
                "test_integration.py:466",
                "Exception in invoke_raw (expected)",
                {"error": str(e)},
                "H1",
            )
            # #endregion
            # Expected to fail
            pass

        # #region agent log
        # Check if Future object still exists and holds references to event loop
        try:
            import gc
            import sys

            loop = asyncio.get_running_loop()

            # Check all tasks
            all_tasks = asyncio.all_tasks(loop)
            debug_log(
                "test_integration.py:after_exception",
                "Tasks after exception",
                {"count": len(all_tasks), "task_reprs": [str(t) for t in all_tasks]},
                "H1",
            )

            # Try to find Future objects that might hold references to the loop
            # Check if 'future' variable exists (it should exist if we stored it)
            try:
                if "future" in locals():
                    debug_log(
                        "test_integration.py:after_exception",
                        "Found 'future' variable (Future object)",
                        {
                            "future_type": str(type(future)),
                            "future_done": (
                                future.done() if hasattr(future, "done") else None
                            ),
                            "future_cancelled": (
                                future.cancelled()
                                if hasattr(future, "cancelled")
                                else None
                            ),
                        },
                        "H1",
                    )
                    # Try to cancel the Future if it's not done
                    if hasattr(future, "cancel") and not future.done():
                        cancelled = future.cancel()
                        debug_log(
                            "test_integration.py:after_exception",
                            "Attempted to cancel Future",
                            {"cancelled": cancelled, "future_done": future.done()},
                            "H1",
                        )
                    # Explicitly clear the Future reference to help with cleanup
                    del future
                    debug_log(
                        "test_integration.py:after_exception",
                        "Deleted 'future' variable",
                        {},
                        "H1",
                    )
            except NameError:
                # 'future' doesn't exist
                debug_log(
                    "test_integration.py:after_exception",
                    "'future' variable does not exist",
                    {},
                    "H1",
                )

            # Force garbage collection to clean up any remaining Future references
            collected = gc.collect()
            debug_log(
                "test_integration.py:after_exception",
                "After garbage collection",
                {"collected": collected},
                "H1",
            )
        except Exception as gc_e:
            debug_log(
                "test_integration.py:after_exception",
                "Error checking Future references",
                {"error": str(gc_e), "error_type": type(gc_e).__name__},
                "H1",
            )
        # #endregion

    finally:
        client.stop()
        # #region agent log
        debug_log(
            "test_integration.py:finally",
            "Client stopped, test finished",
            {},
            "H2",
        )
        # #endregion


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
