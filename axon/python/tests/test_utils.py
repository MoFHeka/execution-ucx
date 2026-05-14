"""Test utilities for Axon Runtime tests."""

from contextlib import contextmanager
from typing import Optional, Callable, List, Tuple

import axon


# Common async functions for testing
async def server_add(x: int, y: int) -> int:
    """Server add function."""
    return x + y


async def server_multiply(x: int, y: int) -> int:
    """Server multiply function."""
    return x * y


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


async def server_bool_func(b: bool) -> bool:
    """Server function with bool."""
    return not b


async def server_float_func(f: float) -> float:
    """Server function with float."""
    return f * 2.0


async def server_vector_func(vec: list) -> list:
    """Server function with vector."""
    return [x * 2 for x in vec]


async def dummy_func(x: int) -> int:
    """Dummy function for testing."""
    return x * 2


@contextmanager
def create_runtime(
    worker_name: str = "test_worker",
    thread_pool_size: Optional[int] = None,
    timeout: Optional[int] = None,
):
    """Context manager for creating and cleaning up Runtime."""
    kwargs = {}
    if thread_pool_size is not None:
        kwargs["thread_pool_size"] = thread_pool_size
    if timeout is not None:
        kwargs["timeout"] = timeout

    runtime = axon.AxonRuntime(worker_name, **kwargs)
    try:
        yield runtime
    finally:
        try:
            runtime.stop()
        except Exception:
            pass


@contextmanager
def create_server(worker_name: str = "server_worker"):
    """Context manager for creating and cleaning up server Runtime."""
    server = axon.AxonRuntime(worker_name)
    try:
        server.start_server()
        yield server
    finally:
        try:
            server.stop()
        except Exception:
            pass


@contextmanager
def create_client(worker_name: str = "client_worker"):
    """Context manager for creating and cleaning up client Runtime."""
    client = axon.AxonRuntime(worker_name)
    try:
        client.start_client()
        yield client
    finally:
        try:
            client.stop()
        except Exception:
            pass


def register_function_raw(
    runtime: axon.AxonRuntime,
    function_id: int,
    function_name: str,
    param_types: List[axon.ParamType],
    return_types: List[axon.ParamType],
    callable_func: Callable,
    input_payload_type: axon.PayloadType = axon.PayloadType.NO_PAYLOAD,
    return_payload_type: axon.PayloadType = axon.PayloadType.NO_PAYLOAD,
    memory_policy: Optional[object] = None,
    lifecycle_policy: Optional[object] = None,
):
    """Helper function to register a function on runtime."""
    runtime.register_function_raw(
        function_id=function_id,
        function_name=function_name,
        param_types=param_types,
        return_types=return_types,
        input_payload_type=input_payload_type,
        return_payload_type=return_payload_type,
        callable=callable_func,
        memory_policy=memory_policy,
        lifecycle_policy=lifecycle_policy,
    )


def create_request_header(
    function_id: int,
    session_id: int = 0,
    request_id: int = 1,
    workflow_id: int = 0,
) -> axon.RpcRequestHeader:
    """Helper function to create RpcRequestHeader."""
    header = axon.RpcRequestHeader()
    header.function_id = function_id
    header.session_id = session_id
    header.request_id = request_id
    header.workflow_id = workflow_id
    return header


async def connect_and_invoke(
    client: axon.AxonRuntime,
    server_address: bytes,
    server_name: str,
    function_id: int,
    request_header: Optional[axon.RpcRequestHeader] = None,
    workflow_id: int = 0,
    payload: Optional[object] = None,
    memory_policy: Optional[object] = None,
) -> Tuple[axon.RpcResponseHeader, object]:
    """Helper function to connect to server and invoke_raw RPC."""
    _ = await client.connect_endpoint_async(server_address, server_name)

    if request_header is None:
        request_header = create_request_header(function_id, workflow_id=workflow_id)

    result = await client.invoke_raw(
        worker_name=server_name,
        request_header=request_header,
        payload=payload,
        memory_policy=memory_policy,
    )

    return result


async def setup_server_client(
    server_worker_name: str = "server_worker",
    client_worker_name: str = "client_worker",
):
    """Setup server and client, return (server, client, server_address)."""
    server = axon.AxonRuntime(server_worker_name)
    server.start_server()

    server_address = server.get_local_address()

    client = axon.AxonRuntime(client_worker_name)
    client.start_client()

    return server, client, server_address


def cleanup_server_client(server: axon.AxonRuntime, client: axon.AxonRuntime):
    """Cleanup server and client."""
    try:
        client.stop()
    except Exception:
        pass
    try:
        server.stop()
    except Exception:
        pass
