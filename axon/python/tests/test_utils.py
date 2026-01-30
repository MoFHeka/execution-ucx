"""Test utilities for Axon Runtime tests."""

import os
import sys
import glob
import asyncio
from contextlib import contextmanager
from typing import Optional, Callable, List, Tuple


def setup_module_path():
    """Setup PYTHONPATH to find axon_python_runtime.so."""
    # Try to find axon_python_runtime.so in common locations
    possible_paths = [
        os.path.join(os.path.dirname(__file__), ".."),
        os.path.join(os.path.dirname(__file__), "..", ".."),
        os.path.dirname(__file__),
        os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "bazel-bin", "axon", "python"
        ),
    ]

    # Also check runfiles (Bazel)
    if "TEST_SRCDIR" in os.environ:
        runfiles_dir = os.environ["TEST_SRCDIR"]
        possible_paths.insert(0, runfiles_dir)
        possible_paths.insert(
            0, os.path.join(runfiles_dir, "execution_ucx", "axon", "python")
        )

    # Possible library names
    # Bazel generates libaxon_python_runtime.so, but we also have axon.so from genrule
    lib_names = [
        "axon.so",
        "libaxon_python_runtime.so",
        "axon_python_runtime.so",
    ]

    # Search for the .so file
    for path in possible_paths:
        for lib_name in lib_names:
            runtime_so = os.path.join(path, lib_name)
            if os.path.exists(runtime_so):
                # Always prefer adding the source directory (where axon.py lives)
                # to sys.path if it exists, so we use the wrapper which handles loading.
                source_dir = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..")
                )
                if os.path.exists(os.path.join(source_dir, "axon.py")):
                    if source_dir not in sys.path:
                        sys.path.insert(0, source_dir)
                    return

                # Fallback: add the directory containing the .so directly
                if path not in sys.path:
                    sys.path.insert(0, path)
                return

    # Try glob search as fallback
    for lib_name in lib_names:
        for pattern in [f"**/{lib_name}", lib_name]:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                # Prefer source dir if wrapper exists
                source_dir = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..")
                )
                if os.path.exists(os.path.join(source_dir, "axon.py")):
                    if source_dir not in sys.path:
                        sys.path.insert(0, source_dir)
                    return

                runtime_dir = os.path.dirname(os.path.abspath(matches[0]))
                if runtime_dir not in sys.path:
                    sys.path.insert(0, runtime_dir)
                return


# Setup module path before importing axon
# This allows tests to run directly without Bazel
setup_module_path()

# Try to import axon with helpful error message
try:
    import axon
except ImportError as e:
    error_msg = (
        "Failed to import axon module.\n"
        "Please ensure that axon_python_runtime.so has been built.\n"
        "Build it using: bazel build //axon/python:axon_python_runtime\n"
        "Or set PYTHONPATH to the directory containing axon_python_runtime.so\n"
        f"Original error: {e}"
    )
    raise ImportError(error_msg) from e


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


async def server_mixed(x: int, name: str, f: float) -> str:
    """Server function with mixed parameters."""
    return f"{name}: {x} + {f}"


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
    """Context manager for creating and cleaning up Runtime.

    Args:
        worker_name: Name of the worker
        thread_pool_size: Size of thread pool
        timeout: Timeout in milliseconds (int)
    """
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
    # Connect
    conn_id = await client.connect_endpoint_async(server_address, server_name)

    # Create request header if not provided
    if request_header is None:
        request_header = create_request_header(function_id, workflow_id=workflow_id)

    # Invoke RPC
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
