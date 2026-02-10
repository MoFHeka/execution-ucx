import pytest
import numpy as np
import axon
import asyncio

# Global registry to track objects created by the policy
created_objects = {}


class TrackedArray(np.ndarray):
    """A subclass of ndarray to easy identify our custom objects"""

    pass


def custom_policy(meta_list):
    """
    Custom memory policy that:
    1. Receives a list of TensorMeta
    2. Allocates a TrackedArray for each meta
    3. Registries the array to verify reuse
    """
    assert isinstance(meta_list, list)
    assert len(meta_list) > 0

    results = []
    created_objects.clear()  # Clear specific to this call ideally, but global for test simplicity

    for i, meta in enumerate(meta_list):
        shape = tuple(meta.shape)
        # Allocate our special array
        arr = np.zeros(shape, dtype=np.float32).view(TrackedArray)
        # Store strict identity
        created_objects[id(arr)] = True
        results.append(arr)

    if len(results) == 1:
        return results[0]
    return results


async def return_large_tensor() -> np.ndarray:
    """
    Returns a large tensor to trigger RNDV and memory policy on client side.
    """
    large_arr = np.ones((1024, 1024), dtype=np.float32)
    return large_arr


@pytest.mark.asyncio
async def test_invoke_memory_policy_rndv():
    # Clear registry
    created_objects.clear()

    # 1. Start Server
    server = axon.AxonRuntime("test_server_invoke_mem")
    server.start()

    # 2. Register function
    server.register_function(
        return_large_tensor, 0, function_name="return_large_tensor"
    )

    server_addr = server.get_local_address()

    # 3. Create Client
    client = axon.AxonRuntime("test_client_invoke_mem")
    client.start_client()

    # Connect
    await client.connect_endpoint_async(server_addr, "test_server_invoke_mem")

    # 4. Invoke RPC
    # We pass memory_policy to invoke.
    # We accept a large tensor return.

    # NOTE: from_dlpack_fn is required for the fallback path if RNDV is NOT used (e.g. if implementation decides so)
    # But for large tensor (4MB) it should use RNDV if configured correctly.
    # However, memory_policy is ONLY used for RNDV.

    # We call invoke
    result = await client.invoke(
        worker_name="test_server_invoke_mem",
        session_id=0,
        function=0,
        memory_policy=custom_policy,
        from_dlpack_fn=np.from_dlpack,
    )

    # 5. Verify result is the one created by policy
    assert isinstance(
        result, TrackedArray
    ), "Result should be instance of TrackedArray created by policy"
    assert (
        id(result) in created_objects
    ), "Result identity should match object created by policy"
    assert np.allclose(result, 1.0), "Data verification failed"

    # Verify cleanup
    try:
        client.stop()
        server.stop()
    except:
        pass
