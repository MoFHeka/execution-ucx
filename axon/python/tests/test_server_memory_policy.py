import pytest
import numpy as np
import axon

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
    for i, meta in enumerate(meta_list):
        # Calculate size/shape
        shape = tuple(meta.shape)
        # Note: dtype in meta is DLDataType, need to map to numpy dtype if needed
        # For simplicity in this test, we assume float32 or int64 based on the strict test case
        # But we can just use Empty to allocate memory

        # Allocate our special array
        # Create a TrackedArray. We use standard numpy allocation and view casting
        # To simulate a real "allocation" based on shape.
        arr = np.zeros(shape, dtype=np.float32).view(TrackedArray)

        # Store strict identity
        created_objects[id(arr)] = True

        results.append(arr)

    if len(results) == 1:
        return results[0]
    return results


async def rpc_handler(arr: np.ndarray) -> bool:
    """
    RPC function that receives the array.
    We check if 'arr' is exactly the one we created AND if data is correct.
    """
    # Check if it is our custom type
    if isinstance(arr, TrackedArray):
        # Verify identity
        if id(arr) in created_objects:
            # Verify data correctness (expecting all 1.0s)
            if np.allclose(arr, 1.0):
                return True
            else:
                print(f"Data verification failed! Mean: {np.mean(arr)}")
                return False

    # Fallback/Failure case
    return False


@pytest.mark.asyncio
async def test_custom_memory_policy_reuse():
    # Clear registry
    created_objects.clear()

    # 1. Start Server
    server = axon.AxonRuntime("test_memory_policy_worker")
    server.start()

    # 2. Register function with custom memory policy
    server.register_function(
        rpc_handler,
        0,
        memory_policy=custom_policy,
        from_dlpack_fn=np.from_dlpack,  # Required by validation
    )

    server_addr = server.get_local_address()

    # 3. Create Client
    client = axon.AxonRuntime("client_worker_memfail")
    client.start_client()

    # Connect
    await client.connect_endpoint_async(server_addr, "test_memory_policy_worker")

    # 4. Invoke RPC with a large tensor to force Rendezvous
    large_arr = np.ones((1024, 1024), dtype=np.float32)  # 4MB, all 1.0s

    # Invoke needs worker_name, function_id(0), session_id(0).
    # And we expect a boolean result.
    is_reused = await client.invoke(
        large_arr, worker_name="test_memory_policy_worker", session_id=0, function=0
    )

    assert (
        is_reused == True
    ), "The server did not receive the reused object from the policy or data verification failed!"

    # 5. Invoke AGAIN to verify the memory policy is reusable (i.e. not moved-from)
    is_reused_2 = await client.invoke(
        large_arr, worker_name="test_memory_policy_worker", session_id=0, function=0
    )
    assert (
        is_reused_2 == True
    ), "The server failed on second invocation (memory policy reuse failed)"

    # Verify cleanup
    try:
        client.stop()
        server.stop()
    except:
        pass
