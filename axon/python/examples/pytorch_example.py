import torch
import axon
import asyncio
import numpy as np

# Global registry to track objects created by the policy
created_objects = {}


def custom_torch_policy(meta_list):
    """
    Custom memory policy that:
    1. Receives a list of TensorMeta
    2. Allocates a torch.Tensor for each meta
    3. Registries the tensor to verify reuse
    """
    assert isinstance(meta_list, list)
    assert len(meta_list) > 0

    results = []
    created_objects.clear()

    for i, meta in enumerate(meta_list):
        shape = tuple(meta.shape)
        # Allocate a torch tensor
        # Note: We assume float32 for this example
        tensor = torch.zeros(shape, dtype=torch.float32)

        # Store strict identity and keep object alive
        created_objects[id(tensor)] = tensor
        results.append(tensor)

    if len(results) == 1:
        return results[0]
    return results


async def tensor_op_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Simple tensor operation function.
    """
    print(f"Server received tensors: type(a)={type(a)}, type(b)={type(b)}")
    return a + b


async def main():
    # 1. Start Server
    server = axon.AxonRuntime("torch_worker")
    server.start()

    # 2. Register function using torch.from_dlpack for conversion
    server.register_function(
        0, tensor_op_func, function_name="tensor_op", from_dlpack_fn=torch.from_dlpack
    )

    server_addr = server.get_local_address()
    print(f"Server started at {server_addr}")

    # 3. Create Client
    client = axon.AxonRuntime("client_worker")
    client.start_client()

    # Connect
    await client.connect_endpoint_async(server_addr, "torch_worker")

    # 4. Prepare data
    # Create large tensors to potentially trigger RNDV path (though policy works regardless if triggered)
    # Using smaller ones for this example, but large enough to be interesting.
    a = torch.ones((1024, 1024), dtype=torch.float32)
    b = torch.ones((1024, 1024), dtype=torch.float32) * 2

    print("Invoking remote function with torch tensors (RNDV Path - Large Tensors)...")

    # 5. Invoke RPC with custom memory policy (RNDV Path)
    # Large tensors should trigger RNDV and thus use the memory policy
    created_objects.clear()
    result_rndv = await client.invoke(
        a,
        b,
        worker_name="torch_worker",
        session_id=0,
        function_id=0,
        memory_policy=custom_torch_policy,
        from_dlpack_fn=torch.from_dlpack,
    )

    print(f"RNDV Result received: type={type(result_rndv)}")

    # Verify RNDV result
    assert isinstance(result_rndv, torch.Tensor), "Result should be a torch.Tensor"
    # RNDV path MUST use the policy
    if id(result_rndv) in created_objects:
        print(
            "RNDV Path Verification SUCCESS: Result matches object created by policy."
        )
    else:
        print(
            "RNDV Path Verification FAILED: Result identity does not match object created by policy."
        )

    expected = a + b
    if torch.allclose(result_rndv, expected):
        print("RNDV Data Verification SUCCESS.")
    else:
        print("RNDV Data Verification FAILED.")

    print("-" * 20)
    print("Invoking remote function with torch tensors (Eager Path - Small Tensors)...")

    # 6. Invoke RPC with custom memory policy (Eager Path)
    # Small tensors should use Eager path and IGNORE the memory policy (fallback to from_dlpack)
    small_a = torch.ones((10, 10), dtype=torch.float32)
    small_b = torch.ones((10, 10), dtype=torch.float32) * 2

    created_objects.clear()
    result_eager = await client.invoke(
        small_a,
        small_b,
        worker_name="torch_worker",
        session_id=0,
        function_id=0,
        memory_policy=custom_torch_policy,
        from_dlpack_fn=torch.from_dlpack,
    )

    print(f"Eager Result received: type={type(result_eager)}")

    # Verify Eager result
    assert isinstance(result_eager, torch.Tensor), "Result should be a torch.Tensor"

    # Eager path should NOT use the policy (it uses internal buffer + from_dlpack)
    # So the object should NOT be in created_objects
    if id(result_eager) not in created_objects:
        print(
            "Eager Path Verification SUCCESS: Result was NOT created by policy (expected behavior)."
        )
    else:
        print(
            "Eager Path Verification FAILED: Result WAS created by policy (unexpected for Eager path)."
        )

    expected_small = small_a + small_b
    if torch.allclose(result_eager, expected_small):
        print("Eager Data Verification SUCCESS.")
    else:
        print("Eager Data Verification FAILED.")

    # Cleanup
    try:
        client.stop()
        server.stop()
    except Exception as e:
        print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    asyncio.run(main())
