import jax

# Force usage of CUDA
jax.config.update("jax_platform_name", "cuda")

import jax.numpy as jnp
import axon
import asyncio
from jax import dlpack as jdlpack

# Global registry to track objects created by the policy
created_objects = {}


def custom_jax_policy(meta_list):
    """
    Custom memory policy that:
    1. Receives a list of TensorMeta
    2. Allocates a jax.Array for each meta
    3. Registries the array to verify reuse
    """
    assert isinstance(meta_list, list)
    assert len(meta_list) > 0

    results = []
    created_objects.clear()

    for i, meta in enumerate(meta_list):
        shape = tuple(meta.shape)
        # Allocate a jax array (lazy zero)
        # Note: We assume float32 for this example
        arr = jnp.zeros(shape, dtype=jnp.float32)

        # JAX arrays are immutable, but we can track them by ID if they are returned directly
        # However, checking identity might be tricky if JAX does some internal management.
        # But if the policy creates it and the RPC triggers it, it should return this exact object.

        # Store strict identity
        created_objects[id(arr)] = True
        results.append(arr)

    if len(results) == 1:
        return results[0]
    return results


def jax_from_dlpack(capsule):
    return jdlpack.from_dlpack(capsule)


async def jax_op_func(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Simple JAX operation function.
    """
    print(f"Server received arrays: type(a)={type(a)}, type(b)={type(b)}")
    return a + b


async def main():
    # Trigger JAX initialization (ensure context is created)
    _ = jnp.zeros((1,))

    print("Using CUDA device")

    # 1. Start Server with CUDA device
    server = axon.AxonRuntime("jax_worker", device=axon.cuda(), timeout=30.0)
    server.start()

    # 2. Register function using jax.dlpack.from_dlpack for conversion
    server.register_function(
        jax_op_func, from_dlpack_fn=jax_from_dlpack, memory_policy=custom_jax_policy
    )

    server_addr = server.get_local_address()
    print(f"Server started at {server_addr}")

    # 3. Create Client with CUDA device
    client = axon.AxonRuntime("client_worker_jax", device=axon.cuda(), timeout=30.0)
    client.start_client()

    # Connect
    await client.connect_endpoint_async(server_addr, "jax_worker")

    # 4. Prepare data
    a = jnp.ones((1024, 1024), dtype=jnp.float32)
    b = jnp.ones((1024, 1024), dtype=jnp.float32) * 2

    print("Invoking remote function with JAX arrays (RNDV Path - Large Arrays)...")

    # 5. Invoke RPC with custom memory policy (RNDV Path)
    created_objects.clear()
    result_rndv = await client.invoke(
        a,
        b,
        worker_name="jax_worker",
        session_id=0,
        function="jax_op_func",
        memory_policy=custom_jax_policy,
        from_dlpack_fn=jax_from_dlpack,
    )

    print(f"RNDV Result received: type={type(result_rndv)}")

    # Verify RNDV result
    assert isinstance(result_rndv, jax.Array), "Result should be a jax.Array"

    # Check identity if possible (JAX wrapping might interfere, but let's try)
    if id(result_rndv) in created_objects:
        print(
            "RNDV Path Verification SUCCESS: Result matches object created by policy."
        )
    else:
        print(
            "RNDV Path Verification WARNING: Result identity does not match. JAX might have wrapped it."
        )

    expected = a + b
    if jnp.allclose(result_rndv, expected):
        print("RNDV Data Verification SUCCESS.")
    else:
        print("RNDV Data Verification FAILED.")

    print("-" * 20)
    print("Invoking remote function with JAX arrays (Eager Path - Small Arrays)...")

    # 6. Invoke RPC with custom memory policy (Eager Path)
    small_a = jnp.ones((10, 10), dtype=jnp.float32)
    small_b = jnp.ones((10, 10), dtype=jnp.float32) * 2

    created_objects.clear()
    result_eager = await client.invoke(
        small_a,
        small_b,
        worker_name="jax_worker",
        session_id=0,
        function="jax_op_func",
        memory_policy=custom_jax_policy,
        from_dlpack_fn=jax_from_dlpack,
    )

    print(f"Eager Result received: type={type(result_eager)}")

    # Verify Eager result
    assert isinstance(result_eager, jax.Array), "Result should be a jax.Array"

    # Eager path should NOT use the policy
    if id(result_eager) not in created_objects:
        print(
            "Eager Path Verification SUCCESS: Result was NOT created by policy (expected behavior)."
        )
    else:
        print(
            "Eager Path Verification FAILED: Result WAS created by policy (unexpected for Eager path)."
        )

    expected_small = small_a + small_b
    if jnp.allclose(result_eager, expected_small):
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
