import pytest
from axon import AxonRuntime


# Define a simple async function to be registered
async def simple_echo(x: int) -> int:
    return x


@pytest.mark.asyncio
async def test_invoke_by_name():
    worker_name = "test_worker_name_func"
    runtime = AxonRuntime(worker_name)
    runtime.start_server()
    runtime.start_client()

    try:
        # Register function without explicit ID (should use None -> auto-generate)
        # We pass function_id=None (or 0, but None triggers the logic)
        # Note: In Python binding, we made function_id optional.
        # But we need to see how to pass it as None.
        # The binding signature is `register_function(function_id, callable, ...)`
        # If we pass None for function_id, it should work.

        # Manually compute expected ID to verify later if needed,
        # but here we just test invocation.
        function_name = "simple_echo"

        # Register
        runtime.register_function(
            callable=simple_echo, function_id=None, function_name=function_name
        )

        # Connect to self
        await runtime.connect_endpoint_async(runtime.get_local_address(), worker_name)

        # Invoke by name
        input_val = 42
        session_id = 1
        workflow_id = 1

        # Invoke passing function_name as the function_id argument (which accepts variant)
        # Note: The signature is invoke(*args, worker_name=..., session_id=..., function=...)
        # input_val is the argument for the remote function
        result_future = runtime.invoke(
            input_val,
            worker_name=worker_name,
            session_id=session_id,
            function=function_name,  # Passing string here!
            workflow_id=workflow_id,
        )

        result = await result_future
        assert result == input_val
        print(f"Invoke by name '{function_name}' successful: {result}")

    finally:
        runtime.stop()


@pytest.mark.asyncio
async def test_invoke_by_name_implicit_name():
    worker_name = "test_worker_implicit"
    runtime = AxonRuntime(worker_name)
    runtime.start()  # Starts both server and client

    try:
        # Register function passing None for both ID and Name
        # Name should be inferred from __name__ ("simple_echo")
        # ID should be inferred from Name + WorkerName
        runtime.register_function(callable=simple_echo, function_id=None)

        await runtime.connect_endpoint_async(runtime.get_local_address(), worker_name)

        # Invoke by the inferred name "simple_echo"
        result = await runtime.invoke(
            100,
            worker_name=worker_name,
            session_id=1,
            function="simple_echo",
            workflow_id=1,
        )
        assert result == 100
        print(f"Invoke by implicit name 'simple_echo' successful: {result}")

    finally:
        runtime.stop()
