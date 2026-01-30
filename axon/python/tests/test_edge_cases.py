"""Edge cases and error handling tests."""

import pytest

# Import test_utils first to setup module path
import test_utils  # noqa: F401

import axon


async def dummy_func(x: int) -> int:
    """Dummy function for testing."""
    return x * 2


def test_invalid_worker_name_empty():
    """Test with empty worker name."""
    # Empty string might be allowed or might raise exception
    try:
        runtime = axon.AxonRuntime("")
        runtime.stop()
    except Exception:
        # May or may not be allowed
        pass


def test_invalid_function_id_zero():
    """Test registering function with function_id=0."""
    runtime = axon.AxonRuntime("test_worker")
    runtime.start_server()

    try:
        # function_id=0 might be valid or invalid
        runtime.register_function_raw(
            function_id=0,
            function_name="dummy_func",
            param_types=[axon.ParamType.PRIMITIVE_INT32],
            return_types=[axon.ParamType.PRIMITIVE_INT32],
            input_payload_type=axon.PayloadType.NO_PAYLOAD,
            return_payload_type=axon.PayloadType.NO_PAYLOAD,
            callable=dummy_func,
        )
    except Exception:
        # May or may not be allowed
        pass

    runtime.stop()


def test_invalid_function_id_large():
    """Test registering function with very large function_id."""
    runtime = axon.AxonRuntime("test_worker")
    runtime.start_server()

    try:
        # Very large function_id
        runtime.register_function_raw(
            function_id=0xFFFFFFFF,
            function_name="dummy_func",
            param_types=[axon.ParamType.PRIMITIVE_INT32],
            return_types=[axon.ParamType.PRIMITIVE_INT32],
            input_payload_type=axon.PayloadType.NO_PAYLOAD,
            return_payload_type=axon.PayloadType.NO_PAYLOAD,
            callable=dummy_func,
        )
    except Exception:
        # May or may not be allowed
        pass

    runtime.stop()


def test_invalid_param_types_mismatch():
    """Test registering function with mismatched param types."""
    runtime = axon.AxonRuntime("test_worker")
    runtime.start_server()

    # This might not be caught at registration time
    # but should fail at runtime
    try:
        runtime.register_function_raw(
            function_id=1,
            function_name="dummy_func",
            param_types=[
                axon.ParamType.PRIMITIVE_INT32,
                axon.ParamType.STRING,  # Mismatch with actual function
            ],
            return_types=[axon.ParamType.PRIMITIVE_INT32],
            input_payload_type=axon.PayloadType.NO_PAYLOAD,
            return_payload_type=axon.PayloadType.NO_PAYLOAD,
            callable=dummy_func,  # Only takes one int parameter
        )
        # Registration might succeed, but RPC call should fail
    except Exception:
        # May fail at registration or runtime
        pass

    runtime.stop()


def test_resource_cleanup_on_destructor():
    """Test resource cleanup when Runtime is destroyed."""
    runtime = axon.AxonRuntime("test_worker")
    runtime.start()

    # Don't explicitly stop - let destructor handle it
    # Runtime should clean up resources
    del runtime


def test_stop_without_start():
    """Test stopping without starting."""
    runtime = axon.AxonRuntime("test_worker")

    # Should handle gracefully
    try:
        runtime.stop()
    except Exception:
        # May or may not raise
        pass


def test_multiple_stops():
    """Test calling stop multiple times."""
    runtime = axon.AxonRuntime("test_worker")
    runtime.start()
    runtime.stop()

    # Should handle gracefully
    try:
        runtime.stop()
        runtime.stop()
    except Exception:
        # May or may not raise
        pass


def test_register_duplicate_function_id():
    """Test registering function with duplicate function_id."""
    runtime = axon.AxonRuntime("test_worker")
    runtime.start_server()

    # Register first function
    runtime.register_function_raw(
        function_id=1,
        function_name="func1",
        param_types=[axon.ParamType.PRIMITIVE_INT32],
        return_types=[axon.ParamType.PRIMITIVE_INT32],
        input_payload_type=axon.PayloadType.NO_PAYLOAD,
        return_payload_type=axon.PayloadType.NO_PAYLOAD,
        callable=dummy_func,
    )

    # Try to register with same function_id
    try:
        runtime.register_function_raw(
            function_id=1,  # Duplicate
            function_name="func2",
            param_types=[axon.ParamType.PRIMITIVE_INT32],
            return_types=[axon.ParamType.PRIMITIVE_INT32],
            input_payload_type=axon.PayloadType.NO_PAYLOAD,
            return_payload_type=axon.PayloadType.NO_PAYLOAD,
            callable=dummy_func,
        )
        # May overwrite or raise exception
    except Exception:
        # Expected if duplicate IDs not allowed
        pass

    runtime.stop()


def test_empty_param_types():
    """Test registering function with empty param_types."""

    async def no_params() -> int:
        return 42

    runtime = axon.AxonRuntime("test_worker")
    runtime.start_server()

    runtime.register_function_raw(
        function_id=1,
        function_name="no_params",
        param_types=[],  # Empty
        return_types=[axon.ParamType.PRIMITIVE_INT32],
        input_payload_type=axon.PayloadType.NO_PAYLOAD,
        return_payload_type=axon.PayloadType.NO_PAYLOAD,
        callable=no_params,
    )

    runtime.stop()


def test_empty_return_types():
    """Test registering function with empty return_types."""

    async def void_func() -> None:
        pass

    runtime = axon.AxonRuntime("test_worker")
    runtime.start_server()

    try:
        runtime.register_function_raw(
            function_id=1,
            function_name="void_func",
            param_types=[],
            return_types=[],  # Empty or VOID
            input_payload_type=axon.PayloadType.NO_PAYLOAD,
            return_payload_type=axon.PayloadType.NO_PAYLOAD,
            callable=void_func,
        )
    except Exception:
        # May require at least one return type or VOID
        pass

    runtime.stop()


def test_invalid_payload_type_combination():
    """Test invalid payload type combinations."""
    runtime = axon.AxonRuntime("test_worker")
    runtime.start_server()

    # Try invalid combinations
    try:
        runtime.register_function_raw(
            function_id=1,
            function_name="dummy_func",
            param_types=[axon.ParamType.PRIMITIVE_INT32],
            return_types=[axon.ParamType.PRIMITIVE_INT32],
            input_payload_type=axon.PayloadType.UCX_BUFFER,
            return_payload_type=axon.PayloadType.NO_PAYLOAD,
            callable=dummy_func,  # No tensor params
        )
        # May or may not be valid
    except Exception:
        # May fail if payload type doesn't match function signature
        pass

    runtime.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
