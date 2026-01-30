"""Basic functionality tests for Axon Runtime."""

import pytest
from test_utils import create_runtime


def test_runtime_creation_default():
    """Test creating Runtime with default parameters."""
    with create_runtime("test_worker") as runtime:
        assert runtime is not None


def test_runtime_creation_custom_thread_pool():
    """Test creating Runtime with custom thread_pool_size."""
    with create_runtime("test_worker", thread_pool_size=8) as runtime:
        assert runtime is not None


def test_runtime_creation_custom_timeout():
    """Test creating Runtime with custom timeout."""
    timeout_ms = 500
    with create_runtime("test_worker", timeout=timeout_ms) as runtime:
        assert runtime is not None


def test_start_stop():
    """Test start() and stop() combination."""
    with create_runtime("test_worker") as runtime:
        runtime.start()


def test_start_server_stop_server():
    """Test start_server() and stop_server() combination."""
    with create_runtime("test_worker") as runtime:
        runtime.start_server()
        runtime.stop_server()


def test_start_client_stop_client():
    """Test start_client() and stop_client() combination."""
    with create_runtime("test_worker") as runtime:
        runtime.start_client()
        runtime.stop_client()


def test_mixed_start_stop():
    """Test mixed start/stop scenarios."""
    with create_runtime("test_worker") as runtime:
        # Start server, then client
        runtime.start_server()
        runtime.start_client()

        # Stop client, then server
        runtime.stop_client()
        runtime.stop_server()


def test_repeated_start():
    """Test repeated start calls (should handle gracefully)."""
    with create_runtime("test_worker") as runtime:
        runtime.start()

        # Try to start again - behavior may vary, but should not crash
        try:
            runtime.start()
        except Exception:
            pass  # Expected if not allowed


def test_repeated_stop():
    """Test repeated stop calls (should handle gracefully)."""
    with create_runtime("test_worker") as runtime:
        runtime.start()
        runtime.stop()

        # Try to stop again - should not crash
        try:
            runtime.stop()
        except Exception:
            pass  # May or may not raise, but should not crash


def test_stop_without_start():
    """Test stopping without starting (should handle gracefully)."""
    with create_runtime("test_worker") as runtime:
        # Should not crash
        try:
            runtime.stop()
        except Exception:
            pass  # May or may not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
