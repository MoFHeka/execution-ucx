"""Zero-dimensional tensors contain one element, rather than zero bytes."""
import axon
import numpy as np
import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("dtype", [np.int32, np.float64])
async def test_scalar_tensor_request_and_response(dtype):
    server, client = axon.AxonRuntime("scalar_server"), axon.AxonRuntime("scalar_client")
    expected = np.array(7, dtype=dtype)

    async def echo(value: np.ndarray) -> np.ndarray:
        np.testing.assert_array_equal(value, expected)
        # NumPy DLPack imports are readonly; export a writable scalar response.
        return value.copy()

    server.start()
    client.start_client()
    try:
        server.register_function(echo, 11, from_dlpack_fn=np.from_dlpack)
        await client.connect_endpoint_async(server.get_local_address(), "scalar_server")
        result = await client.invoke(expected, worker_name="scalar_server", session_id=0,
                                     function=11, from_dlpack_fn=np.from_dlpack)
        assert result.shape == ()
        assert result.dtype == expected.dtype
        np.testing.assert_array_equal(result, expected)
    finally:
        client.stop()
        server.stop()
    np.testing.assert_array_equal(result, expected)
