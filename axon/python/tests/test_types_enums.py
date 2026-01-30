"""Tests for types and enums."""

import pytest

# Import test_utils first to setup module path
import test_utils  # noqa: F401

import axon


def test_rpc_request_header_fields():
    """Test RpcRequestHeader all fields."""
    header = axon.RpcRequestHeader()

    # Test writable fields
    header.session_id = 1
    header.request_id = 2
    header.function_id = 3
    header.workflow_id = 4

    assert header.session_id == 1
    assert header.request_id == 2
    assert header.function_id == 3
    assert header.workflow_id == 4

    # Test readonly params field
    assert hasattr(header, "params")
    # params should be a list (empty initially)
    assert isinstance(header.params, list)
    assert len(header.params) == 0

    # Add a parameter and verify it appears in params
    header.AddParam(42)
    assert len(header.params) == 1
    assert header.params[0] == 42


def test_rpc_response_header_fields():
    """Test RpcResponseHeader all fields."""
    # ResponseHeader is typically created by the runtime
    # We can test that it has the expected readonly fields
    header = axon.RpcResponseHeader()

    # All fields should be readonly
    assert hasattr(header, "session_id")
    assert hasattr(header, "request_id")
    assert hasattr(header, "workflow_id")
    assert hasattr(header, "status")
    assert hasattr(header, "results")

    # Test status field structure
    assert isinstance(header.status, dict)
    assert "value" in header.status
    assert "category_name" in header.status
    # Default status should be OK (value 0)
    assert header.status["value"] == 0

    # Test results field
    assert isinstance(header.results, list)
    assert len(header.results) == 0


def test_param_type_enum_values():
    """Test all ParamType enum values."""
    # Primitive types
    assert axon.ParamType.PRIMITIVE_BOOL is not None
    assert axon.ParamType.PRIMITIVE_INT8 is not None
    assert axon.ParamType.PRIMITIVE_INT16 is not None
    assert axon.ParamType.PRIMITIVE_INT32 is not None
    assert axon.ParamType.PRIMITIVE_INT64 is not None
    assert axon.ParamType.PRIMITIVE_UINT8 is not None
    assert axon.ParamType.PRIMITIVE_UINT16 is not None
    assert axon.ParamType.PRIMITIVE_UINT32 is not None
    assert axon.ParamType.PRIMITIVE_UINT64 is not None
    assert axon.ParamType.PRIMITIVE_FLOAT32 is not None
    assert axon.ParamType.PRIMITIVE_FLOAT64 is not None

    # Vector types
    assert axon.ParamType.VECTOR_BOOL is not None
    assert axon.ParamType.VECTOR_INT8 is not None
    assert axon.ParamType.VECTOR_INT16 is not None
    assert axon.ParamType.VECTOR_INT32 is not None
    assert axon.ParamType.VECTOR_INT64 is not None
    assert axon.ParamType.VECTOR_UINT8 is not None
    assert axon.ParamType.VECTOR_UINT16 is not None
    assert axon.ParamType.VECTOR_UINT32 is not None
    assert axon.ParamType.VECTOR_UINT64 is not None
    assert axon.ParamType.VECTOR_FLOAT32 is not None
    assert axon.ParamType.VECTOR_FLOAT64 is not None

    # Other types
    assert axon.ParamType.STRING is not None
    assert axon.ParamType.VOID is not None
    assert axon.ParamType.TENSOR_META is not None
    assert axon.ParamType.UNKNOWN is not None


def test_payload_type_enum_values():
    """Test all PayloadType enum values."""
    assert axon.PayloadType.UCX_BUFFER is not None
    assert axon.PayloadType.UCX_BUFFER_VEC is not None
    assert axon.PayloadType.NO_PAYLOAD is not None
    assert axon.PayloadType.MONOSTATE is not None


def test_rpc_errc_enum_values():
    """Test all RpcErrc enum values."""
    assert axon.RpcErrc.OK is not None
    assert axon.RpcErrc.CANCELLED is not None
    assert axon.RpcErrc.UNKNOWN is not None
    assert axon.RpcErrc.INVALID_ARGUMENT is not None
    assert axon.RpcErrc.DEADLINE_EXCEEDED is not None
    assert axon.RpcErrc.NOT_FOUND is not None
    assert axon.RpcErrc.ALREADY_EXISTS is not None
    assert axon.RpcErrc.PERMISSION_DENIED is not None
    assert axon.RpcErrc.RESOURCE_EXHAUSTED is not None
    assert axon.RpcErrc.FAILED_PRECONDITION is not None
    assert axon.RpcErrc.ABORTED is not None
    assert axon.RpcErrc.OUT_OF_RANGE is not None
    assert axon.RpcErrc.UNIMPLEMENTED is not None
    assert axon.RpcErrc.INTERNAL is not None
    assert axon.RpcErrc.UNAVAILABLE is not None
    assert axon.RpcErrc.DATA_LOSS is not None
    assert axon.RpcErrc.UNAUTHENTICATED is not None


def test_ucx_buffer_type():
    """Test UcxBuffer type is registered and accessible."""
    # Test that UcxBuffer class exists
    assert hasattr(axon, "UcxBuffer")
    assert axon.UcxBuffer is not None

    # Test that UcxBuffer is a class type
    # Note: UcxBuffer requires UcxMemoryResourceManager to construct,
    # so we can't easily create an instance in Python tests.
    # But we can verify the type is registered and has expected attributes.
    assert callable(axon.UcxBuffer)

    # Check that UcxBuffer has expected method names (as strings)
    # These would be available on instances, but we can't create instances easily
    # The type registration ensures these methods exist:
    # - data()
    # - size()
    # - type()
    # - __dlpack__()
    # - __dlpack_device__()


def test_ucx_buffer_vec_type():
    """Test UcxBufferVec type is registered and accessible."""
    # Test that UcxBufferVec class exists
    assert hasattr(axon, "UcxBufferVec")
    assert axon.UcxBufferVec is not None

    # Test that UcxBufferVec is a class type
    # Note: UcxBufferVec requires UcxMemoryResourceManager to construct,
    # so we can't easily create an instance in Python tests.
    # But we can verify the type is registered and has expected attributes.
    assert callable(axon.UcxBufferVec)

    # Check that UcxBufferVec has expected method names (as strings)
    # These would be available on instances, but we can't create instances easily
    # The type registration ensures these methods exist:
    # - __len__()
    # - __getitem__()
    # - size()
    # - type()
    # - buffers()


def test_tensor_meta_type():
    """Test TensorMeta type is registered and all fields are accessible."""
    # Test that TensorMeta class exists
    assert hasattr(axon, "TensorMeta")
    assert axon.TensorMeta is not None

    # Create a TensorMeta instance
    meta = axon.TensorMeta()

    # Test all fields are accessible
    assert hasattr(meta, "device")
    assert hasattr(meta, "ndim")
    assert hasattr(meta, "dtype")
    assert hasattr(meta, "byte_offset")
    assert hasattr(meta, "shape")
    assert hasattr(meta, "strides")

    # Test default values
    assert meta.ndim == 0
    assert meta.byte_offset == 0
    assert isinstance(meta.device, dict)
    assert isinstance(meta.dtype, dict)
    assert isinstance(meta.shape, list)
    assert isinstance(meta.strides, list)

    # Test device dict structure
    assert "device_type" in meta.device
    assert "device_id" in meta.device

    # Test dtype dict structure
    assert "code" in meta.dtype
    assert "bits" in meta.dtype
    assert "lanes" in meta.dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
