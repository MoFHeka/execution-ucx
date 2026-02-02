"""Test the new Device API."""

import pytest
import axon


def test_device_types():
    """Test that all device types are available."""
    # Check that DeviceType enum exists
    assert hasattr(axon, "DeviceType")
    assert hasattr(axon.DeviceType, "CPU")
    assert hasattr(axon.DeviceType, "CUDA")
    assert hasattr(axon.DeviceType, "ROCM")
    assert hasattr(axon.DeviceType, "SYCL")


def test_cpu_device():
    """Test CpuDevice creation and methods."""
    # Basic creation
    cpu_dev = axon.CpuDevice()
    assert cpu_dev.get_type() == axon.DeviceType.CPU
    assert cpu_dev.get_type_string() == "cpu"
    assert cpu_dev.get_context_handle() is None
    assert "CpuDevice" in repr(cpu_dev)

    # With NUMA node
    cpu_dev_numa = axon.CpuDevice(numa_node=1)
    assert cpu_dev_numa.numa_node == 1
    assert "numa_node=1" in repr(cpu_dev_numa)


def test_cuda_device():
    """Test CudaDevice creation and methods."""
    # Basic creation (may or may not have context depending on CUDA availability)
    cuda_dev = axon.CudaDevice(device_id=0)
    assert cuda_dev.get_type() == axon.DeviceType.CUDA
    assert cuda_dev.get_type_string() == "cuda"
    assert cuda_dev.device_id == 0
    assert "CudaDevice" in repr(cuda_dev)

    # With explicit context
    cuda_dev_ctx = axon.CudaDevice(device_id=1, context=0x12345678)
    assert cuda_dev_ctx.device_id == 1
    assert cuda_dev_ctx.get_context_handle() == 0x12345678


def test_rocm_device():
    """Test RocmDevice creation and methods."""
    rocm_dev = axon.RocmDevice(device_id=0)
    assert rocm_dev.get_type() == axon.DeviceType.ROCM
    assert rocm_dev.get_type_string() == "rocm"
    assert rocm_dev.device_id == 0
    assert "RocmDevice" in repr(rocm_dev)


def test_sycl_device():
    """Test SyclDevice creation and methods."""
    sycl_dev = axon.SyclDevice(device_id=0)
    assert sycl_dev.get_type() == axon.DeviceType.SYCL
    assert sycl_dev.get_type_string() == "sycl"
    assert sycl_dev.device_id == 0
    assert "SyclDevice" in repr(sycl_dev)


def test_factory_functions():
    """Test convenience factory functions."""
    # CPU
    cpu_dev = axon.cpu()
    assert isinstance(cpu_dev, axon.CpuDevice)

    cpu_dev_numa = axon.cpu(numa_node=0)
    assert cpu_dev_numa.numa_node == 0

    # CUDA
    cuda_dev = axon.cuda()
    assert isinstance(cuda_dev, axon.CudaDevice)
    assert cuda_dev.device_id == 0

    cuda_dev1 = axon.cuda(device_id=1)
    assert cuda_dev1.device_id == 1

    # ROCm
    rocm_dev = axon.rocm()
    assert isinstance(rocm_dev, axon.RocmDevice)

    # SYCL
    sycl_dev = axon.sycl()
    assert isinstance(sycl_dev, axon.SyclDevice)


def test_auto_detect():
    """Test auto_detect returns a valid device."""
    device = axon.auto_detect()
    assert isinstance(device, axon.Device)
    # Should at least have CPU as fallback
    assert device is not None


def test_runtime_with_cpu_device():
    """Test creating AxonRuntime with CPU device."""
    cpu_dev = axon.cpu()
    runtime = axon.AxonRuntime("test_worker", device=cpu_dev)
    assert runtime is not None


def test_runtime_with_cuda_device():
    """Test creating AxonRuntime with CUDA device."""
    # This should work even if CUDA is not available
    # The runtime will handle the error internally
    cuda_dev = axon.cuda(context=None)  # No context
    runtime = axon.AxonRuntime("test_worker", device=cuda_dev)
    assert runtime is not None


def test_runtime_without_device():
    """Test creating AxonRuntime without explicit device (defaults to CPU)."""
    runtime = axon.AxonRuntime("test_worker")
    assert runtime is not None


@pytest.mark.asyncio
async def test_runtime_lifecycle_with_device():
    """Test full lifecycle with device."""
    device = axon.cpu()
    runtime = axon.AxonRuntime("test_worker", device=device)
    runtime.start()

    # Verify it's running by getting local address
    addr = runtime.get_local_address()
    assert len(addr) > 0

    runtime.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
