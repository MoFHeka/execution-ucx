"""Device abstraction for Axon runtime.

This module provides device abstractions for different hardware backends
(CPU, CUDA, ROCm, SYCL, etc.) to enable heterogeneous computing support.
"""

from abc import ABC, abstractmethod
from typing import Optional
import enum


class DeviceType(enum.Enum):
    """Enumeration of supported device types."""

    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    SYCL = "sycl"


class Device(ABC):
    """Abstract base class for all device types.

    Each device implementation encapsulates device-specific configuration
    and context management.
    """

    @abstractmethod
    def get_type(self) -> DeviceType:
        """Return the device type.

        Returns:
            DeviceType: The type of this device.
        """
        pass

    @abstractmethod
    def get_type_string(self) -> str:
        """Return the device type as a string.

        Returns:
            str: The device type string (e.g., "cuda", "cpu").
        """
        pass

    @abstractmethod
    def get_context_handle(self) -> Optional[int]:
        """Get the device context handle.

        Returns:
            Optional[int]: The device context handle as an integer,
                          or None if not applicable (e.g., CPU).
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class CpuDevice(Device):
    """CPU device - default fallback device.

    Args:
        numa_node: Optional NUMA node affinity.
    """

    def __init__(self, numa_node: Optional[int] = None):
        self.numa_node = numa_node

    def get_type(self) -> DeviceType:
        return DeviceType.CPU

    def get_type_string(self) -> str:
        return "cpu"

    def get_context_handle(self) -> Optional[int]:
        return None  # CPU doesn't need a context handle

    def __repr__(self) -> str:
        if self.numa_node is not None:
            return f"CpuDevice(numa_node={self.numa_node})"
        return "CpuDevice()"


class CudaDevice(Device):
    """CUDA device for NVIDIA GPUs.

    Args:
        device_id: CUDA device ID (default: 0).
        context: Optional explicit CUDA context handle. If None,
                automatically retrieves the current context.
    """

    def __init__(
        self,
        device_id: int = 0,
        context: Optional[int] = None,
    ):
        self.device_id = device_id
        self._context = context

        # If no explicit context provided, try to get current context
        if self._context is None:
            self._context = self._get_current_context()

    def _get_current_context(self) -> Optional[int]:
        """Automatically get the current CUDA context."""
        try:
            import axon

            return axon.get_device_context_handle("cuda")
        except Exception:
            return None

    def get_type(self) -> DeviceType:
        return DeviceType.CUDA

    def get_type_string(self) -> str:
        return "cuda"

    def get_context_handle(self) -> Optional[int]:
        return self._context

    def __repr__(self) -> str:
        if self._context is not None:
            return f"CudaDevice(device_id={self.device_id}, context={self._context:#x})"
        return f"CudaDevice(device_id={self.device_id})"


class RocmDevice(Device):
    """ROCm/HIP device for AMD GPUs.

    Args:
        device_id: ROCm device ID (default: 0).
        context: Optional explicit HIP context handle. If None,
                automatically retrieves the current context.
    """

    def __init__(
        self,
        device_id: int = 0,
        context: Optional[int] = None,
    ):
        self.device_id = device_id
        self._context = context or self._get_current_context()

    def _get_current_context(self) -> Optional[int]:
        """Automatically get the current HIP context."""
        try:
            import axon

            return axon.get_device_context_handle("rocm")
        except Exception:
            return None

    def get_type(self) -> DeviceType:
        return DeviceType.ROCM

    def get_type_string(self) -> str:
        return "rocm"

    def get_context_handle(self) -> Optional[int]:
        return self._context

    def __repr__(self) -> str:
        return f"RocmDevice(device_id={self.device_id})"


class SyclDevice(Device):
    """SYCL device for Intel oneAPI.

    Args:
        device_id: SYCL device ID (default: 0).
        context: Optional explicit SYCL context handle.
        queue: Optional SYCL queue handle.
    """

    def __init__(
        self,
        device_id: int = 0,
        context: Optional[int] = None,
        queue: Optional[int] = None,
    ):
        self.device_id = device_id
        self._context = context
        self.queue = queue

    def get_type(self) -> DeviceType:
        return DeviceType.SYCL

    def get_type_string(self) -> str:
        return "sycl"

    def get_context_handle(self) -> Optional[int]:
        return self._context

    def __repr__(self) -> str:
        return f"SyclDevice(device_id={self.device_id})"


# Convenience factory functions


def cpu(numa_node: Optional[int] = None) -> CpuDevice:
    """Create a CPU device.

    Args:
        numa_node: Optional NUMA node affinity.

    Returns:
        CpuDevice: A CPU device instance.
    """
    return CpuDevice(numa_node=numa_node)


def cuda(device_id: int = 0, context: Optional[int] = None) -> CudaDevice:
    """Create a CUDA device.

    Args:
        device_id: CUDA device ID (default: 0).
        context: Optional explicit CUDA context handle.

    Returns:
        CudaDevice: A CUDA device instance.
    """
    return CudaDevice(device_id=device_id, context=context)


def rocm(device_id: int = 0, context: Optional[int] = None) -> RocmDevice:
    """Create a ROCm device.

    Args:
        device_id: ROCm device ID (default: 0).
        context: Optional explicit HIP context handle.

    Returns:
        RocmDevice: A ROCm device instance.
    """
    return RocmDevice(device_id=device_id, context=context)


def sycl(
    device_id: int = 0, context: Optional[int] = None, queue: Optional[int] = None
) -> SyclDevice:
    """Create a SYCL device.

    Args:
        device_id: SYCL device ID (default: 0).
        context: Optional explicit SYCL context handle.
        queue: Optional SYCL queue handle.

    Returns:
        SyclDevice: A SYCL device instance.
    """
    return SyclDevice(device_id=device_id, context=context, queue=queue)


def auto_detect() -> Device:
    """Automatically detect and return the best available device.

    Detection priority:
    1. CUDA (via PyTorch or JAX)
    2. ROCm (via PyTorch)
    3. CPU (fallback)

    Returns:
        Device: The best available device instance.
    """
    # Try PyTorch CUDA
    try:
        import torch

        if torch.cuda.is_available():
            return CudaDevice()
    except ImportError:
        pass

    # Try JAX CUDA
    try:
        import jax

        devices = jax.devices()
        if devices and devices[0].platform == "cuda":
            return CudaDevice()
    except ImportError:
        pass

    # Try PyTorch ROCm
    try:
        import torch

        if hasattr(torch, "hip") and torch.hip.is_available():
            return RocmDevice()
    except (ImportError, AttributeError):
        pass

    # Fallback to CPU
    return CpuDevice()
