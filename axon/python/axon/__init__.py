"""Python wrapper for Axon Runtime C++ extension module.

This module dynamically loads the C++ extension library built by Bazel.
"""

import os
import sys

# Import and re-export device module
from .future import AxonFuture
from .device import (
    Device,
    DeviceType,
    CpuDevice,
    CudaDevice,
    RocmDevice,
    SyclDevice,
    cpu,
    cuda,
    rocm,
    sycl,
    auto_detect,
)

try:
    # UCX memory hooks (ucm) require RTLD_GLOBAL to intercept munmap/madvise
    # from other libraries like NumPy. Without this, UCX's registration cache
    # (rcache) holds stale references to freed memory, causing shared memory leaks.
    old_flags = sys.getdlopenflags()
    sys.setdlopenflags(old_flags | os.RTLD_GLOBAL)
    from ._axon import *  # type: ignore # noqa: F403
    from ._axon import UcxMemoryResourceManager, DefaultUcxMemoryResourceManager  # type: ignore

    sys.setdlopenflags(old_flags)

    _module = sys.modules.get(__name__ + "._axon")
except ImportError as e:
    raise ImportError(
        "Failed to load axon extension module.\n"
        "Please ensure that axon.so has been built.\n"
        f"Original error: {e}"
    )

# Create a package-level DefaultUcxMemoryResourceManager instance.
# Its lifetime matches the package, outliving any AxonRuntime or tensor objects,
# which prevents the underlying memory from being freed while tensors still exist.
_default_resource_manager = None
if _module and hasattr(_module, "DefaultUcxMemoryResourceManager"):
    _default_resource_manager = _module.DefaultUcxMemoryResourceManager()


def get_default_resource_manager():
    """Return the package-level DefaultUcxMemoryResourceManager instance.

    This manager's lifetime is tied to the axon package. Pass it to
    AxonRuntime to ensure UCX memory buffers are not freed while
    DLPack tensors derived from them are still alive.
    """
    return _default_resource_manager


# Re-export everything from the C++ module
if _module:
    # Copy all attributes from the C++ module to this module's namespace
    for name in dir(_module):
        if not name.startswith("_"):
            setattr(sys.modules[__name__], name, getattr(_module, name))

# Wrap AxonRuntime to auto-inject the global memory manager when none is provided
_CppAxonRuntime = getattr(sys.modules[__name__], "AxonRuntime", None)
if _CppAxonRuntime is not None:

    class AxonRuntime(_CppAxonRuntime):
        def __init__(self, *args, **kwargs):
            if (
                not any(isinstance(x, UcxMemoryResourceManager) for x in args)
                and "resource_manager" not in kwargs
                and _default_resource_manager is not None
            ):
                kwargs["resource_manager"] = _default_resource_manager
            super().__init__(*args, **kwargs)

        def invoke(self, *args, **kwargs):
            asyncio_future = super().invoke(*args, **kwargs)
            return AxonFuture(asyncio_future)

        def invoke_raw(self, *args, **kwargs):
            asyncio_future = super().invoke_raw(*args, **kwargs)
            return AxonFuture(asyncio_future)

    setattr(sys.modules[__name__], "AxonRuntime", AxonRuntime)

__all__ = [
    # Runtime
    "AxonRuntime",
    # Memory resource manager
    "UcxMemoryResourceManager",
    "DefaultUcxMemoryResourceManager",
    "get_default_resource_manager",
    # Device types and classes
    "Device",
    "DeviceType",
    "CpuDevice",
    "CudaDevice",
    "RocmDevice",
    "SyclDevice",
    # Device factory functions
    "cpu",
    "cuda",
    "rocm",
    "sycl",
    "auto_detect",
]
