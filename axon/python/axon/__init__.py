"""Python wrapper for Axon Runtime C++ extension module.

This module dynamically loads the C++ extension library built by Bazel.
"""

import os
import sys
import importlib.util
from pathlib import Path


def _preload_libs(base_dir=None):
    """Preload UCX libraries from libs directory."""
    import ctypes
    import os
    import glob

    # Path to libs directory
    if base_dir:
        libs_dir = os.path.join(base_dir, "libs")
    else:
        libs_dir = os.path.join(os.path.dirname(__file__), "libs")

    if not os.path.isdir(libs_dir):
        return

    # Add to LD_LIBRARY_PATH (may help subprocesses)
    os.environ["LD_LIBRARY_PATH"] = (
        libs_dir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    )

    # Explicitly load libraries with RTLD_GLOBAL to resolve symbols
    # Order matters: lower level libs first (ucs -> uct -> ucp)
    # libucm depends on libucs symbols, but libucs depends on libucm file.
    # We use RTLD_LAZY to resolve this circular dependency.
    priority_libs = [
        "libucm.so.0",  # Memory hook
        "libucs.so.0",  # Services
        "libuct.so.0",  # Transport
        "libucp.so.0",  # Protocols
    ]

    loaded = set()

    # Load priority libs first
    for lib_name in priority_libs:
        lib_path = os.path.join(libs_dir, lib_name)
        if os.path.exists(lib_path):
            try:
                # Use RTLD_LAZY to allow loading libucm even if symbols (from libucs) are missing initially
                # os.RTLD_LAZY is used because ctypes.RTLD_LAZY might not be available on all platforms/versions
                mode = os.RTLD_GLOBAL | os.RTLD_LAZY
                ctypes.CDLL(lib_path, mode=mode)
                loaded.add(lib_name)
            except OSError:
                pass

    # Load rest
    for lib_path in glob.glob(os.path.join(libs_dir, "*.so*")):
        lib_name = os.path.basename(lib_path)
        if lib_name not in loaded:
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


# Preload bundled libraries before loading extension
# Try default location first
# _preload_libs()


def _find_and_load_module():
    """Find and load the C++ extension module."""
    # Get the directory where this file is located
    current_dir = Path(__file__).parent.absolute()

    # Possible library names
    # Bazel generates libaxon_python_runtime.so, we copy it to axon.so for Python import
    possible_names = [
        "axon.so",  # Preferred: copied by genrule to axon/axon.so
        "libaxon_python_runtime.so",  # Bazel raw output (fallback)
    ]

    # Search in current directory and common Bazel output locations
    search_paths = [
        current_dir,  # axon/ (source location)
        current_dir.parent.parent.parent
        / "bazel-bin"
        / "axon"
        / "python"
        / "axon",  # bazel-bin output
    ]

    # Also check runfiles if running under Bazel
    if "TEST_SRCDIR" in os.environ:
        runfiles_dir = Path(os.environ["TEST_SRCDIR"])
        search_paths.insert(
            0, runfiles_dir / "execution_ucx" / "axon" / "python" / "axon"
        )
        search_paths.insert(0, runfiles_dir)

    # Try to find the library
    for search_path in search_paths:
        for lib_name in possible_names:
            lib_path = Path(search_path) / lib_name
            if lib_path.exists():
                # Add directory to sys.path
                lib_dir = str(lib_path.parent)

                # Preload libs from this directory
                _preload_libs(lib_dir)

                if lib_dir not in sys.path:
                    sys.path.insert(0, lib_dir)

                # Try to load using importlib (works for nanobind modules)
                try:
                    spec = importlib.util.spec_from_file_location("axon", lib_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        try:
                            spec.loader.exec_module(module)
                            return module
                        except Exception:
                            raise
                except Exception:
                    # If we found the file but failed to load it, we should raise the error
                    # (e.g. missing dependencies) instead of continuing search.
                    raise

    # If we still can't find it, try importing directly (might work if PYTHONPATH is set)
    try:
        import axon  # noqa: F401

        return sys.modules["axon"]
    except ImportError:
        pass

    raise ImportError(
        "Failed to load axon extension module.\n"
        "Please ensure that axon.so has been built.\n"
        "Build it using: bazel build //axon/python:axon_python_lib\n"
        "Or set PYTHONPATH to the directory containing the .so file."
    )


# Load the module
_module = _find_and_load_module()

# Re-export everything from the C++ module
if _module:
    # Copy all attributes from the C++ module to this module's namespace
    for name in dir(_module):
        if not name.startswith("_"):
            setattr(sys.modules[__name__], name, getattr(_module, name))

# Import and re-export device module
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

__all__ = [
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
