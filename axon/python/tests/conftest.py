"""Pytest configuration for Axon Runtime tests."""

import os
import sys


def pytest_configure(config):
    """Configure pytest to set up PYTHONPATH and inject extension module."""
    import importlib.util

    # 1. Add source directory to sys.path
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if base_path not in sys.path:
        sys.path.insert(0, base_path)

    # 2. Injection logic for AXON_EXTENSION_PATH
    extension_path = os.environ.get("AXON_EXTENSION_PATH")
    if extension_path and os.path.exists(extension_path):
        # Set UCX module directory
        ucx_modules_path = os.environ.get("AXON_UCX_MODULES_PATH")
        if ucx_modules_path and os.path.isdir(ucx_modules_path):
            os.environ["UCX_MODULE_DIR"] = ucx_modules_path

        # In-memory injection
        module_name = "axon._axon"
        if module_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(module_name, extension_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

    # 3. Fallback logic
    possible_paths = [
        os.path.join(os.path.dirname(__file__), ".."),
        os.path.join(os.path.dirname(__file__), "..", ".."),
        os.path.dirname(__file__),
    ]

    if "TEST_SRCDIR" in os.environ:
        runfiles_dir = os.environ["TEST_SRCDIR"]
        possible_paths.insert(0, runfiles_dir)
        possible_paths.insert(
            0, os.path.join(runfiles_dir, "execution_ucx", "axon", "python")
        )

    for path in possible_paths:
        runtime_so = os.path.join(path, "_axon.so")
        if os.path.exists(runtime_so):
            if path not in sys.path:
                sys.path.insert(0, path)
            break

    # Also check if it's already importable
    try:
        import axon  # noqa: F401
    except ImportError:
        import glob

        for pattern in ["**/_axon.so", "_axon.so"]:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                runtime_dir = os.path.dirname(os.path.abspath(matches[0]))
                if runtime_dir not in sys.path:
                    sys.path.insert(0, runtime_dir)
                break
