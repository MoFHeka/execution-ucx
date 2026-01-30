"""Pytest configuration for Axon Runtime tests."""

import os
import sys


def pytest_configure(config):
    """Configure pytest to set up PYTHONPATH for axon module."""
    # In Bazel, the axon_python_runtime.so will be in the runfiles
    # We need to add its directory to PYTHONPATH
    # Try to find the module in common locations
    possible_paths = [
        os.path.join(os.path.dirname(__file__), ".."),
        os.path.join(os.path.dirname(__file__), "..", ".."),
        os.path.dirname(__file__),
    ]

    # Also check runfiles (Bazel)
    if "TEST_SRCDIR" in os.environ:
        runfiles_dir = os.environ["TEST_SRCDIR"]
        possible_paths.insert(0, runfiles_dir)
        possible_paths.insert(
            0, os.path.join(runfiles_dir, "execution_ucx", "axon", "python")
        )

    for path in possible_paths:
        runtime_so = os.path.join(path, "axon_python_runtime.so")
        if os.path.exists(runtime_so):
            if path not in sys.path:
                sys.path.insert(0, path)
            break

    # Also check if it's already importable
    try:
        import axon  # noqa: F401
    except ImportError:
        # Try to find it in the current directory or parent
        import glob

        for pattern in ["**/axon_python_runtime.so", "axon_python_runtime.so"]:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                runtime_dir = os.path.dirname(os.path.abspath(matches[0]))
                if runtime_dir not in sys.path:
                    sys.path.insert(0, runtime_dir)
                break
