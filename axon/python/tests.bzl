"""Definitions for Axon Python tests."""

load("@python_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_test")

def axon_pytest_test(name, srcs, deps = [], data = [], **kwargs):
    """
    Wrapper around py_test for Axon Python tests using pytest.

    Args:
        name: The name of the test target.
        srcs: The source files for the test. The first file is assumed to be the main entry point.
        deps: Additional dependencies for the test. Common dependencies are added automatically.
        data: Additional data files for the test. :axon_python_lib is added automatically.
        **kwargs: Additional arguments passed to py_test.
    """

    # Common dependencies for all Axon pytest tests
    common_deps = [
        "//axon/python:axon_python",
        "//axon/python:conftest",
        "//axon/python:test_utils_lib",
        requirement("pytest"),
    ]

    # Common data files
    common_data = [
        "//axon/python:axon_python_lib",
    ]

    py_test(
        name = name,
        srcs = srcs,
        # If main is not provided, default to the first source file
        main = kwargs.pop("main", srcs[0]),
        python_version = "PY3",
        data = common_data + data,
        deps = common_deps + deps,
        **kwargs
    )
