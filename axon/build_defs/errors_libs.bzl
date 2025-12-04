"""Errors libraries definitions for Axon core."""

load("@rules_cc//cc:cc_library.bzl", "cc_library")

def axon_errors_libs():
    """Defines all errors-related libraries."""
    cc_library(
        name = "axon_error",
        srcs = ["src/errors/error_types.cpp"],
        hdrs = ["include/axon/errors/error_types.hpp"],
        includes = ["include"],
        copts = ["-std=c++23"],
        deps = [
            "@execution-ucx//rpc_core:rpc_status_lib",
            "@execution-ucx//rpc_core:hybrid_logical_clock_lib",
            "@proxy",
        ],
    )
