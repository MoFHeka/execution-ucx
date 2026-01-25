"""Errors libraries definitions for Axon core."""

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//axon:build_defs/utils_libs.bzl", "SUPPORTED_CPP_STANDARDS")

def axon_errors_libs():
    """Defines all errors-related libraries."""
    cc_library(
        name = "axon_error",
        srcs = ["src/errors/error_types.cpp"],
        hdrs = ["include/axon/errors/error_types.hpp"],
        includes = ["include"],
        target_compatible_with = select(
            {":is_cpp" + v: [] for v in SUPPORTED_CPP_STANDARDS} |
            {"//conditions:default": ["@platforms//:incompatible"]},
        ),
        deps = [
            "@execution-ucx//rpc_core:rpc_status_lib",
            "@execution-ucx//rpc_core:hybrid_logical_clock_lib",
            "@proxy",
        ],
    )
