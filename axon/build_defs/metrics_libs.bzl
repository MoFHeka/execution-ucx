"""Metrics libraries definitions for Axon core."""

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//axon:build_defs/utils_libs.bzl", "SUPPORTED_CPP_STANDARDS")

def axon_metrics_libs():
    """Defines all metrics-related libraries."""
    cc_library(
        name = "axon_metrics",
        hdrs = ["include/axon/metrics/metrics_observer.hpp"],
        includes = ["include"],
        target_compatible_with = select(
            {":is_cpp" + v: [] for v in SUPPORTED_CPP_STANDARDS} |
            {"//conditions:default": ["@platforms//:incompatible"]},
        ),
        deps = [
            "@execution-ucx//rpc_core:rpc_types_lib",
            "@proxy",
        ],
    )
