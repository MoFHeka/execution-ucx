"""Metrics libraries definitions for Axon core."""

load("@rules_cc//cc:cc_library.bzl", "cc_library")

def axon_metrics_libs():
    """Defines all metrics-related libraries."""
    cc_library(
        name = "axon_metrics",
        hdrs = ["include/axon/metrics/metrics_observer.hpp"],
        includes = ["include"],
        copts = ["-std=c++23"],
        deps = [
            "@execution-ucx//rpc_core:rpc_types_lib",
            "@proxy",
        ],
    )

