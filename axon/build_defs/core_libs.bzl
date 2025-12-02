"""Core libraries definitions for Axon core."""

load("@rules_cc//cc:cc_library.bzl", "cc_library")

def axon_core_libs():
    """Defines all core libraries."""
    cc_library(
        name = "axon_memory_policy",
        hdrs = ["include/axon/memory_policy.hpp"],
        includes = ["include"],
        copts = ["-std=c++23"],
        deps = [
            "@proxy",
        ],
    )

    cc_library(
        name = "axon_message_lifecycle_policy",
        hdrs = ["include/axon/message_lifecycle_policy.hpp"],
        includes = ["include"],
        copts = ["-std=c++23"],
        deps = [
            ":axon_message",
        ],
    )

    cc_library(
        name = "axon_execution_policy",
        includes = ["include"],
        copts = ["-std=c++23"],
        deps = [
            ":axon_memory_policy",
            ":axon_message_lifecycle_policy",
            "@proxy",
        ],
    )

    cc_library(
        name = "axon_worker",
        srcs = ["src/axon_worker.cpp"],
        hdrs = ["include/axon/axon_worker.hpp"],
        includes = ["include"],
        copts = ["-std=c++23"],
        deps = [
            ":axon_execution_policy",
            ":axon_error",
            ":axon_metrics",
            ":axon_storage",
            ":axon_utils",
            "@cista",
            "@execution-ucx//rpc_core:async_rpc_headers_lib",
            "@execution-ucx//ucx_context:ucx_am_context",
            "@plf_hive",
            "@unifex",
        ],
    )

