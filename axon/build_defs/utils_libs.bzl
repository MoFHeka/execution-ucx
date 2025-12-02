"""Utils libraries definitions for Axon core."""

load("@rules_cc//cc:cc_library.bzl", "cc_library")

def axon_utils_libs():
    """Defines all utils-related libraries."""
    cc_library(
        name = "tensor",
        hdrs = ["include/axon/utils/tensor.hpp"],
        includes = ["include"],
        copts = ["-std=c++20"],
        deps = [
            "@cista",
            "@dlpack",
            "@execution-ucx//rpc_core:rpc_headers_lib",
        ],
    )

    cc_library(
        name = "axon_message",
        srcs = ["src/utils/axon_message.cpp"],
        hdrs = ["include/axon/utils/axon_message.hpp"],
        includes = ["include"],
        copts = ["-std=c++23"],
        deps = [
            ":tensor",
            "@execution-ucx//rpc_core:rpc_headers_lib",
            "@execution-ucx//ucx_context:ucx_context_data_lib",
        ],
    )

    cc_library(
        name = "hash",
        srcs = ["src/utils/hash.cpp"],
        hdrs = ["include/axon/utils/hash.hpp"],
        includes = ["include"],
    )

    cc_library(
        name = "ring_buffer",
        hdrs = ["include/axon/utils/ring_buffer.hpp"],
        includes = ["include"],
        copts = ["-std=c++23"],
    )

    cc_library(
        name = "slot_map",
        hdrs = ["include/axon/utils/slot_map.hpp"],
        includes = ["include"],
        copts = ["-std=c++23"],
    )

    cc_library(
        name = "axon_utils",
        deps = [
            ":axon_message",
            ":hash",
            ":slot_map",
            ":tensor",
            ":ring_buffer",
        ],
    )
