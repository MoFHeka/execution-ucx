"""Storage libraries definitions for Axon core."""

load("@rules_cc//cc:cc_library.bzl", "cc_library")

def axon_storage_libs():
    """Defines all storage-related libraries."""
    cc_library(
        name = "avro_schema",
        srcs = ["src/storage/avro_schema.cpp"],
        hdrs = ["include/axon/storage/avro_schema.hpp"],
        includes = ["include"],
        copts = ["-std=c++23"],
        deps = [
            ":axon_message",
            "@avro-cpp",
        ],
    )

    cc_library(
        name = "avro_serialization",
        srcs = ["src/storage/avro_serialization.cpp"],
        hdrs = ["include/axon/storage/avro_serialization.hpp"],
        includes = ["include"],
        copts = ["-std=c++23"],
        deps = [
            ":avro_schema",
            ":axon_message",
            "@avro-cpp",
            "@execution-ucx//rpc_core:rpc_headers_lib",
            "@execution-ucx//ucx_context:ucx_context_data_lib",
        ],
    )

    cc_library(
        name = "unifex_io",
        srcs = ["src/storage/unifex_io.cpp"],
        hdrs = ["include/axon/storage/unifex_io.hpp"],
        includes = ["include"],
        copts = ["-std=c++23"],
        deps = [
            "@unifex",
        ],
    )

    # A Pimpl implementation of avro::OutputStream and avro::InputStream
    # that uses unifex_io.hpp to implement the asynchronous I/O.
    cc_library(
        name = "avro_unifex_io",
        srcs = ["src/storage/avro_unifex_io.cpp"],
        hdrs = ["include/axon/storage/avro_unifex_io.hpp"],
        includes = ["include"],
        deps = [
            ":unifex_io",
            "@avro-cpp",
        ],
    )

    cc_library(
        name = "async_io",
        srcs = ["src/storage/async_io.cpp"],
        hdrs = ["include/axon/storage/async_io.hpp"],
        includes = ["include"],
        copts = ["-std=c++23"],
        deps = [
            ":avro_serialization",
            ":avro_unifex_io",
        ],
    )

    cc_library(
        name = "axon_storage",
        srcs = ["src/storage/axon_storage.cpp"],
        hdrs = ["include/axon/storage/axon_storage.hpp"],
        includes = ["include"],
        copts = ["-std=c++23"],
        deps = [
            ":async_io",
            ":axon_message",
            "@cista",
            "@execution-ucx//rpc_core:rpc_headers_lib",
            "@plf_hive",
        ],
    )
