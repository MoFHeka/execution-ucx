"""Test definitions for Axon core."""

load("@rules_cc//cc:defs.bzl", "cc_test")

def axon_tests():
    """Defines all test targets."""
    cc_test(
        name = "axon_worker_basic_test",
        srcs = ["tests/axon_worker_basic_test.cpp"],
        includes = ["include"],
        copts = ["-std=c++23"],
        deps = [
            ":axon_worker",
            "@execution-ucx//ucx_context:ucx_memory_resource_lib",
            "@googletest//:gtest_main",
        ],
    )

    cc_test(
        name = "axon_worker_test",
        srcs = ["tests/axon_worker_test.cpp"],
        includes = ["include"],
        copts = ["-std=c++23"],
        deps = [
            ":axon_worker",
            "@execution-ucx//ucx_context:ucx_memory_resource_lib",
            "@googletest//:gtest_main",
        ],
    )

    cc_test(
        name = "axon_worker_integration_test",
        srcs = ["tests/axon_worker_test.cpp"],
        includes = ["include"],
        copts = ["-std=c++23"],
        deps = [
            ":axon_worker",
            "@execution-ucx//ucx_context:ucx_memory_resource_lib",
            "@googletest//:gtest_main",
        ],
    )

    cc_test(
        name = "avro_serialization_test",
        srcs = ["tests/avro_serialization_test.cpp"],
        includes = ["include"],
        copts = ["-std=c++23"],
        deps = [
            ":avro_serialization",
            "@execution-ucx//ucx_context:ucx_memory_resource_lib",
            "@googletest//:gtest_main",
        ],
    )

    cc_test(
        name = "unifex_io_test",
        srcs = ["tests/unifex_io_test.cpp"],
        includes = ["include"],
        copts = ["-std=c++23"],
        deps = [
            ":unifex_io",
            "@googletest//:gtest_main",
        ],
    )

    cc_test(
        name = "async_io_test",
        srcs = ["tests/async_io_test.cpp"],
        includes = ["include"],
        copts = ["-std=c++23"],
        deps = [
            ":async_io",
            "@googletest//:gtest_main",
        ],
    )

    cc_test(
        name = "axon_storage_test",
        srcs = ["tests/axon_storage_test.cpp"],
        includes = ["include"],
        copts = ["-std=c++23"],
        deps = [
            ":axon_storage",
            "@googletest//:gtest_main",
        ],
    )

