"""DLPack is a protocol for sharing arrays between deep learning frameworks."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def dlpack_repo():
    http_archive(
        name = "dlpack",
        sha256 = "cf965c26a5430ba4cc53d61963f288edddcd77443aa4c85ce722aaf1e2f29513",
        strip_prefix = "dlpack-0.8",
        url = "https://github.com/dmlc/dlpack/archive/refs/tags/v0.8.tar.gz",
        build_file = "//third_party/dlpack:BUILD.bazel",
    )
