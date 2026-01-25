"""Bzip2/libbz2, a program and library for lossless, block-sorting data compression."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def bzip2_repo():
    http_archive(
        name = "bzip2",
        sha256 = "ab5a03176ee106d3f0fa90e381da478ddae405918153cca248e682cd0c4a2269",
        strip_prefix = "bzip2-1.0.8",
        url = "https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz",
        build_file = "//third_party/bzip2:BUILD.bazel",
    )
