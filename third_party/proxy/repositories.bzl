"""Proxy: Next Generation Polymorphism in C++."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def proxy_repo():
    new_git_repository(
        name = "proxy",
        remote = "https://github.com/microsoft/proxy.git",
        commit = "eea7350dd072322a96674f580142c9a71038fa32",
        build_file = "//third_party/proxy:BUILD.bazel",
    )
