"""Proxy: Next Generation Polymorphism in C++."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def proxy_repo():
    new_git_repository(
        name = "proxy",
        remote = "https://github.com/microsoft/proxy.git",
        branch = "main",
        build_file = "//third_party/proxy:BUILD.bazel",
    )
