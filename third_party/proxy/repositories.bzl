"""Proxy: Next Generation Polymorphism in C++."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def proxy_repo():
    new_git_repository(
        name = "proxy",
        remote = "https://github.com/ngcpp/proxy.git",
        commit = "bcbe0c792cd42cb48edfbbdb1e272255cbe73cae",
        build_file = "//third_party/proxy:BUILD.bazel",
    )
