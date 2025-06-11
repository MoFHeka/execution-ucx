"""
Unified Communication X (UCX) is an award winning, optimized production proven-communication framework for modern, high-bandwidth and low-latency networks.
"""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def openucx_repo(name = "ucx", ucx_path = "", source_path = "", git_tag = ""):
    """Creates a UCX repository rule based on the provided parameters.

    Args:
        name: Repository name
        ucx_path: Path to local UCX installation
        source_path: Path to local UCX source code
        git_tag: Git tag to use
    """
    if ucx_path and native.existing_rule(name) == None:
        maybe(
            new_local_repository,
            name = name,
            path = ucx_path,
            build_file_content = """
cc_library(
    name = "ucx",
    srcs = glob(["lib/*.so*"]),
    hdrs = glob(["include/**/*.h"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
            """,
        )
    elif source_path and native.existing_rule(name) == None:
        maybe(
            new_local_repository,
            name = name,
            path = source_path,
            build_file = "//third_party/openucx:BUILD.bazel",
        )
    else:
        maybe(
            new_git_repository,
            name = name,
            remote = "https://github.com/openucx/ucx.git",
            build_file = "//third_party/openucx:BUILD.bazel",
            tag = git_tag if git_tag else "",
        )
