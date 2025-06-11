"""
The 'libunifex' project is a prototype implementation of the C++ sender/receiver async programming model that is currently being considered for standardisation.
"""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def libunifex_repo():
    new_git_repository(
        name = "unifex",
        remote = "https://github.com/facebookexperimental/libunifex.git",
        branch = "main",
        build_file = "//third_party/libunifex:BUILD.bazel",
    )
