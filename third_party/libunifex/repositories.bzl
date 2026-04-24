"""
The 'libunifex' project is a prototype implementation of the C++ sender/receiver async programming model that is currently being considered for standardisation.
"""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def libunifex_repo():
    new_git_repository(
        name = "unifex",
        remote = "https://github.com/facebookexperimental/libunifex.git",
        commit = "8c5fa963e0198db303f816de33790f03fbca7f45",
        build_file = "//third_party/libunifex:BUILD.bazel",
    )
