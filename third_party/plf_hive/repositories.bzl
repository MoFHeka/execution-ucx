"""
plf::hive is a fork of plf::colony to match the current C++ standards proposal (https://wg21.link/p0447).
"""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def plf_hive_repo():
    new_git_repository(
        name = "plf_hive",
        remote = "https://github.com/mattreecebentley/plf_hive.git",
        branch = "main",
        build_file = "//third_party/plf_hive:BUILD.bazel",
    )
