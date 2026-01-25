"""Bzip2/libbz2, a program and library for lossless, block-sorting data compression."""

load("//third_party/bzip2:repositories.bzl", "bzip2_repo")

def _bzip2_dep_impl(_ctx):
    bzip2_repo()

bzip2_dep = module_extension(
    implementation = _bzip2_dep_impl,
)
