"""Cista is a simple, high-performance, zero-copy C++ serialization & reflection library."""

load("//third_party/cista:repositories.bzl", "cista_repo")

def _cista_dep_impl(ctx):
    cista_repo()

cista_dep = module_extension(
    implementation = _cista_dep_impl,
)
