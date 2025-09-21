"""Cista is a simple, high-performance, zero-copy C++ serialization & reflection library."""

def _cista_repo_impl(repository_ctx):
    repository_ctx.download(
        url = "https://github.com/felixguendling/cista/releases/download/v0.15/cista.h",
        output = "cista.h.orig",
        sha256 = "45685fde5c5ef576ad821c4db7f01e375eb2668aa6dafd6b9627fe58ce0641eb",
    )

    # Forbid fmt/ostream.h directly
    original_content = repository_ctx.read("cista.h.orig")
    modified_content = original_content.replace(
        '#if __has_include("fmt/ostream.h")',
        '#if 0 && __has_include("fmt/ostream.h")',
    )
    repository_ctx.file("cista.h", modified_content)

    repository_ctx.file(
        "BUILD.bazel",
        """
cc_library(
    name = "cista",
    hdrs = ["cista.h"],
    includes = ["."],
    copts = ["-std=c++17"],
    visibility = ["//visibility:public"],
)
""",
    )

cista_repo_rule = repository_rule(
    implementation = _cista_repo_impl,
)

def cista_repo():
    cista_repo_rule(name = "cista")
