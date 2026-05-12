"""Repository rule to download and extract pre-built libraries from .deb packages.

This provides real, pre-compiled shared libraries for UCX cross-compilation
in a fully hermetic Bazel build. No host system libraries are used.
"""

def _deb_import_impl(repo_ctx):
    repo_ctx.execute(["mkdir", "-p", "lib", "include"])

    for i, url in enumerate(repo_ctx.attr.urls):
        deb_file = "pkg_{}.deb".format(i)
        repo_ctx.download(url, output = deb_file, sha256 = repo_ctx.attr.sha256s[i] if i < len(repo_ctx.attr.sha256s) else "")
        repo_ctx.execute(["ar", "x", deb_file])

        result = repo_ctx.execute(["sh", "-c", "ls data.tar.* 2>/dev/null | head -1"])
        data_tar = result.stdout.strip()
        if data_tar:
            repo_ctx.execute(["tar", "xf", data_tar])
            repo_ctx.execute(["rm", "-f", data_tar])
        repo_ctx.execute(["rm", "-f", deb_file, "control.tar.gz", "control.tar.xz", "control.tar.zst", "debian-binary"])

    lib_search = repo_ctx.attr.lib_search_path
    result = repo_ctx.execute(["sh", "-c", "find . -path '*{}*' -name '*.so*' \\( -type f -o -type l \\)".format(lib_search)])
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        basename = repo_ctx.execute(["basename", line]).stdout.strip()
        repo_ctx.execute(["cp", "-a", line, "lib/" + basename])

    result = repo_ctx.execute(["sh", "-c", "find . -path '*/include/*' -name '*.h' -type f"])
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        idx = line.find("/include/")
        if idx >= 0:
            rel = line[idx + len("/include/"):]
            parent = repo_ctx.execute(["dirname", "include/" + rel]).stdout.strip()
            repo_ctx.execute(["mkdir", "-p", parent])
            repo_ctx.execute(["cp", line, "include/" + rel])

    repo_ctx.execute(["sh", "-c", "rm -rf usr etc var"])

    repo_ctx.file("BUILD.bazel", """
package(default_visibility = ["//visibility:public"])

filegroup(
    name = "libs",
    srcs = glob(["lib/*.so*"]),
)

filegroup(
    name = "headers",
    srcs = glob(["include/**/*.h"]),
)

cc_library(
    name = "cc_libs",
    hdrs = glob(["include/**/*.h"]),
    includes = ["include"],
    srcs = glob(["lib/*.so*"]),
    linkopts = ["-Wl,--as-needed"],
)
""")

deb_import = repository_rule(
    implementation = _deb_import_impl,
    attrs = {
        "urls": attr.string_list(mandatory = True),
        "sha256s": attr.string_list(default = []),
        "lib_search_path": attr.string(default = "x86_64-linux-gnu"),
    },
    doc = "Downloads .deb packages and extracts pre-built shared libraries and headers.",
)
