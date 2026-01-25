"""
Utility functions for OpenUCX.
"""

def _file_filter_impl(ctx):
    files = []
    for f in ctx.files.srcs:
        for ext in ctx.attr.extensions:
            if f.basename.endswith(ext):
                files.append(f)
                break
    return [DefaultInfo(files = depset(files))]

file_filter = rule(
    implementation = _file_filter_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "extensions": attr.string_list(),
    },
)
