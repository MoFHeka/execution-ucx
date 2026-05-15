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

def _file_prefix_filter_impl(ctx):
    files = []
    for f in ctx.files.srcs:
        excluded = False
        for ex in ctx.attr.exclude_prefixes:
            if f.basename.startswith(ex):
                excluded = True
                break
        if excluded:
            continue
        for prefix in ctx.attr.prefixes:
            if f.basename.startswith(prefix):
                files.append(f)
                break
    return [DefaultInfo(files = depset(files))]

file_prefix_filter = rule(
    implementation = _file_prefix_filter_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "prefixes": attr.string_list(),
        "exclude_prefixes": attr.string_list(default = []),
    },
)
