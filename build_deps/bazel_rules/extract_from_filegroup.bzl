"""Export extract directory from an output group."""

def _extract_from_filegroup_impl(ctx):
    output_dir = ctx.actions.declare_directory(ctx.attr.out_dir)

    extract_string = ""
    for extract in ctx.attr.extracts:
        extract_string += "--include=%s " % extract

    ctx.actions.run_shell(
        inputs = [ctx.file.src],
        outputs = [output_dir],
        command = "rsync -a -L " +
                  extract_string +
                  "--exclude=* " +
                  ctx.file.src.path + "/ " + output_dir.path + "/ ",
    )

    return [DefaultInfo(files = depset([output_dir]))]

extract_from_filegroup = rule(
    implementation = _extract_from_filegroup_impl,
    attrs = {
        "src": attr.label(allow_single_file = True, providers = ["files"], mandatory = True),
        "extracts": attr.string_list(mandatory = True, allow_empty = False),
        "out_dir": attr.string(mandatory = True),
    },
)

def _find_and_group_outputs_impl(ctx):
    """
    Implementation of the find_and_group_outputs rule.

    Args:
        ctx: The Starlark rule context.

    Returns:
        A DefaultInfo provider containing all files found in matching output groups.
    """
    pattern = ctx.attr.pattern

    pattern_func = {
        "startswith": lambda x, y: x.startswith(y),
        "endswith": lambda x, y: x.endswith(y),
        "contains": lambda x, y: y in x,
    }

    transitive_depsets = []
    for dep_target in ctx.attr.targets_to_inspect:
        # Check if the target exports OutputGroupInfo
        if OutputGroupInfo in dep_target:
            output_group_info = dep_target[OutputGroupInfo]

            # Iterate over all group names in OutputGroupInfo
            for group_name in output_group_info:
                # Check if the group name contains the specified pattern
                if pattern_func[ctx.attr.pattern_type](group_name, pattern):
                    # Starlark doesn't support try/except. Check for attribute existence instead.
                    if hasattr(output_group_info, group_name):
                        # Get the depset of files for the group
                        files_in_group = getattr(output_group_info, group_name)
                        transitive_depsets.append(files_in_group)

    all_matched_files = depset(transitive = transitive_depsets, order = "default")

    return [
        DefaultInfo(files = all_matched_files),
    ]

find_and_group_outputs = rule(
    implementation = _find_and_group_outputs_impl,
    attrs = {
        "targets_to_inspect": attr.label_list(
            doc = "The list of targets whose OutputGroupInfo should be inspected.",
            allow_files = True,
        ),
        "pattern": attr.string(
            mandatory = True,
            doc = "The substring pattern to match against output group names.",
        ),
        "pattern_type": attr.string(
            default = "endswith",
            doc = "The type of pattern to match against output group names.",
            values = ["startswith", "endswith", "contains"],
        ),
    },
    doc = """
    A rule to collect files from specific output groups of its dependencies.

    It inspects the OutputGroupInfo of each dependency, finds groups whose names
    contain the specified pattern, and aggregates all files from these matching
    groups into its own default outputs.
    """,
)
