"""Export a library from an output group."""

def cc_export_shared_library(
        name = None,
        srcs = [],
        shared_library = None,
        visibility = None):
    """Export shared library from output group.

    Args:
        name: Name of this target.
        srcs: List of labels.  See filegroups.
        shared_library: Library to export.
        visibility: The visibility attribute on the target.

    """
    filegroup_name = name + "_fg"
    native.filegroup(
        name = filegroup_name,
        srcs = srcs,
        output_group = shared_library,
        visibility = ["//visibility:private"],
    )
    native.cc_import(
        name = name,
        shared_library = filegroup_name,
        visibility = visibility,
    )

def cc_export_static_library(
        name = None,
        srcs = [],
        static_library = None,
        visibility = None):
    """Export static library from output group.

    Args:
        name: Name of this target.
        srcs: List of labels.  See filegroups.
        static_library: Library to export.
        visibility: The visibility attribute on the target.

    """
    filegroup_name = name + "_fg"
    native.filegroup(
        name = filegroup_name,
        srcs = srcs,
        output_group = static_library,
        visibility = ["//visibility:private"],
    )
    native.cc_import(
        name = name,
        static_library = filegroup_name,
        alwayslink = 1,
        visibility = visibility,
    )

def cc_export_library(
        name = None,
        srcs = [],
        shared_library = None,
        static_library = None,
        visibility = None):
    """Export shared and static library from output group.

    Args:
        name: Name of this target.
        srcs: List of labels.  See filegroups.
        shared_library: Library to export.
        static_library: Library to export.
        visibility: The visibility attribute on the target.

    """
    shared_filegroup_name = name + "_fg_shared"
    native.filegroup(
        name = shared_filegroup_name,
        srcs = srcs,
        output_group = shared_library,
        visibility = ["//visibility:private"],
    )
    static_filegroup_name = name + "_fg_static"
    native.filegroup(
        name = static_filegroup_name,
        srcs = srcs,
        output_group = static_library,
        visibility = ["//visibility:private"],
    )
    native.cc_import(
        name = name,
        shared_library = shared_filegroup_name,
        static_library = static_filegroup_name,
        visibility = visibility,
    )
