"""Macro for building Python wheels for multiple Python versions.

This module provides functionality to build Python wheel packages
that support multiple Python versions. For C++ extension modules,
each Python version requires a separate build with the corresponding
Python toolchain.
"""

load("@rules_python//python:packaging.bzl", "py_wheel")

def _copy_to_dir_impl(ctx):
    out_dir = ctx.actions.declare_directory(ctx.attr.dirname)

    # Helper script to copy files
    # We use a script to avoid command line length limits if there are many files
    script = """
    out_dir="$1"
    match_pattern="$2"
    shift 2
    
    mkdir -p "$out_dir"
    
    for f in "$@"; do
        basename=$(basename "$f")
        # Check pattern match using case (portable)
        case "$basename" in
            $match_pattern)
                if [[ -f "$f" ]]; then
                    cp -L "$f" "$out_dir/"
                fi
                ;;
        esac
    done
    """

    ctx.actions.run_shell(
        inputs = ctx.files.srcs,
        outputs = [out_dir],
        command = script,
        arguments = [out_dir.path, ctx.attr.match_pattern] + [f.path for f in ctx.files.srcs],
        mnemonic = "CopyFiles",
    )

    return [DefaultInfo(files = depset([out_dir]), runfiles = ctx.runfiles(files = [out_dir]))]

copy_to_dir = rule(
    implementation = _copy_to_dir_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "dirname": attr.string(default = "libs"),
        "match_pattern": attr.string(default = "*"),
    },
)

def axon_python_wheel(name, version, description = "", **kwargs):
    """Build Python wheel package for axon_runtime.

    This creates a universal wheel that supports Python 3.8+.
    The actual Python version used during build is determined by
    the active Python toolchain (set via --python_version flag
    or .bazelrc configuration).

    To build wheels for multiple Python versions:
      1. Use different Python toolchain configurations
      2. Build with: bazel build //axon/python:wheel --python_version=PY3
      3. Or use configs: bazel build //axon/python:wheel --config=python38

    Args:
      name: Name for the wheel target
      version: Package version (e.g., "0.0.1")
      description: Package description
      **kwargs: Additional arguments passed to py_wheel
    """

    py_wheel(
        name = name,
        distribution = "execution-ucx",
        version = version,
        summary = description or "Axon Runtime Python bindings for distributed execution",
        # Universal wheel: supports Python 3.8+
        # The actual Python version used is determined by the active toolchain
        python_tag = "py3",
        # ABI tag: "none" means pure Python or platform-specific
        # For C++ extensions, this will be determined by the build
        abi = "none",
        # Platform tag: "any" for pure Python, platform-specific for extensions
        # Will be auto-detected based on the build platform
        platform = "any",
        # Package contents: If axon_python is included in deps, it will preserve Bazel's directory structure,
        # which results in non-idiomatic import paths like 'import axon.python.axon'.
        deps = kwargs.pop("deps", []),
        # Package metadata
        author = "He Jia",
        author_email = "mofhejia@163.com",
        license = "Apache-2.0",
        homepage = "https://github.com/MoFHeka/execution-ucx",
        classifiers = [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: Apache Software License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Software Development :: Libraries :: Python Modules",
        ],
        **kwargs
    )
