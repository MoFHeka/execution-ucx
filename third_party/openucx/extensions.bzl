"""
Unified Communication X (UCX) is an award winning, optimized production proven-communication framework for modern, high-bandwidth and low-latency networks.
"""

load("//third_party/openucx:repositories.bzl", "openucx_repo")

def _find_modules(module_ctx):
    root = None
    our_module = None
    for mod in module_ctx.modules:
        if mod.is_root:
            root = mod
        if mod.name == "rules_nccl":
            our_module = mod
    if root == None:
        root = our_module

    return root, our_module

def _openucx_dep_impl(module_ctx):
    root, rules_ucx = _find_modules(module_ctx)

    # Handle ucx_path tags
    if hasattr(root, "tags") and hasattr(root.tags, "ucx_path"):
        path_deps = root.tags.ucx_path
    elif hasattr(rules_ucx, "tags") and hasattr(rules_ucx.tags, "ucx_path"):
        path_deps = root.tags.ucx_path
    else:
        path_deps = []

    # Handle ucx_source tags
    if hasattr(root, "tags") and hasattr(root.tags, "ucx_source"):
        source_deps = root.tags.ucx_source
    elif hasattr(rules_ucx, "tags") and hasattr(rules_ucx.tags, "ucx_source"):
        source_deps = root.tags.ucx_source
    else:
        source_deps = []

    # Handle ucx_git tags
    if hasattr(root, "tags") and hasattr(root.tags, "ucx_git"):
        git_deps = root.tags.ucx_git
    elif hasattr(rules_ucx, "tags") and hasattr(rules_ucx.tags, "ucx_git"):
        git_deps = root.tags.ucx_git
    else:
        git_deps = []

    if not path_deps and not source_deps and not git_deps:
        openucx_repo(name = "openucx")
        return

    registrations = {}

    # Process ucx_path tags
    for dep in path_deps:
        if dep.name in registrations:
            fail("Multiple conflicting dependencies declared for name {}".format(dep.name))
        registrations[dep.name] = {
            "type": "path",
            "path": dep.ucx_path,
        }

    # Process ucx_source tags
    for dep in source_deps:
        if dep.name in registrations:
            fail("Multiple conflicting dependencies declared for name {}".format(dep.name))
        registrations[dep.name] = {
            "type": "source",
            "path": dep.source_path,
        }

    # Process ucx_git tags
    for dep in git_deps:
        if dep.name in registrations:
            fail("Multiple conflicting dependencies declared for name {}".format(dep.name))
        registrations[dep.name] = {
            "type": "git",
            "tag": dep.tag,
        }

    for name, info in registrations.items():
        if info["type"] == "path":
            openucx_repo(name = name, ucx_path = info["path"])
        elif info["type"] == "source":
            openucx_repo(name = name, source_path = info["path"])
        else:  # git
            openucx_repo(name = name, git_tag = info["tag"])

ucx_path_tag = tag_class(
    attrs = {
        "name": attr.string(doc = "Name for the library repository", default = "openucx"),
        "ucx_path": attr.string(
            doc = "Path to local UCX installation.",
        ),
    },
)

ucx_source_tag = tag_class(
    attrs = {
        "name": attr.string(doc = "Name for the source repository", default = "openucx"),
        "source_path": attr.string(
            doc = "Path to local UCX source code.",
        ),
    },
)

ucx_git_tag = tag_class(
    attrs = {
        "name": attr.string(doc = "Name for the git repository", default = "openucx"),
        "tag": attr.string(
            doc = "Git tag to use.",
        ),
    },
)

openucx_dependencie = module_extension(
    implementation = _openucx_dep_impl,
    tag_classes = {
        "ucx_path": ucx_path_tag,
        "ucx_source": ucx_source_tag,
        "ucx_git": ucx_git_tag,
    },
)

def if_cuda_enabled(if_true, if_false = []):
    """Returns if_true if CUDA is enabled, otherwise returns if_false.

    Args:
        if_true: Value to return if CUDA is enabled
        if_false: Value to return if CUDA is disabled, defaults to empty list

    Returns:
        A select() expression that evaluates to if_true or if_false based on CUDA availability
    """
    return select({
        "@rules_cuda//cuda:is_enabled": if_true,
        "//conditions:default": if_false,
    })
