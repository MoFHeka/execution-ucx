"""
Unified Communication X (UCX) is an award winning, optimized production proven-communication framework for modern, high-bandwidth and low-latency networks.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party/openucx:deb_import.bzl", "deb_import")
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
        "@rules_ml_toolchain//common:is_cuda_enabled": if_true,
        "//conditions:default": if_false,
    })

def _openucx_rdma_deps_impl(module_ctx):
    deb_import(
        name = "rdma_core_libs",
        urls = [
            "http://archive.ubuntu.com/ubuntu/pool/main/r/rdma-core/libibverbs1_28.0-1ubuntu1_amd64.deb",
            "http://archive.ubuntu.com/ubuntu/pool/main/r/rdma-core/libibverbs-dev_28.0-1ubuntu1_amd64.deb",
            "http://archive.ubuntu.com/ubuntu/pool/main/r/rdma-core/ibverbs-providers_28.0-1ubuntu1_amd64.deb",
            "http://archive.ubuntu.com/ubuntu/pool/main/r/rdma-core/librdmacm1_28.0-1ubuntu1_amd64.deb",
            "http://archive.ubuntu.com/ubuntu/pool/main/r/rdma-core/librdmacm-dev_28.0-1ubuntu1_amd64.deb",
            "http://archive.ubuntu.com/ubuntu/pool/main/r/rdma-core/libibumad3_28.0-1ubuntu1_amd64.deb",
            "http://archive.ubuntu.com/ubuntu/pool/main/r/rdma-core/libibumad-dev_28.0-1ubuntu1_amd64.deb",
            "http://archive.ubuntu.com/ubuntu/pool/main/r/rdma-core/libibmad5_28.0-1ubuntu1_amd64.deb",
            "http://archive.ubuntu.com/ubuntu/pool/main/r/rdma-core/libibmad-dev_28.0-1ubuntu1_amd64.deb",
        ],
        sha256s = [
            "b9e32fa96bf146e2b7ba48d0d938b53244f5566f83367bd34f549f61629ef235",
            "d5c340e4494ed4243621a20e24691291974c5e2050fe8ba0bfb64a31d878d854",
            "df4907eee048b15c2816ad72262a3867fe89810e1d0353410243ccb07d989c55",
            "d868d8d6b70d204ecaf757445a9c7fb477ee9f3817b7a7b089130117e4b6c634",
            "5fe97416beb8037abbab57563f7b32292ae3a2e3ac09a09497abda9aa95c7afa",
            "7105c9bbed97cbdbe96c05334bebec2ba9762605a13e08f6efe2777f3a288909",
            "32a71a81a9961871da79db7f9c22ccd10abb78729f87a57eaedcfdf4d1c161a9",
            "9a04c7ae852f48880484e91831e5b5e922e4e33863ac0e0da78024e7a3806745",
            "81a40afc3c2b561bd5c38bc570c6e20c7cb2bbabc41561cca49345d3fe4ca39e",
        ],
    )

    http_archive(
        name = "rdma_core",
        urls = ["https://github.com/linux-rdma/rdma-core/releases/download/v50.0/rdma-core-50.0.tar.gz"],
        strip_prefix = "rdma-core-50.0",
        build_file_content = """
filegroup(
    name = "headers",
    srcs = glob(["libibverbs/**/*.h", "librdmacm/**/*.h", "providers/mlx5/**/*.h", "kernel-headers/**/*.h", "infiniband-diags/**/*.h"]),
    visibility = ["//visibility:public"],
)
""",
    )

    deb_import(
        name = "numactl_libs",
        urls = [
            "http://archive.ubuntu.com/ubuntu/pool/main/n/numactl/libnuma1_2.0.12-1_amd64.deb",
            "http://archive.ubuntu.com/ubuntu/pool/main/n/numactl/libnuma-dev_2.0.12-1_amd64.deb",
        ],
        sha256s = [
            "0b1edf08cf9befecd21fe94e298ac25e476f87fd876ddd4adf42ef713449e637",
            "3c3c3388d50a3ff0c8841ed3c6d893e59a371743faaa916a667e72def1c44f45",
        ],
    )

    http_archive(
        name = "numactl",
        urls = ["https://github.com/numactl/numactl/releases/download/v2.0.18/numactl-2.0.18.tar.gz"],
        strip_prefix = "numactl-2.0.18",
        build_file_content = """
filegroup(
    name = "headers",
    srcs = ["numa.h", "numaif.h", "numacompat1.h"],
    visibility = ["//visibility:public"],
)
""",
    )

    deb_import(
        name = "libfuse_libs",
        urls = [
            "http://archive.ubuntu.com/ubuntu/pool/universe/f/fuse3/libfuse3-3_3.9.0-2_amd64.deb",
            "http://archive.ubuntu.com/ubuntu/pool/universe/f/fuse3/libfuse3-dev_3.9.0-2_amd64.deb",
        ],
        sha256s = [
            "b3e61e761737945ea1e8e5b19fe33d9391ed4f10ab2e09367f0216959099dfcd",
            "c2ca972c50b6794891149d8612a3523c651f3f697e3af2a1c27e4ac3bb83a446",
        ],
    )

    http_archive(
        name = "libfuse",
        urls = ["https://github.com/libfuse/libfuse/releases/download/fuse-3.16.2/fuse-3.16.2.tar.gz"],
        strip_prefix = "fuse-3.16.2",
        build_file_content = """
filegroup(
    name = "headers",
    srcs = glob(["include/**/*.h"]),
    visibility = ["//visibility:public"],
)
""",
    )

    http_archive(
        name = "xpmem",
        urls = ["https://github.com/hpc/xpmem/archive/refs/heads/master.tar.gz"],
        strip_prefix = "xpmem-master",
        build_file_content = """
filegroup(
    name = "headers",
    srcs = glob(["include/**/*.h"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)
""",
    )

    http_archive(
        name = "gdrcopy",
        urls = ["https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v2.4.tar.gz"],
        strip_prefix = "gdrcopy-2.4",
        build_file_content = """
filegroup(
    name = "headers",
    srcs = glob(["include/**/*.h"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)
""",
    )

    http_archive(
        name = "knem",
        urls = ["https://gitlab.inria.fr/knem/knem/-/archive/master/knem-master.tar.gz"],
        strip_prefix = "knem-master",
        build_file_content = """
filegroup(
    name = "headers",
    srcs = glob(["common/*.h"]),
    visibility = ["//visibility:public"],
)
""",
    )

    http_archive(
        name = "infiniband_diags",
        urls = ["https://github.com/linux-rdma/rdma-core/releases/download/v48.0/rdma-core-48.0.tar.gz"],
        strip_prefix = "rdma-core-48.0",
        build_file_content = """
filegroup(
    name = "headers",
    srcs = glob(["libibmad/*.h", "libibumad/*.h"]),
    visibility = ["//visibility:public"],
)
""",
    )

openucx_rdma_deps = module_extension(
    implementation = _openucx_rdma_deps_impl,
)
