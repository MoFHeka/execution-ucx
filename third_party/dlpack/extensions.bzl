"""DLPack is a protocol for sharing arrays between deep learning frameworks."""

load("//third_party/dlpack:repositories.bzl", "dlpack_repo")

def _dlpack_dep_impl(_ctx):
    dlpack_repo()

dlpack_dep = module_extension(
    implementation = _dlpack_dep_impl,
)
