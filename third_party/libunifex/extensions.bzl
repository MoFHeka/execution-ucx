"""
The 'libunifex' project is a prototype implementation of the C++ sender/receiver async programming model that is currently being considered for standardisation.
"""

load("//third_party/libunifex:repositories.bzl", "libunifex_repo")

def _unifex_dep_impl(_ctx):
    libunifex_repo()

libunifex_dep = module_extension(
    implementation = _unifex_dep_impl,
)
