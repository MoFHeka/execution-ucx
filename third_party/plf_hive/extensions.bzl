"""
plf::hive is a fork of plf::colony to match the current C++ standards proposal (https://wg21.link/p0447).
"""

load("//third_party/plf_hive:repositories.bzl", "plf_hive_repo")

def _hive_dep_impl(_ctx):
    plf_hive_repo()

plf_hive_dep = module_extension(
    implementation = _hive_dep_impl,
)
