"""Native Bazel build rule for UCX.

This module provides rules to build UCX from source using Bazel's native C++ toolchain.
"""

load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

def _native_ucx_build_impl(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )

    c_compiler_path = cc_common.get_tool_for_action(
        feature_configuration = feature_configuration,
        action_name = "c-compile",
    )

    outs = []
    for out in ctx.attr.outs:
        outs.append(ctx.actions.declare_file(out))
    if ctx.attr.is_cuda_enabled:
        for out in ctx.attr.cuda_outs:
            outs.append(ctx.actions.declare_file(out))

    env = {
        "CC": c_compiler_path,
        "CFLAGS": "-Wno-error",
        "CXXFLAGS": "-Wno-error",
        "ENABLE_CUDA": "1" if ctx.attr.is_cuda_enabled else "0",
    }

    cc_env = cc_common.get_environment_variables(
        feature_configuration = feature_configuration,
        action_name = "c-compile",
        variables = cc_common.create_compile_variables(
            feature_configuration = feature_configuration,
            cc_toolchain = cc_toolchain,
            user_compile_flags = ctx.fragments.cpp.copts + ctx.fragments.cpp.conlyopts,
        ),
    )
    env.update(cc_env)

    # Extract headers and include paths from CcInfo in srcs
    transitive_inputs = [cc_toolchain.all_files]
    cflags = env.get("CFLAGS", "")
    cxxflags = env.get("CXXFLAGS", "")
    cppflags = env.get("CPPFLAGS", "")
    for src in ctx.attr.srcs:
        if CcInfo in src:
            ctx_cc = src[CcInfo].compilation_context
            transitive_inputs.append(ctx_cc.headers)
            for inc in ctx_cc.includes.to_list():
                cflags += " -I" + inc
                cxxflags += " -I" + inc
                cppflags += " -I" + inc
            for inc in ctx_cc.system_includes.to_list():
                cflags += " -isystem " + inc
                cxxflags += " -isystem " + inc
                cppflags += " -isystem " + inc
            for inc in ctx_cc.quote_includes.to_list():
                cflags += " -iquote " + inc
                cxxflags += " -iquote " + inc
                cppflags += " -iquote " + inc

            for li in src[CcInfo].linking_context.linker_inputs.to_list():
                for lib in li.libraries:
                    if hasattr(lib, "dynamic_library") and lib.dynamic_library != None:
                        transitive_inputs.append(depset([lib.dynamic_library]))
                    if hasattr(lib, "resolved_symlink_dynamic_library") and lib.resolved_symlink_dynamic_library != None:
                        transitive_inputs.append(depset([lib.resolved_symlink_dynamic_library]))
                    if hasattr(lib, "interface_library") and lib.interface_library != None:
                        transitive_inputs.append(depset([lib.interface_library]))
                    if hasattr(lib, "resolved_symlink_interface_library") and lib.resolved_symlink_interface_library != None:
                        transitive_inputs.append(depset([lib.resolved_symlink_interface_library]))
                    if hasattr(lib, "static_library") and lib.static_library != None:
                        transitive_inputs.append(depset([lib.static_library]))

    env["CFLAGS"] = cflags
    env["CXXFLAGS"] = cxxflags
    env["CPPFLAGS"] = cppflags

    env.update(ctx.configuration.default_shell_env)

    args = ctx.actions.args()
    args.add("unused_out_dir")
    args.add(" ".join([f.path for f in outs]))
    args.add_all(ctx.attr.configure_options)

    ctx.actions.run(
        inputs = depset(direct = ctx.files.srcs, transitive = transitive_inputs),
        outputs = outs,
        executable = ctx.executable._build_script,
        arguments = [args],
        env = env,
        use_default_shell_env = True,
    )

    return DefaultInfo(files = depset(outs))

_native_ucx_build_rule = rule(
    implementation = _native_ucx_build_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "outs": attr.string_list(),
        "cuda_outs": attr.string_list(),
        "configure_options": attr.string_list(),
        "is_cuda_enabled": attr.bool(),
        "_build_script": attr.label(default = Label("//third_party/openucx:build_ucx.sh"), executable = True, cfg = "exec", allow_single_file = True),
        "_cc_toolchain": attr.label(default = Label("@bazel_tools//tools/cpp:current_cc_toolchain")),
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    fragments = ["cpp"],
)

def native_ucx_build(name, srcs, outs, cuda_outs = [], configure_options = [], debug_options = [], opt_options = [], **kwargs):
    final_opts = configure_options + select({
        ":debug_build": debug_options,
        "//conditions:default": opt_options,
    })

    is_cuda = select({
        "@rules_ml_toolchain//common:is_cuda_enabled": True,
        "//conditions:default": False,
    })

    _native_ucx_build_rule(
        name = name,
        srcs = srcs,
        outs = outs,
        cuda_outs = cuda_outs,
        configure_options = final_opts,
        is_cuda_enabled = is_cuda,
        **kwargs
    )
