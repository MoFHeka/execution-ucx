load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")

def _test_cuda_impl(ctx):
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

    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.write(
        output = script,
        content = """#!/bin/bash
cat << 'EOF' > conftest.c
#include <cuda.h>
int main() { cuDeviceGetUuid(0, 0); return 0; }
EOF
echo "Testing compilation and link..."
echo "LDFLAGS: $LDFLAGS"
ls -la external/rules_ml_toolchain~~cuda_redist_init_ext~cuda_cudart/lib || true
$CC $CFLAGS $LDFLAGS -o conftest conftest.c -lcuda -Wl,--trace
if [ $? -eq 0 ]; then
    echo "SUCCESS_CUDA_LINK"
else
    echo "FAILED_CUDA_LINK"
fi
exit 0
""",
        is_executable = True,
    )

    out_file = ctx.actions.declare_file(ctx.label.name + "_out.txt")

    cc_env = cc_common.get_environment_variables(
        feature_configuration = feature_configuration,
        action_name = "c-compile",
        variables = cc_common.create_compile_variables(
            feature_configuration = feature_configuration,
            cc_toolchain = cc_toolchain,
            user_compile_flags = ctx.fragments.cpp.copts + ctx.fragments.cpp.conlyopts,
        ),
    )

    transitive_inputs = [cc_toolchain.all_files]
    cflags = cc_env.get("CFLAGS", "")
    ldflags = cc_env.get("LDFLAGS", "")
    for src in ctx.attr.deps:
        ctx_cc = src[CcInfo].compilation_context
        transitive_inputs.append(ctx_cc.headers)
        for inc in ctx_cc.includes.to_list():
            cflags += " -I" + inc
        for inc in ctx_cc.system_includes.to_list():
            cflags += " -isystem " + inc

        for li in src[CcInfo].linking_context.linker_inputs.to_list():
            for lib in li.libraries:
                for l in [lib.dynamic_library, lib.static_library, lib.pic_static_library, lib.resolved_symlink_dynamic_library]:
                    if l != None:
                        transitive_inputs.append(depset([l]))
                        libdir = l.dirname
                        ldflags += " -L" + libdir + " -Wl,-rpath," + libdir

    env = dict(cc_env)
    env["CC"] = c_compiler_path
    env["CFLAGS"] = cflags
    env["LDFLAGS"] = ldflags

    ctx.actions.run(
        inputs = depset(direct = [script], transitive = transitive_inputs),
        outputs = [out_file],
        executable = script,
        env = env,
        use_default_shell_env = True,
        arguments = [],
    )

    return [DefaultInfo(files = depset([out_file]))]

test_cuda = rule(
    implementation = _test_cuda_impl,
    attrs = {
        "deps": attr.label_list(providers = [CcInfo]),
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    fragments = ["cpp"],
)
