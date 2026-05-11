#!/bin/bash
set -e

# Arguments:
# $1: Output directory (where genrule expects outputs)
# $2: OUTS
# $3...: configure options

OUT_DIR=$1
OUTS=$2
shift 2

# Save the execroot path
EXECROOT=$PWD

    # Ensure we strictly use the hermetic compiler provided by Bazel, with no fallbacks.
    REAL_CC=$(find -L . -path "*/bin/clang" -type f | grep -v "sysroot" | head -n 1)
    if [ -z "$REAL_CC" ]; then
        echo "FATAL: Hermetic C compiler (clang) not found in Bazel sandbox!"
        exit 1
    fi
    export CC="$EXECROOT/$REAL_CC"

    REAL_CXX=$(find -L . -path "*/bin/clang++" -type f | grep -v "sysroot" | head -n 1)
    if [ -z "$REAL_CXX" ]; then
        echo "FATAL: Hermetic C++ compiler (clang++) not found in Bazel sandbox!"
        exit 1
    fi
    export CXX="$EXECROOT/$REAL_CXX"

    export CPP="$CC -E"
    export CXXCPP="$CXX -E"

# We need to find the source root.
SRC_ROOT=$(dirname $(find -L . -name "autogen.sh" | grep -v "install_out" | head -n 1))
while [ ! -f "$SRC_ROOT/contrib/configure-release" ]; do
    SRC_ROOT=$(dirname $SRC_ROOT)
    if [ "$SRC_ROOT" = "/" ]; then
        echo "Cannot find UCX source root"
        exit 1
    fi
done

echo "Source root: $SRC_ROOT"

# Create a clean source directory to run autogen.sh (to avoid modifying read-only cache)
SRC_TMP=$(mktemp -d)
cp -r $SRC_ROOT/* $SRC_TMP/
cp -a $SRC_ROOT/.git $SRC_TMP/ 2>/dev/null || true
cd $SRC_TMP

# Patch Makefile.am to completely skip building tests, examples, and tools
sed -i 's|src/tools/perf||g' Makefile.am
sed -i 's|src/tools/profile||g' Makefile.am
sed -i 's|test/apps||g' Makefile.am
sed -i 's|test/gtest||g' Makefile.am
sed -i 's|test/mpi||g' Makefile.am
sed -i 's|examples||g' Makefile.am
sed -i 's|bindings/go||g' Makefile.am
sed -i 's|bindings/java||g' Makefile.am

echo "Running autogen.sh"
./autogen.sh

# Go back to execroot so Bazel CC wrappers (which use relative paths) work correctly!
cd $EXECROOT

# The Bazel genrule environment natively provides CC, CXX, and CFLAGS

# NVCC path injection if CUDA is enabled
if [ "$ENABLE_CUDA" = "1" ]; then
    CUDA_OPT="--with-cuda"
    
    mkdir -p $EXECROOT/sysroot/lib
    mkdir -p $EXECROOT/sysroot/lib64
    ln -sf $EXECROOT/sysroot/lib $EXECROOT/sysroot/lib64
    
    # Resolve CUDA libraries STRICTLY from Bazel sandbox (hermetic) without host fallbacks
    for lib_prefix in "libcudart" "libnvidia-ml" "libcuda"; do
        libpath=$(find -L . -name "${lib_prefix}.so*" | grep -v "sysroot" | head -n 1)
        if [ -n "$libpath" ]; then
            # Copy instead of symlinking to avoid sandbox symlink resolution issues
            cp "$EXECROOT/$libpath" "$EXECROOT/sysroot/lib/${lib_prefix}.so"
            echo "Hermetically copied ${lib_prefix}.so from $libpath to sysroot/lib"
        else
            if [ "$lib_prefix" = "libcuda" ]; then
                echo "FATAL: Hermetic libcuda.so not found in Bazel sandbox!"
                echo "Available .so files in sandbox:"
                find -L . -name "*.so*" | head -n 50
                exit 1
            fi
        fi
    done

    # Add sysroot lib to LDFLAGS
    export LDFLAGS="$LDFLAGS -L$EXECROOT/sysroot/lib -Wl,-rpath,$EXECROOT/sysroot/lib"

    NVCC_BIN=$(find -L . -name "nvcc" -type f | grep -v "sysroot" | head -n 1)
    if [ -n "$NVCC_BIN" ]; then
        export NVCC="$EXECROOT/$NVCC_BIN"
    elif command -v nvcc &>/dev/null; then
        export NVCC=$(command -v nvcc)
    fi
else
    CUDA_OPT=""
fi

# Build sysroot for cross-compiling RDMA dependencies
echo "Building hermetic sysroot..."

find_parent_dir() {
    local found
    found=$(find -L . -path "$1" 2>/dev/null | head -n 1)
    if [ -n "$found" ]; then
        dirname "$found"
    fi
}

RDMA_CORE_DIR=$(find_parent_dir "*/libibverbs/verbs.h")
if [ -n "$RDMA_CORE_DIR" ]; then RDMA_CORE_DIR="$RDMA_CORE_DIR/.."; fi

XPMEM_DIR=$(find_parent_dir "*/include/xpmem.h")
if [ -n "$XPMEM_DIR" ]; then XPMEM_DIR="$XPMEM_DIR/.."; fi

NUMA_DIR=$(find_parent_dir "*/numa.h")

GDRCOPY_DIR=$(find_parent_dir "*/include/gdrapi.h")
if [ -n "$GDRCOPY_DIR" ]; then GDRCOPY_DIR="$GDRCOPY_DIR/.."; fi

FUSE_DIR=$(find_parent_dir "*/include/fuse.h")
if [ -n "$FUSE_DIR" ]; then FUSE_DIR="$FUSE_DIR/.."; fi

KNEM_DIR=$(find_parent_dir "*/common/knem_io.h")
if [ -n "$KNEM_DIR" ]; then KNEM_DIR="$KNEM_DIR/.."; fi

mkdir -p $EXECROOT/sysroot/include/infiniband
mkdir -p $EXECROOT/sysroot/include/rdma
mkdir -p $EXECROOT/sysroot/include/fuse3
mkdir -p $EXECROOT/sysroot/lib/pkgconfig

if [ -n "$RDMA_CORE_DIR" ] && [ -d "$RDMA_CORE_DIR/libibverbs" ]; then
    cp -r $RDMA_CORE_DIR/libibverbs/*.h $EXECROOT/sysroot/include/infiniband/ 2>/dev/null || true
    cp -r $RDMA_CORE_DIR/librdmacm/*.h $EXECROOT/sysroot/include/rdma/ 2>/dev/null || true
    cp -r $RDMA_CORE_DIR/libibumad/*.h $EXECROOT/sysroot/include/infiniband/ 2>/dev/null || true
    cp -r $RDMA_CORE_DIR/providers/mlx5/*.h $EXECROOT/sysroot/include/infiniband/ 2>/dev/null || true
    cp -r $RDMA_CORE_DIR/kernel-headers/rdma/*.h $EXECROOT/sysroot/include/rdma/ 2>/dev/null || true
    cp -r $RDMA_CORE_DIR/kernel-headers/rdma/*.h $EXECROOT/sysroot/include/infiniband/ 2>/dev/null || true
fi
if [ -n "$XPMEM_DIR" ] && [ -d "$XPMEM_DIR/include" ]; then
    cp -r $XPMEM_DIR/include/*.h $EXECROOT/sysroot/include/ 2>/dev/null || true
fi
if [ -n "$NUMA_DIR" ] && [ -f "$NUMA_DIR/numa.h" ]; then
    cp -r $NUMA_DIR/*.h $EXECROOT/sysroot/include/ 2>/dev/null || true
fi
if [ -n "$GDRCOPY_DIR" ] && [ -d "$GDRCOPY_DIR/include" ]; then
    cp -r $GDRCOPY_DIR/include/*.h $EXECROOT/sysroot/include/ 2>/dev/null || true
fi
if [ -n "$FUSE_DIR" ] && [ -d "$FUSE_DIR/include" ]; then
    echo "FUSE_DIR is $FUSE_DIR, copying headers..."
    cp -r $FUSE_DIR/include/*.h $EXECROOT/sysroot/include/fuse3/ 2>/dev/null || true
    touch $EXECROOT/sysroot/include/fuse3/libfuse_config.h
fi
if [ -n "$KNEM_DIR" ] && [ -d "$KNEM_DIR/common" ]; then
    cp -r $KNEM_DIR/common/*.h $EXECROOT/sysroot/include/ 2>/dev/null || true
fi

# Copy all CUDA headers to sysroot/include without flattening the directory structure
echo "Copying CUDA headers to sysroot/include..."
for d in $(find -L . -type d -name "include" | grep -iE "cuda|cudart|nvml" | grep -v "sysroot"); do
    cp -rn $d/* $EXECROOT/sysroot/include/ 2>/dev/null || true
done
mad_h=$(find -L . -path "*/libibmad/mad.h" | head -n 1 2>/dev/null || true)
if [ -n "$mad_h" ]; then IBMAD_DIR=$(dirname $mad_h); else IBMAD_DIR=""; fi

umad_h=$(find -L . -path "*/libibumad/umad.h" | head -n 1 2>/dev/null || true)
if [ -n "$umad_h" ]; then IBUMAD_DIR=$(dirname $umad_h); else IBUMAD_DIR=""; fi

if [ -n "$IBMAD_DIR" ] && [ -d "$IBMAD_DIR" ]; then
    mkdir -p $EXECROOT/sysroot/include/infiniband
    cp $IBMAD_DIR/*.h $EXECROOT/sysroot/include/infiniband/ 2>/dev/null || true
fi
if [ -n "$IBUMAD_DIR" ] && [ -d "$IBUMAD_DIR" ]; then
    mkdir -p $EXECROOT/sysroot/include/infiniband
    cp $IBUMAD_DIR/*.h $EXECROOT/sysroot/include/infiniband/ 2>/dev/null || true
fi

declare -A STUB_SYMBOLS=(
    ["libcuda.so"]="cuInit"
    ["libcudart.so"]="cudaGetDeviceCount"
    ["libnvidia-ml.so"]="nvmlInit"
    ["libibverbs.so"]="ibv_get_device_list"
    ["librdmacm.so"]="rdma_establish"
    ["libmlx5.so"]="mlx5dv_query_device"
    ["libxpmem.so"]="xpmem_init"
    ["libnuma.so"]="numa_available"
    ["libfuse3.so"]="fuse_session_new fuse_daemonize fuse_mount fuse_unmount fuse_open_channel"
    ["libibmad.so"]="mad_rpc_open_port mad_build_pkt"
    ["libibumad.so"]="umad_init umad_send"
    ["libgdrapi.so"]="gdr_open gdr_pin_buffer"
)

generate_stub_lib() {
    local lib_name=$1
    local symbols=$2
    local output=$3
    local stub_src
    stub_src=$(mktemp --suffix=.c)

    cat > "$stub_src" <<'STUB_HEADER'
#include <stdlib.h>
#include <stdio.h>
STUB_HEADER

    for sym in $symbols; do
        cat >> "$stub_src" <<STUB_FUNC
void ${sym}(void) {
    fprintf(stderr, "FATAL: stub function ${sym} from ${lib_name} called at runtime\\n");
    abort();
}
STUB_FUNC
    done

    echo "Generating stub library: $lib_name"
    $CC -shared -fPIC "$stub_src" -o "$output"
    rm -f "$stub_src"
}

for lib in "${!STUB_SYMBOLS[@]}"; do
    if [ ! -f "$EXECROOT/sysroot/lib/$lib" ]; then
        generate_stub_lib "$lib" "${STUB_SYMBOLS[$lib]}" "$EXECROOT/sysroot/lib/$lib"
    else
        echo "Skipping stub generation for $lib because it already exists."
    fi
done

# Generate dummy pkg-config for fuse3
cat <<EOF > $EXECROOT/sysroot/lib/pkgconfig/fuse3.pc
Name: fuse3
Description: Filesystem in Userspace
Version: 3.16.2
Libs: -L$EXECROOT/sysroot/lib -lfuse3
Cflags: -I$EXECROOT/sysroot/include/fuse3
EOF

export PKG_CONFIG_PATH="$EXECROOT/sysroot/lib/pkgconfig:$PKG_CONFIG_PATH"

export CFLAGS="$CFLAGS -I$EXECROOT/sysroot/include -I$EXECROOT/sysroot/include/fuse3 -Wno-error -gdwarf-4"
export CXXFLAGS="$CXXFLAGS -I$EXECROOT/sysroot/include -I$EXECROOT/sysroot/include/fuse3 -Wno-error -gdwarf-4"
export CPPFLAGS="$CPPFLAGS -I$EXECROOT/sysroot/include -I$EXECROOT/sysroot/include/fuse3"
export LDFLAGS="$LDFLAGS -L$EXECROOT/sysroot/lib -Wl,-rpath,$EXECROOT/sysroot/lib"
export LIBS="$LIBS -lfuse3 -libmad -libumad -lgdrapi"
export LD_LIBRARY_PATH="$EXECROOT/sysroot/lib:$LD_LIBRARY_PATH"

export FUSE3_CFLAGS="-I$EXECROOT/sysroot/include/fuse3"
export FUSE3_LIBS="-L$EXECROOT/sysroot/lib -lfuse3"

ls -la $EXECROOT/sysroot/include/cuda.h || true

# Configure UCX
$SRC_TMP/contrib/configure-release "$@" $CUDA_OPT --prefix=$PWD/install_out \
    --with-rdmacm=$EXECROOT/sysroot \
    --with-verbs=$EXECROOT/sysroot \
    --with-xpmem=$EXECROOT/sysroot \
    --with-knem=$EXECROOT/sysroot \
    --with-gdrcopy=$EXECROOT/sysroot \
    FUSE3_CFLAGS="-I$EXECROOT/sysroot/include/fuse3" \
    FUSE3_LIBS="-L$EXECROOT/sysroot/lib -lfuse3" || {
    echo "Configure failed. Extracting top and bottom of config.log:"
    head -n 200 config.log
    echo "======================================"
    tail -n 200 config.log
    exit 1
}

echo "Running make..."

make -j$(nproc)
make install

# Copy all declared outputs to their respective locations
for out in $OUTS; do
    BASENAME=$(basename $out)
    DIRNAME=$(basename $(dirname $out))

    FOUND=""
    if [ -f install_out/$DIRNAME/$BASENAME ] || [ -L install_out/$DIRNAME/$BASENAME ]; then
        FOUND="install_out/$DIRNAME/$BASENAME"
    else
        FOUND=$(find install_out -name "$BASENAME" \( -type f -o -type l \) | head -n 1)
    fi

    if [ -n "$FOUND" ]; then
        mkdir -p $(dirname $out)
        cp -L "$FOUND" $out
    else
        echo "ERROR: Declared output file $DIRNAME/$BASENAME not found after make install."
        echo "Available files in install_out:"
        find install_out \( -type f -o -type l \) | sort
        exit 1
    fi
done
