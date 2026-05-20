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

# Keep clang from picking up an incompatible linker from the invoking environment
# (for example, conda's ld when Bazel is launched via `conda run`).
REAL_LD=$(find -L . -path "*/bin/ld.lld" -type f | grep -v "sysroot" | head -n 1)
if [ -z "$REAL_LD" ]; then
	echo "FATAL: Hermetic LLVM linker (ld.lld) not found in Bazel sandbox!"
	exit 1
fi
export LD="$EXECROOT/$REAL_LD"
export PATH="$(dirname "$LD"):$PATH"
export LDFLAGS="$LDFLAGS -fuse-ld=$LD"

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
SRC_TMP=$(mktemp -d "$EXECROOT/ucx_src_tmp.XXXXXX")
trap 'rm -rf "$SRC_TMP"' EXIT
cp -r $SRC_ROOT/* $SRC_TMP/
cp -a $SRC_ROOT/.git $SRC_TMP/ 2>/dev/null || true
cd $SRC_TMP

# We use sed to remove tests, examples, tools, and bindings from Makefile.am and configure.ac.
# This prevents `autoreconf` (called by autogen.sh) from parsing these components and searching
# for missing M4 macros (like AM_PATH_GTEST or AM_PATH_OMPI) on the host system, which would
# otherwise cause fatal errors during the hermetic build process.
# NOTE: Using standard configure options (e.g. --disable-tests) is insufficient here because
# autoreconf parses these directories and throws fatal errors on missing M4 macros BEFORE configure
# can execute. Thus, this specialized sed-strip workaround is strictly required for Bazel hermetic sandbox build.
sed -i 's|src/tools/perf||g' Makefile.am
sed -i 's|src/tools/profile||g' Makefile.am
sed -i 's|test/apps||g' Makefile.am
sed -i 's|test/gtest||g' Makefile.am
sed -i 's|test/mpi||g' Makefile.am
sed -i 's|examples||g' Makefile.am
sed -i 's|bindings/go||g' Makefile.am
sed -i 's|bindings/java||g' Makefile.am

find . -name "configure.ac" -o -name "configure.m4" | xargs sed -i 's|AC_CONFIG_FILES(\[test/.*\])||g'
find . -name "configure.ac" -o -name "configure.m4" | xargs sed -i 's|AC_CONFIG_FILES(\[examples/.*\])||g'
find . -name "configure.ac" -o -name "configure.m4" | xargs sed -i 's|AC_CONFIG_FILES(\[bindings/.*\])||g'

sed -i 's|src/tools/perf/Makefile||g' configure.ac
sed -i 's|src/tools/profile/Makefile||g' configure.ac
sed -i 's|test/apps/Makefile||g' configure.ac
sed -i 's|test/apps/iodemo/Makefile||g' configure.ac
sed -i 's|test/apps/profiling/Makefile||g' configure.ac
sed -i 's|test/mpi/Makefile||g' configure.ac
sed -i 's|bindings/go/Makefile||g' configure.ac
sed -i 's|bindings/java/Makefile||g' configure.ac
sed -i 's|bindings/java/pom.xml||g' configure.ac
sed -i 's|bindings/java/src/main/native/Makefile||g' configure.ac
sed -i 's|examples/Makefile||g' configure.ac
sed -i 's|test/mpi/run_mpi.sh||g' configure.ac

echo "Running autogen.sh"
./autogen.sh

# Go back to execroot so Bazel CC wrappers (which use relative paths) work correctly!
cd $EXECROOT

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
				exit 1
			fi
		fi
	done

	# Add sysroot lib to LDFLAGS
	export LDFLAGS="$LDFLAGS -L$EXECROOT/sysroot/lib -Wl,-rpath,$EXECROOT/sysroot/lib"

	NVCC_BIN=$(find -L . -name "nvcc" -type f | grep -v "sysroot" | head -n 1)
	if [ -n "$NVCC_BIN" ]; then
		export NVCC="$EXECROOT/$NVCC_BIN"
	else
		echo "FATAL: Hermetic nvcc not found in Bazel sandbox!"
		exit 1
	fi
else
	CUDA_OPT=""
fi

# ==============================================================================
# Build hermetic sysroot — all libraries come from Bazel-managed downloads
# ==============================================================================
echo "Building hermetic sysroot..."
mkdir -p $EXECROOT/sysroot/include/infiniband
mkdir -p $EXECROOT/sysroot/include/rdma
mkdir -p $EXECROOT/sysroot/include/fuse3
mkdir -p $EXECROOT/sysroot/lib/pkgconfig

find_parent_dir() {
	local found
	found=$(find -L . -path "$1" 2>/dev/null | head -n 1)
	if [ -n "$found" ]; then
		dirname "$found"
	fi
}

# --- Step 1: Import pre-built binary libraries from .deb packages ---
echo "Importing pre-built libraries from .deb packages..."
for pkg_pattern in "rdma_core_libs" "numactl_libs" "libfuse_libs"; do
	for so_file in $(find -L . -path "*${pkg_pattern}/lib/*" 2>/dev/null); do
		basename_so=$(basename "$so_file")
		cp -a -L "$so_file" "$EXECROOT/sysroot/lib/$basename_so" 2>/dev/null || true
		echo "  Imported: $basename_so (from $pkg_pattern)"
	done
done

# Create linker symlinks if only versioned .so exists
for lib in libibverbs librdmacm libnuma libmlx5 libibumad libfuse3 libibmad; do
	if [ ! -f "$EXECROOT/sysroot/lib/${lib}.so" ]; then
		versioned=$(ls "$EXECROOT/sysroot/lib/${lib}.so."* 2>/dev/null | head -1)
		if [ -n "$versioned" ]; then
			ln -sf "$(basename $versioned)" "$EXECROOT/sysroot/lib/${lib}.so"
			echo "  Created linker symlink: ${lib}.so -> $(basename $versioned)"
		fi
	fi
done

echo "Pre-built libraries imported. Contents of sysroot/lib:"
ls -la $EXECROOT/sysroot/lib/*.so* 2>/dev/null || echo "  (none)"

# --- Step 2: Copy headers from Bazel-provided source downloads ---
RDMA_CORE_DIR=$(find_parent_dir "*/libibverbs/verbs.h")
if [ -n "$RDMA_CORE_DIR" ]; then RDMA_CORE_DIR="$RDMA_CORE_DIR/.."; fi

NUMA_DIR=$(find_parent_dir "*/numa.h")

KNEM_DIR=$(find_parent_dir "*/common/knem_io.h")
if [ -n "$KNEM_DIR" ]; then KNEM_DIR="$KNEM_DIR/.."; fi

XPMEM_DIR=$(find_parent_dir "*/include/xpmem.h")
if [ -n "$XPMEM_DIR" ]; then XPMEM_DIR="$XPMEM_DIR/.."; fi

GDRCOPY_DIR=$(find_parent_dir "*/include/gdrapi.h")
if [ -n "$GDRCOPY_DIR" ]; then GDRCOPY_DIR="$GDRCOPY_DIR/.."; fi

FUSE_DIR=$(find_parent_dir "*/include/fuse.h")
if [ -n "$FUSE_DIR" ]; then FUSE_DIR="$FUSE_DIR/.."; fi

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
	cp -r $FUSE_DIR/include/*.h $EXECROOT/sysroot/include/fuse3/ 2>/dev/null || true
	touch $EXECROOT/sysroot/include/fuse3/libfuse_config.h
fi
if [ -n "$KNEM_DIR" ] && [ -d "$KNEM_DIR/common" ]; then
	cp -r $KNEM_DIR/common/*.h $EXECROOT/sysroot/include/ 2>/dev/null || true
fi

# Also copy headers from .deb packages (infiniband/, rdma/)
for pkg_pattern in "rdma_core_libs" "numactl_libs" "libfuse_libs"; do
	deb_inc=$(find -L . -path "*${pkg_pattern}/include" -type d 2>/dev/null | head -1)
	if [ -n "$deb_inc" ]; then
		cp -rn $deb_inc/* $EXECROOT/sysroot/include/ 2>/dev/null || true
		echo "  Copied headers from $pkg_pattern"
	fi
done

# Copy CUDA headers
echo "Copying CUDA headers to sysroot/include..."
for d in $(find -L . -type d -name "include" | grep -iE "cuda|cudart|nvml" | grep -v "sysroot"); do
	cp -rn $d/* $EXECROOT/sysroot/include/ 2>/dev/null || true
done

# Copy ibmad/ibumad headers
mad_h=$(find -L . -path "*/libibmad/mad.h" | head -n 1 2>/dev/null || true)
if [ -n "$mad_h" ]; then IBMAD_DIR=$(dirname $mad_h); else IBMAD_DIR=""; fi
umad_h=$(find -L . -path "*/libibumad/umad.h" | head -n 1 2>/dev/null || true)
if [ -n "$umad_h" ]; then IBUMAD_DIR=$(dirname $umad_h); else IBUMAD_DIR=""; fi

if [ -n "$IBMAD_DIR" ] && [ -d "$IBMAD_DIR" ]; then
	cp $IBMAD_DIR/*.h $EXECROOT/sysroot/include/infiniband/ 2>/dev/null || true
fi
if [ -n "$IBUMAD_DIR" ] && [ -d "$IBUMAD_DIR" ]; then
	cp $IBUMAD_DIR/*.h $EXECROOT/sysroot/include/infiniband/ 2>/dev/null || true
fi

# --- Step 3: Build source-only libraries (no pre-built binaries available) ---
echo "Building libraries from source (no pre-built binaries available)..."

# Build libxpmem.so from source (single file, no binary package exists)
XPMEM_SRC=$(find -L . -path "*xpmem*/lib/libxpmem.c" | head -n 1)
if [ -n "$XPMEM_SRC" ]; then
	XPMEM_LIB_DIR=$(dirname "$XPMEM_SRC")
	XPMEM_INC=$(find -L . -path "*xpmem*/include" -type d | head -n 1)
	if [ -n "$XPMEM_INC" ]; then
		echo "  Building libxpmem.so from source..."
		$CC -shared -fPIC $LDFLAGS -Wno-error \
			-I"$XPMEM_INC" -I"$XPMEM_LIB_DIR" \
			-o "$EXECROOT/sysroot/lib/libxpmem.so" \
			"$XPMEM_SRC"
		echo "  Built libxpmem.so"
	fi
fi

# Build libgdrapi.so from source (no binary package exists)
GDRCOPY_SRC=$(find -L . -path "*gdrcopy*/src/gdrapi.c" | head -n 1)
if [ -n "$GDRCOPY_SRC" ]; then
	GDRCOPY_SRC_DIR=$(dirname "$GDRCOPY_SRC")
	GDRCOPY_INC=$(find -L . -path "*gdrcopy*/include" -type d | head -n 1)
	if [ -n "$GDRCOPY_INC" ]; then
		echo "  Building libgdrapi.so from source..."
		GDRCOPY_COMMON="-fPIC -Wno-error -I$GDRCOPY_INC -I$GDRCOPY_SRC_DIR/gdrdrv -DGDRAPI_VERSION_MAJOR=2 -DGDRAPI_VERSION_MINOR=4"
		GDRCOPY_OBJ_DIR="$EXECROOT/tmp_gdrcopy"
		mkdir -p "$GDRCOPY_OBJ_DIR"
		$CC $GDRCOPY_COMMON -c -o "$GDRCOPY_OBJ_DIR/gdrapi.o" "$GDRCOPY_SRC_DIR/gdrapi.c"
		$CC $GDRCOPY_COMMON -mavx -c -o "$GDRCOPY_OBJ_DIR/memcpy_avx.o" "$GDRCOPY_SRC_DIR/memcpy_avx.c"
		$CC $GDRCOPY_COMMON -msse -c -o "$GDRCOPY_OBJ_DIR/memcpy_sse.o" "$GDRCOPY_SRC_DIR/memcpy_sse.c"
		$CC $GDRCOPY_COMMON -msse4.1 -c -o "$GDRCOPY_OBJ_DIR/memcpy_sse41.o" "$GDRCOPY_SRC_DIR/memcpy_sse41.c"
		$CC -shared -o "$EXECROOT/sysroot/lib/libgdrapi.so" \
			"$GDRCOPY_OBJ_DIR/gdrapi.o" "$GDRCOPY_OBJ_DIR/memcpy_avx.o" "$GDRCOPY_OBJ_DIR/memcpy_sse.o" "$GDRCOPY_OBJ_DIR/memcpy_sse41.o"
		echo "  Built libgdrapi.so"
	fi
fi

# Generate pkg-config for fuse3
cat <<EOF >$EXECROOT/sysroot/lib/pkgconfig/fuse3.pc
Name: fuse3
Description: Filesystem in Userspace
Version: 3.9.0
Libs: -L$EXECROOT/sysroot/lib -lfuse3
Cflags: -I$EXECROOT/sysroot/include/fuse3
EOF

export PKG_CONFIG_PATH="$EXECROOT/sysroot/lib/pkgconfig:$PKG_CONFIG_PATH"

export CFLAGS="$CFLAGS -I$EXECROOT/sysroot/include -I$EXECROOT/sysroot/include/fuse3 -Wno-error -gdwarf-4"
export CXXFLAGS="$CXXFLAGS -I$EXECROOT/sysroot/include -I$EXECROOT/sysroot/include/fuse3 -Wno-error -gdwarf-4"
export CPPFLAGS="$CPPFLAGS -I$EXECROOT/sysroot/include -I$EXECROOT/sysroot/include/fuse3"
export LDFLAGS="$LDFLAGS -L$EXECROOT/sysroot/lib -Wl,-rpath,$EXECROOT/sysroot/lib"

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
	--with-mad=$EXECROOT/sysroot \
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

# UCX outputs live in install_out/, dependency libs live in sysroot/lib/
for out in $OUTS; do
	BASENAME=$(basename $out)
	DIRNAME=$(basename $(dirname $out))

	FOUND=""
	if [ -f install_out/$DIRNAME/$BASENAME ] || [ -L install_out/$DIRNAME/$BASENAME ]; then
		FOUND="install_out/$DIRNAME/$BASENAME"
	elif [ -f "$EXECROOT/sysroot/lib/$BASENAME" ]; then
		FOUND="$EXECROOT/sysroot/lib/$BASENAME"
	else
		FOUND=$(find install_out -name "$BASENAME" \( -type f -o -type l \) | head -n 1)
	fi

	if [ -z "$FOUND" ]; then
		echo "ERROR: Declared output $BASENAME not found in install_out/ or sysroot/lib/"
		exit 1
	fi

	mkdir -p $(dirname $out)
	cp -L "$FOUND" $out
done
