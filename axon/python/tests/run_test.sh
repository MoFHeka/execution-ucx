#!/bin/bash
# Test runner script for Python tests

set -e

# Get the test file from arguments
TEST_FILE="$1"

# Find the _axon.so location using Bazel cquery for precision
# This returns the exact path where Bazel outputs the library
RUNTIME_SO=$(bazel cquery "//axon/python:axon_python_lib" --output=files 2>/dev/null | head -1)

if [ -z "$RUNTIME_SO" ] || [ ! -f "$RUNTIME_SO" ]; then
	# Fallback: try to find it in common Bazel output locations
	if [ -f "bazel-bin/axon/python/axon/_axon.so" ]; then
		RUNTIME_SO="bazel-bin/axon/python/axon/_axon.so"
	elif [ -f "axon/python/axon/_axon.so" ]; then
		RUNTIME_SO="axon/python/axon/_axon.so"
	elif [ -f "bazel-bin/axon/python/libaxon_python_runtime.so" ]; then
		RUNTIME_SO="bazel-bin/axon/python/libaxon_python_runtime.so"
	else
		# Last resort: search for the file
		RUNTIME_SO=$(find . -name "_axon.so" 2>/dev/null | head -1)
	fi
fi

# Pass the library paths to the test process via environment variables
# This avoids the need for polluting the source tree with symlinks
if [ -n "$RUNTIME_SO" ]; then
	export AXON_EXTENSION_PATH=$(readlink -f "$RUNTIME_SO")
	RUNTIME_DIR=$(dirname "$RUNTIME_SO")
	if [ -d "$RUNTIME_DIR/libs" ]; then
		export AXON_LIBS_PATH=$(readlink -f "$RUNTIME_DIR/libs")
		export LD_LIBRARY_PATH="$AXON_LIBS_PATH:$LD_LIBRARY_PATH"
	fi
	if [ -d "$RUNTIME_DIR/ucx_modules" ]; then
		export AXON_UCX_MODULES_PATH=$(readlink -f "$RUNTIME_DIR/ucx_modules")
	fi
fi

# Find CUDA redist libraries from Bazel external directory and add to LD_LIBRARY_PATH
# This is needed because the extension may depend on specific CUDA versions downloaded by Bazel
OUTPUT_BASE=$(bazel info output_base 2>/dev/null)
if [ -n "$OUTPUT_BASE" ] && [ -d "$OUTPUT_BASE/external" ]; then
	# Look for lib directories in cuda-related external repos
	CUDA_REDIST_LIBS=$(find "$OUTPUT_BASE/external" -maxdepth 3 -name "lib" -path "*cuda*/lib" 2>/dev/null)
	for LIB_DIR in $CUDA_REDIST_LIBS; do
		if [ -d "$LIB_DIR" ]; then
			export LD_LIBRARY_PATH="$LIB_DIR:$LD_LIBRARY_PATH"
		fi
	done
fi

# Set default UCX network device to loopback for local tests if not specified
if [ -z "$UCX_NET_DEVICES" ]; then
	export UCX_NET_DEVICES=lo
fi

# Set PYTHONPATH
export PYTHONPATH="$(pwd)/axon/python:$(pwd)/axon/python/tests:$PYTHONPATH"

# Check if pytest is available
if ! python3 -c "import pytest" 2>/dev/null; then
	echo "Error: pytest not found. Please ensure it's installed."
	exit 1
fi

# Run pytest
echo "Using PYTHONPATH: $PYTHONPATH"
python3 -m pytest "$TEST_FILE" -v
