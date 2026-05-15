#!/bin/bash
# Test runner script for Python tests

set -e

# Convert test file to absolute path before changing directory
if [ -n "$1" ]; then
	TEST_FILE="$(realpath "$1")"
else
	TEST_FILE=""
fi

# Change to project root so paths and bazel commands work from anywhere
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

# Find the _axon.so location using Bazel cquery for precision
# This returns the exact path where Bazel outputs the library
RUNTIME_SO=$(bazel cquery "//axon/python:axon_python_lib" --output=files 2>/dev/null | head -1)

if [ -z "$RUNTIME_SO" ] || [ ! -f "$RUNTIME_SO" ]; then
	echo "ERROR: Could not find _axon.so using bazel cquery."
	echo "Please ensure the extension is built via: bazel build //axon/python:axon_python_lib"
	exit 1
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
