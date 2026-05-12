#!/bin/bash
# scripts/build_python_wheels.sh
#
# Builds the axon Python wheel and processes it with auditwheel
# to create a manylinux-compatible wheel while explicitly EXCLUDING
# CUDA and CUDNN libraries, ensuring a lean package.
#
# Usage:
#   ./scripts/build_python_wheels.sh [bazel_python_config]
# Example:
#   ./scripts/build_python_wheels.sh python310

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PY_CONFIG="${1:-""}"

# 1. Build the wheel via Bazel
echo "=== Building Wheel via Bazel ==="
if [ -n "$PY_CONFIG" ]; then
	echo "Using Python config: --config=${PY_CONFIG}"
	bazel build //axon/python:wheel --config=${PY_CONFIG}
else
	echo "Using default Python config"
	bazel build //axon/python:wheel
fi

# 2. Locate the built wheel
WHEEL_DIR="bazel-bin/axon/python"
# Extract the exact wheel name generated
WHEEL_FILE=$(find "${WHEEL_DIR}" -maxdepth 1 -name "*.whl" | head -n 1)

if [ -z "$WHEEL_FILE" ]; then
	echo "ERROR: Could not find generated wheel in ${WHEEL_DIR}."
	exit 1
fi
echo "Found bazel wheel: ${WHEEL_FILE}"

# 3. Check for auditwheel
if ! command -v auditwheel &>/dev/null; then
	echo "WARNING: auditwheel not found. Attempting to install via pip..."
	pip install auditwheel patchelf
fi

# 4. Repair the wheel with auditwheel
echo "=== Repairing Wheel via Auditwheel ==="
mkdir -p dist

# Dynamically exclude standard CUDA and RDMA libraries to keep the wheel lean
# and rely on the host's system configuration or pip-installed nvidia packages.
# We parse the actual SONAMEs from the compiled extension so it's version-agnostic.
SO_FILE="${PROJECT_ROOT}/bazel-bin/axon/python/axon/axon.so"
DEPS=$(patchelf --print-needed "${SO_FILE}" || true)

EXCLUDES=()
for dep in $DEPS; do
	# Match CUDA, cuDNN, RDMA, IB, and other system/hardware specific libraries
	if [[ "$dep" == libcu* ]] ||
		[[ "$dep" == libnvrtc* ]] ||
		[[ "$dep" == lib*rdma* ]] ||
		[[ "$dep" == libib* ]] ||
		[[ "$dep" == libhfi* ]] ||
		[[ "$dep" == libbnxt* ]] ||
		[[ "$dep" == libcxgb* ]] ||
		[[ "$dep" == libhns* ]] ||
		[[ "$dep" == libi40* ]] ||
		[[ "$dep" == libmthca* ]] ||
		[[ "$dep" == libocrdma* ]] ||
		[[ "$dep" == libqedr* ]] ||
		[[ "$dep" == librspreload* ]] ||
		[[ "$dep" == librxe* ]] ||
		[[ "$dep" == libsiw* ]] ||
		[[ "$dep" == libvmw* ]] ||
		[[ "$dep" == libxpmem* ]] ||
		[[ "$dep" == libgdrapi* ]]; then
		EXCLUDES+=("--exclude" "$dep")
		echo "Excluding dynamically detected system library: $dep"
	fi
done

# Add UCX libs to LD_LIBRARY_PATH so auditwheel can find and bundle them
export LD_LIBRARY_PATH="${PROJECT_ROOT}/bazel-bin/axon/python/axon/libs:${LD_LIBRARY_PATH}"

# Run auditwheel repair
auditwheel repair "${WHEEL_FILE}" "${EXCLUDES[@]}" --wheel-dir dist/

echo "=== Packaging Complete ==="
echo "Repaired wheels available in dist/:"
ls -lh dist/
