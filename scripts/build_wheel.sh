#!/bin/bash
# Production wheel build script for Axon Python package
# This script builds the standard wheel and repairs it using auditwheel.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_DIR="$ROOT_DIR/axon/python"
WHEELHOUSE_DIR="$PYTHON_DIR/wheelhouse"

echo "========================================================"
echo "    Axon Python Production Wheel Build"
echo "========================================================"

# Check for auditwheel
if ! command -v auditwheel &>/dev/null; then
	echo ">> Installing build and auditwheel..."
	python3 -m pip install build auditwheel
fi

cd "$PYTHON_DIR"

echo ">> Cleaning previous builds..."
rm -rf build dist wheelhouse
mkdir -p "$WHEELHOUSE_DIR"

echo ">> Building pure Python wheel using setuptools..."
# This will call our custom setup.py which runs bazel under the hood
python3 -m build --wheel

# Find the generated wheel
RAW_WHEEL=$(ls dist/axon_ucx-*.whl)

if [[ -z "$RAW_WHEEL" ]]; then
	echo "❌ Failed to generate wheel in dist/"
	exit 1
fi

echo ">> Generated raw wheel: $RAW_WHEEL"
echo ">> Running auditwheel repair..."
echo "Auditwheel will bundle libstdc++.so.6 and fix RPATHs for UCX libraries."

# We need to ensure auditwheel ignores some system libraries that are safe
# but generally auditwheel defaults are good enough for manylinux.
# If we are building on a modern system, we might need --plat manylinux_2_31_x86_64
auditwheel repair "$RAW_WHEEL" -w "$WHEELHOUSE_DIR"

echo "========================================================"
echo "✅ Build complete! Self-contained wheels are located in:"
echo "   $WHEELHOUSE_DIR"
echo "========================================================"
