#!/bin/bash
# scripts/run_python_tests.sh
#
# Runs the Axon Python test suite and benchmarks.
# It automatically detects and injects the necessary system paths
# for CUDA to resolve dynamically linked dependencies that are
# explicitly excluded from the bazel runfiles.
#
# Usage:
#   ./scripts/run_python_tests.sh [benchmark|all|<bazel_test_target>]
# Example:
#   ./scripts/run_python_tests.sh benchmark
#   ./scripts/run_python_tests.sh all

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

TARGET="${1:-all}"

echo "=== Setting up environment for Testing ==="
# Source the env setup script to get LD_LIBRARY_PATH
source "${PROJECT_ROOT}/scripts/env_setup.sh"

echo ""
echo "=== Running Tests ==="
if [ "$TARGET" = "benchmark" ]; then
	echo "Running benchmark script..."
	bazel run //axon/python:benchmark_active_message
elif [ "$TARGET" = "all" ]; then
	echo "Running full python test suite..."
	bazel test //axon/python:python_tests
else
	echo "Running specific target: $TARGET"
	bazel test "$TARGET"
fi

echo "=== Testing Complete ==="
