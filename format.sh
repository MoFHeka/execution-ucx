#!/bin/bash
# This script formats all source files in the workspace.
# It should be run from the workspace root, e.g., via `bazel run //:format`.
set -euo pipefail

if [ -n "${BUILD_WORKSPACE_DIRECTORY:-}" ]; then
	cd ${BUILD_WORKSPACE_DIRECTORY}
fi
echo "Current directory: "$PWD

echo "Searching for files to format in workspace..."

# Find files and format them.
# -print0 and xargs -0 handle filenames with spaces or other special characters.
# Exclude bazel-* directories to avoid formatting generated files.
find . -type d -name "bazel-*" -prune -o -type f \( \
	-name "*.h" \
	-o -name "*.cpp" \
	-o -name "*.hpp" \
	-o -name "*.cu" \
	-o -name "*.cuh" \
	\) -print0 | xargs -0 clang-format-16 -style=file -i

find . -type f \( \
	-name "*.bazel" \
	-o -name "*.bzl" \
	-o -name "BUILD" \
	-o -name "WORKSPACE" \
	\) -exec buildifier {} \;

# Format Python files using Black with Google style (line length 100)
find . -type d -name "bazel-*" -prune -o -type f \( \
	-name "*.py" \
	\) -print0 | xargs -0 black --target-version=py311

echo "Formatting complete."
