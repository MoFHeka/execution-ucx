#!/bin/bash
# This script formats files in the ucx_context directory.
# It should be run from the workspace root, e.g., via `bazel run //ucx_context:format`.
set -euo pipefail

# The directory where this BUILD.bazel file lives.
TARGET_DIR="ucx_context"
if [ -n "${BUILD_WORKSPACE_DIRECTORY:-}" ]; then
    cd ${BUILD_WORKSPACE_DIRECTORY}
fi
echo "Current directory: "$PWD

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' not found. \
          This script must be run from the workspace root."
    exit 1
fi

echo "Searching for files to format in '$TARGET_DIR'..."

# Find files and format them.
# -print0 and xargs -0 handle filenames with spaces or other special characters.
find "$TARGET_DIR" -type f \( \
    -name "*.h" \
    -o -name "*.cpp" \
    -o -name "*.hpp" \
    -o -name "*.cu" \
    -o -name "*.cuh" \
    \) -print0 | xargs -0 clang-format-16 -style=file -i

echo "Formatting complete."
