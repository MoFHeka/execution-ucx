#!/bin/bash
# Test runner script for Python tests

set -e

# Get the test file from arguments
TEST_FILE="$1"

# Find the axon_runtime.so (or axon_python_runtime.so) location
# In Bazel, it should be in the runfiles
RUNTIME_SO=$(find . -name "axon_runtime.so" 2>/dev/null | head -1)

if [ -z "$RUNTIME_SO" ]; then
  # Fallback: try to find it in the current directory structure
  if [ -f "bazel-bin/axon/python/axon_runtime.so" ]; then
    RUNTIME_SO="bazel-bin/axon/python/axon_runtime.so"
  elif [ -f "axon/python/axon_runtime.so" ]; then
    RUNTIME_SO="axon/python/axon_runtime.so"
  elif [ -f "bazel-bin/axon/python/axon_python_runtime.so" ]; then
    RUNTIME_SO="bazel-bin/axon/python/axon_python_runtime.so"
  else
    echo "Warning: axon_runtime.so not found. Trying to import axon_runtime anyway..."
    RUNTIME_SO=""
  fi
fi

# Set PYTHONPATH to include the directory with axon_python_runtime.so
if [ -n "$RUNTIME_SO" ]; then
  RUNTIME_DIR=$(dirname "$RUNTIME_SO")
  export PYTHONPATH="$RUNTIME_DIR:$PYTHONPATH"
fi

# Check if pytest is available (should be provided by Bazel)
if ! python3 -c "import pytest" 2>/dev/null; then
  echo "Error: pytest not found. Please ensure it's installed via Bazel."
  exit 1
fi

# Run pytest
python3 -m pytest "$TEST_FILE" -v

