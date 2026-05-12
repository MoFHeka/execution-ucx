#!/bin/bash
# scripts/env_setup.sh
#
# Usage: source scripts/env_setup.sh
#
# Sets up the environment variables needed for local debugging and development
# without needing to rely entirely on bazel run/test sandboxing.
# Provides system paths for CUDA libraries to satisfy axon_python_runtime dependencies.

# Project Root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Check if we are sourcing the script
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
	echo "ERROR: This script must be sourced, not executed directly."
	echo "Usage: source scripts/env_setup.sh"
	exit 1
fi

# ==========================================
# 1. Setup PYTHONPATH for the built extension
# ==========================================
export PYTHONPATH="${PROJECT_ROOT}/bazel-bin/axon/python:${PROJECT_ROOT}/axon/python:${PYTHONPATH}"
echo "[ENV] PYTHONPATH configured to include bazel-bin/axon/python"

# ==========================================
# 2. Setup LD_LIBRARY_PATH for CUDA dependencies
# ==========================================
# Determine possible CUDA paths
HPC_SDK_CUDA_LIB="/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/math_libs/12.5/targets/x86_64-linux/lib"
LOCAL_CUDA_LIB="/usr/local/cuda/lib64"

# Append them if they exist
if [[ -d "$HPC_SDK_CUDA_LIB" ]]; then
	export LD_LIBRARY_PATH="${HPC_SDK_CUDA_LIB}:${LD_LIBRARY_PATH}"
	echo "[ENV] Added HPC SDK CUDA path to LD_LIBRARY_PATH: ${HPC_SDK_CUDA_LIB}"
fi

if [[ -d "$LOCAL_CUDA_LIB" ]]; then
	export LD_LIBRARY_PATH="${LOCAL_CUDA_LIB}:${LD_LIBRARY_PATH}"
	echo "[ENV] Added local CUDA path to LD_LIBRARY_PATH: ${LOCAL_CUDA_LIB}"
fi

echo "[ENV] Setup complete. You can now run python scripts directly or attach GDB."
