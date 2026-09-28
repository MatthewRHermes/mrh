#!/usr/bin/env bash

# Run the tests under tests/gpu against the CPU-only `host` libgpu backend.
#
# These exercise the libgpu call path, the Device class and the CPU kernel
# translations without needing a GPU. They are excluded from the main test.sh
# run (via `-k 'not gpu'`) because they need libgpu.so to have been built; this
# script is the entry point for the workflow job that does build it.
#
# Not a substitute for GPU testing: the host backend makes every stream
# operation synchronous and every barrier a no-op, so kernel-ordering bugs are
# invisible here, as is anything CUDA/HIP/SYCL-specific.

set -e

# tests import `mrh` (parent of the checkout) and `gpu4mrh` (under mrh/gpu)
export PYTHONPATH=${PWD%/*}:${PWD}/gpu:$PYTHONPATH

cd ./tests/gpu
pytest -q
