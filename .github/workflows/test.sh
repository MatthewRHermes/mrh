#!/usr/bin/env bash

set -e

export PYTHONPATH=${PWD%/*}:$PYTHONPATH

# Default run is unchanged: CPU only, no gpu4mrh on the path, and tests/gpu
# deselected by the 'not gpu' term in the -k expression.
#
# RUN_GPU=1 is the libgpu pass, used after libgpu has been built. It puts
# mrh/gpu on the path so the gpu4mrh plugin is importable, and runs only
# tests/gpu -- the tests that actually exercise libgpu. The CPU suite already
# ran in the cpu job, so rerunning it here would double CI time for no extra
# coverage.
if [ "${RUN_GPU:-0}" = "1" ]; then
    export PYTHONPATH=${PWD%/*}/mrh/gpu:$PYTHONPATH
    cd ./tests/gpu
    pytest
else
    cd ./tests
    pytest -k 'not _slow and not _dupe and not gpu'
fi
