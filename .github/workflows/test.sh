#!/usr/bin/env bash

set -e

export PYTHONPATH=${PWD%/*}:$PYTHONPATH

# Default run is unchanged: CPU only, no gpu4mrh on the path, and tests/gpu
# deselected by the 'not gpu' term in the -k expression.
#
# RUN_GPU=1 is the second pass, used after libgpu has been built. It puts mrh/gpu
# on the path so the gpu4mrh plugin is importable, and drops the 'not gpu' term so
# the tests under tests/gpu are collected. Note that -k matches the *directory*
# name too, so removing that single term is what re-enables the whole suite.
if [ "${RUN_GPU:-0}" = "1" ]; then
    export PYTHONPATH=${PWD%/*}/mrh/gpu:$PYTHONPATH
    PYTEST_K='not _slow and not _dupe'
else
    PYTEST_K='not _slow and not _dupe and not gpu'
fi

cd ./tests
pytest -k "$PYTEST_K"
