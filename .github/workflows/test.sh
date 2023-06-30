#!/usr/bin/env bash

set -e

export PYTHONPATH=${PWD%/*}:$PYTHONPATH
cd ./tests/lasscf
pytest test_lasci.py
