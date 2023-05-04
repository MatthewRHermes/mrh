#!/bin/bash

export PYTHONPATH=${PWD%/*}:$PYTHONPATH
cd ./tests
pytest
