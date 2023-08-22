#!/bin/bash

#INPUT=1_6-31g_inp.py
INPUT=1_6-31g_inp_gpu.py

# python -u is used with tee

rm -rf profile.txt

#export OMP_NUM_THREADS=1

export PYTHONUNBUFFERED=1
COMMAND="python ${INPUT} | tee profile.txt"
#COMMAND="python my_profile.py ${INPUT} | tee profile.txt"
#COMMAND="nsys profile python ${INPUT}"

echo "COMMAND= ${COMMAND}"
${COMMAND}
