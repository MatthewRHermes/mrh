#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug 
#PBS -A Catalyst
#PBS -l filesystems=home:grand:eagle

cd /lus/grand/projects/LASSCF_gpudev/knight/soft/mrh/gpu/mini-apps/openmp/python

# MPI example w/ 16 MPI ranks per node spread evenly across cores
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=1
NTHREADS=32
NDEPTH=${NTHREADS}

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

#MPI_ARGS="-n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} "
MPI_ARGS="-n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth "

OMP_ARGS=" "
#OMP_ARGS="--env OMP_NUM_THREADS=${NTHREADS} "
OMP_ARGS=" --env OMP_NUM_THREADS=${NTHREADS} --env OMP_PROC_BIND=spread --env OMP_PLACES=threads "
#OMP_ARGS+=" --env OMP_WAIT_POLICY=ACTIVE "

INPUT="main.py"

#export CUDA_VISIBLE_DEVICES=0

EXE="python ${INPUT} "

#python -m cProfile -o out.prof ${INPUT}
#{ time ${EXE} ;} 2>&1 | tee profile.txt
{ time mpiexec ${MPI_ARGS} ${OMP_ARGS} ${EXE} ;} 2>&1 | tee profile.txt
#nsys profile --stats=true -t cuda,nvtx mpiexec ${MPI_ARGS} ${OMP_ARGS} ${EXE} 2>&1 | tee profile.txt
