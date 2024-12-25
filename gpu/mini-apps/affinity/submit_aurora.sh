#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug 
#PBS -A Catalyst
#PBS -l filesystems=home:grand:eagle

#cd ${PBS_O_WORKDIR}

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=1
NDEPTH=32
NTHREADS=32

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

export ONEAPI_DEVICE_SELECTOR=level_zero:*.* 
#export ONEAPI_DEVICE_SELECTOR=level_zero:0.0

MPI_ARGS="-n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth "
OMP_ARGS="--env OMP_NUM_THREADS=${NTHREADS} --env OMP_PROC_BIND=spread --env OMP_PLACES=cores "

mpiexec ${MPI_ARGS} ${OMP_ARGS} ./a.out
