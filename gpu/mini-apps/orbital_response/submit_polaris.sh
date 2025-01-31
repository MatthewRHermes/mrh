#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug 
#PBS -A Catalyst
#PBS -l filesystems=home:grand:eagle

cd /lus/grand/projects/LASSCF_gpudev/knight/soft/mrh/gpu/mini-apps/orbital_response

# MPI example w/ 16 MPI ranks per node spread evenly across cores
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=1
NDEPTH=64
NTHREADS=64

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

MPI_ARGS="-n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth "
OMP_ARGS="--env OMP_NUM_THREADS=${NTHREADS} --env OMP_PROC_BIND=spread --env OMP_PLACES=threads"

#EXE="python main.py "
EXE="python ../my_profile.py main.py "

mpiexec ${MPI_ARGS} ${OMP_ARGS} /home/knight/repos/GettingStarted/Examples/Polaris/affinity_omp/hello_affinity 
 
#mpiexec ${MPI_ARGS} ${OMP_ARGS} ${EXE} 
mpiexec ${MPI_ARGS} ${OMP_ARGS} ${EXE} > profile.txt
