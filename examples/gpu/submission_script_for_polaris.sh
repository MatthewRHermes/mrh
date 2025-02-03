#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug 
#PBS -A Catalyst
#PBS -l filesystems=home:grand:eagle


## example file for running a GPU accelerated LASSCF calculation on a HPC cluster

cd /lus/grand/projects/LASSCF_gpudev/valayagarawal/soft/mrh2/mrh/examples/gpu/polymer_sync  # change to working directory

# MPI example w/ 16 MPI ranks per node spread evenly across cores
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=1
NTHREADS=32
NDEPTH=${NTHREADS}

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

MPI_ARGS="-n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth "

OMP_ARGS=" "
OMP_ARGS=" --env OMP_NUM_THREADS=${NTHREADS} --env OMP_PROC_BIND=spread --env OMP_PLACES=threads "

INPUT="1_6-31g_inp_gpu.py" # can also use it through $1

export CUDA_VISIBLE_DEVICES=0  #for however many devices you want to use. 4 GPUs can be written as export CUDA_VISIBLE_DEVICES=0,1,2,3

EXE="python ${INPUT} "

time mpiexec ${MPI_ARGS} ${OMP_ARGS} ${EXE} | tee profile.txt
#--- turn this on instead of above to generate a nsys assisted profile.  
#nsys profile --stats=true -t cuda,nvtx mpiexec ${MPI_ARGS} ${OMP_ARGS} ${EXE} | tee profile.txt 
