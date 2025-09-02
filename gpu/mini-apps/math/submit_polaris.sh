#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug 
#PBS -A LASSCF_gpudev
#PBS -l filesystems=home:grand

cd /grand/LASSCF_gpudev/valayagarawal/soft/original/mrh2/mrh/gpu/mini-apps/math
. /grand/LASSCF_gpudev/valayagarawal/scripts/setup_polaris_3.sh
#. /grand/LASSCF_gpudev/knight/scripts/polaris/setup_polaris.sh

# MPI example w/ 16 MPI ranks per node spread evenly across cores
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=1
NTHREADS=32
NDEPTH=32

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

MPI_ARGS="-n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth "

OMP_ARGS=" "
#OMP_ARGS="--env OMP_NUM_THREADS=${NTHREADS} "
#OMP_ARGS=" --env OMP_NUM_THREADS=${NTHREADS} --env OMP_PLACES=threads "
OMP_ARGS=" --env OMP_NUM_THREADS=${NTHREADS} --env OMP_PROC_BIND=spread --env OMP_PLACES=cores "
#OMP_ARGS+=" --env OMP_WAIT_POLICY=ACTIVE "
#OMP_ARGS+=" --env OMP_WAIT_POLICY=PASSIVE "

#OMP_ARGS+=" --env OMP_STACKSIZE=1M "

echo "OMP_ARGS= ${OMP_ARGS}"

export CUDA_VISIBLE_DEVICES=3

EXE="./a.out "

ARGS_LIST[0]=" -num_iter 10 -num_repeat 10    gemm_batch T T 674 674 674 674 674 674 1 0 240 " 
ARGS_LIST[1]=" -num_iter 10 -num_repeat 10    gemm N N 674 674 161760 674 161760 674 1 0     " 
ARGS_LIST[2]=" -num_iter 10 -num_repeat 10    gemm N N 674 674 161760 674 161760 674 1 1     " 
ARGS_LIST[3]=" -num_iter 10 -num_repeat 10   gemm_batch T T 674 674 674 674 674 674 1 0 228  " 
ARGS_LIST[4]=" -num_iter 10 -num_repeat 10   gemm N N 674 674 153672 674 153672 674 1 1      " 
ARGS_LIST[5]=" -num_iter 10 -num_repeat 10   gemm_batch N N 674 674 674 674 674 674 1 0 240  " 
ARGS_LIST[6]=" -num_iter 10 -num_repeat 10   gemm_batch T N 674 674 674 674 674 674 1 0 240  " 
ARGS_LIST[7]=" -num_iter 10 -num_repeat 10   gemm_batch N T 1 1 240 1 1 1 1 0 67400          " 
ARGS_LIST[8]=" -num_iter 10 -num_repeat 10   gemm N T 100 674 240 674 674 100 1 0            " 
ARGS_LIST[9]=" -num_iter 10 -num_repeat 10   gemm N N 100 454276 240 100 240 100 1 0         " 
ARGS_LIST[10]=" -num_iter 10 -num_repeat 10   gemm N T 6740 6740 240 6740 6740 6740 1 0       " 
ARGS_LIST[11]=" -num_iter 10 -num_repeat 10   gemm_batch N T 1 1 240 1 1 1 1 1 67400          " 
ARGS_LIST[12]=" -num_iter 10 -num_repeat 10   gemm N T 100 674 240 674 674 100 1 1            " 
ARGS_LIST[13]=" -num_iter 10 -num_repeat 10   gemm N N 100 454276 240 100 240 100 1 1         " 
ARGS_LIST[14]=" -num_iter 10 -num_repeat 10   gemm N T 6740 6740 240 6740 6740 6740 1 1       " 
ARGS_LIST[15]=" -num_iter 10 -num_repeat 10   gemm_batch N N 674 674 674 674 674 674 1 0 228  " 
ARGS_LIST[16]=" -num_iter 10 -num_repeat 10   gemm_batch T N 674 674 674 674 674 674 1 0 228  " 
ARGS_LIST[17]=" -num_iter 10 -num_repeat 10   gemm_batch N T 1 1 228 1 1 1 1 1 67400          " 
ARGS_LIST[18]=" -num_iter 10 -num_repeat 10   gemm N T 100 674 228 674 674 100 1 1            " 
ARGS_LIST[19]=" -num_iter 10 -num_repeat 10   gemm N N 100 454276 228 100 228 100 1 1         " 
ARGS_LIST[20]=" -num_iter 10 -num_repeat 10   gemm N T 6740 6740 228 6740 6740 6740 1 1       " 

#ARGS_LIST[0]=" -mnk 32768 32768 32768 -num_iter 1 -num_repeat 3"
#ARGS_LIST[0]=" -mnk 16384 16384 16384 -num_iter 1 -num_repeat 3"
#ARGS_LIST[0]=" -mnk 256 256 256 -num_iter 100 -num_repeat 10"
#ARGS_LIST[0]=" -trans N N -mnk 256 256 256 -num_iter 100 -num_repeat 10 -batched -num_batches 100"
#ARGS_LIST[0]=" -num_iter 1 -num_repeat 2 -replay gemm_batch T T 674 674 674 674 674 674 1 0 228 "
#ARGS_LIST[1]=" -mnk 512 512 512 -num_iter 100 -num_repeat 10"
#ARGS_LIST[2]=" -mnk 1024 1024 1024 -num_iter 100 -num_repeat 10"
#ARGS_LIST[3]=" -mnk 2048 2048 2048 -num_iter 100 -num_repeat 5"
#ARGS_LIST[4]=" -mnk 4096 4096 4096 -num_iter 10 -num_repeat 3"
#ARGS_LIST[5]=" -mnk 8192 8192 8192 -num_iter 1 -num_repeat 3"
NUM_ARGS=${#ARGS_LIST[*]}

for EXE_ARGS in "${ARGS_LIST[@]}"; do
  echo "EXE_ARGS= ${EXE_ARGS}"
  mpiexec ${MPI_ARGS} ${OMP_ARGS} ${EXE} ${EXE_ARGS}
done

#mpiexec ${MPI_ARGS} ${OMP_ARGS} ${EXE} ${EXE_ARGS}
#nsys profile --stats=true -t cuda,nvtx mpiexec ${MPI_ARGS} ${OMP_ARGS} ${EXE} ${EXE_ARGS} 2>&1 | tee profile.txt
#ncu --print-summary per-kernel ${EXE} ${EXE_ARGS} 2>&1 | tee profile.txt
