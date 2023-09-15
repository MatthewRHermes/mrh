#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug 
#PBS -A Catalyst
#PBS -l filesystems=home:grand:eagle

cd /lus/grand/projects/LASSCF_gpudev/knight/soft/mrh/examples/gpu/polymer_async

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

#INPUT="1_6-31g_inp.py"
INPUT="1_6-31g_inp_gpu.py"
#INPUT="1_6-31g_inp_scf_gpu.py"

export CUDA_VISIBLE_DEVICES=0

#EXE=/home/knight/repos/GettingStarted/Examples/Polaris/affinity_omp/hello_affinity
EXE="python ${INPUT} "
#EXE="python my_profile.py ${INPUT} "

#mpiexec ${MPI_ARGS} ${OMP_ARGS} /home/knight/repos/GettingStarted/Examples/Polaris/affinity_omp/hello_affinity 

#python -m cProfile -o out.prof ${INPUT}
#{ time ${EXE} ;} 2>&1 | tee profile.txt
{ time mpiexec ${MPI_ARGS} ${OMP_ARGS} ${EXE} ;} 2>&1 | tee profile.txt
#nsys profile --stats=true -t cuda,nvtx mpiexec ${MPI_ARGS} ${OMP_ARGS} ${EXE} 2>&1 | tee profile.txt
