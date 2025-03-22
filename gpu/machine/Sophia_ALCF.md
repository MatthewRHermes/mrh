# Sophia @ ALCF

The following is a short summary documenting how to build and run GPU-accelerated LASSCF (and similar) calculations using the `mrh` code on [Sophia](https://www.alcf.anl.gov/sophia) at the Argonne Leadership Computing Facility.

## Creating Python environment

A local virtual Python environment is derived from the `conda` environment provided by ALCF, which will require only a few additional modules to be installed. The Python environment can be created on a Sophia login node as follows.

``` bash
WORKDIR=/path/to/installation
cd $WORKDIR

module restore
module load conda

conda activate
python -m venv --system-site-packages my_env

. ./my_env/bin/activate

pip install pybind11
pip install nvtx
```

The `pybind11` module is required. The `nvtx` module is used to aid profiling with NVIDIA's tools.

## Setting up software environment

With the virtual Python environment ready to go, the following helper script can be used to quickly initialize the software environment for all work related to the `PySCF` and `mrh` codes.
``` bash
$ cd /path/to/installation
$ cat setup_env.sh

WORKDIR=/path/to/installation

module load cmake
module swap PrgEnv-nvhpc/8.3.3 PrgEnv-gnu
module load cudatoolkit-standalone

. ${WORKDIR}/my_env/bin/activate

export PYTHONPATH=${WORKDIR}/pyscf:$PYTHONPATH
export PYTHONPATH=${WORKDIR}/mrh/gpu:$PYTHONPATH
export PYTHONPATH=${WORKDIR}:$PYTHONPATH

# Ensure use of standalone OpenBlas library
export LD_LIBRARY_PATH=${WORKDIR}/openblas/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=${WORKDIR}/openblas/lib/libopenblas.so
```

The environment is then loaded in current shell or start of a batch job submission as follows.
```
$ . /path/to/installation/setup_env.sh
```

### Installing OpenBLAS

The OpenBLAS library can be installed for BLAS and LAPACK functionality on the CPU. 

```bash
WORKDIR=/path/to/installation
cd ${WORKDIR}

git clone https://github.com/OpenMathLib/OpenBLAS.git

cd OpenBLAS
make CC=mpicc FC=mpif90 TARGET=ZEN USE_OPENMP=1
mkdir install
make CC=mpicc FC=mpif90 TARGET=ZEN USE_OPENMP=1 PREFIX=${WORKDIR}/OpenBLAS/install install 
```

### Installing PySCF

The following script is an example to install PySCF on Sophia. Building from source is not required, but it can help with resolving some software issues.

``` bash
WORKDIR=/path/to/installation
cd ${WORKDIR}

. ./setup_env.sh

git clone https://github.com/pyscf/pyscf.git

cd ./pyscf/pyscf/lib
mkdir build
cd build

cmake .. -DDISABLE_DFT=OFF -DBLAS_LIBRARIES="${WORKDIR}/OpenBLAS/install/lib/libopenblas.so" -DBUILD_MARCH_NATIVE=ON

make -j 4

```

The build can be completed faster a Sophia compute node in an interactive job. During the build, git will attempt to clone some additional repos and this requires outbound access otherwise the build will fail. More info on the proxy settings is available [here]((https://docs.alcf.anl.gov/sophia/getting-started/#proxy)).

### Installing mrh

The mrh code can similarly be installed in a straightforward manner from source.

```bash
WORKDIR=/path/to/installation
cd ${WORKDIR}

git clone https://github.com/MatthewRHermes/mrh.git

cd mrh/lib
mkdir build
cd build

CXX=mpicxx CC=mpicc FC=mpif90 cmake .. -DBLAS_LIBRARIES="${WORKDIR}/OpenBLAS/install/lib/libopenblas.so" 

make -j 4
```

### Building gpu4mrh with CUDA backend 

Once `mrh` and `PySCF` have been installed and verified to work, it is straightforward to build and install the `gpu4mrh` package. The following build on Sophia uses GNU compilers for CPU and `nvcc` for GPU code.

```bash
cd mrh/gpu/src
make clean
make ARCH=sophia install
```
The generated `libgpu.so` library will be copied to `mrh/my_pyscf/gpu/libgpu.so`. When this library is not installed, there is a STUB `mrh/my_pyscf/gpu/libgpu.py` used instead that prints a helpful error message. For reference, the architecture file `gpu/src/arch/sophia` can be updated as needed for similar architectures.

### Submission script for batch job

The following is an example submission script for a PBS batch job that takes the PySCF input script as a single command-line argument. By default, all GPUs requested in batch job will be used.


``` bash
$ cat ./submit_sophia.sh
#!/bin/bash -l
#PBS -l select=1:system=sophia
#PBS -l walltime=0:30:00
#PBS -q by-gpu
#PBS -A LASSCF_gpudev
#PBS -l filesystems=home:grand

INPUT="${1}"

WORKDIR=/path/to/installation

cd /path/to/test

. ${WORKDIR}/setup_env.sh

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=1
NTHREADS=16
NDEPTH=${NTHREADS}

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

MPI_ARGS="-n ${NTOTRANKS} --npernode ${NRANKS_PER_NODE} "
MPI_ARGS+=" -x OMP_NUM_THREADS=${NTHREADS} -x OMP_PROC_BIND=spread -x OMP_PLACES=cores "

export PYSCF_TMPDIR=/tmp
export PYSCF_MAX_MEMORY=160000

EXE="python ${INPUT} "

{ time mpiexec ${MPI_ARGS} ${EXE} ;} 2>&1 | tee screen.txt
```