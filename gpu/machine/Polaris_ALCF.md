# Polaris @ ALCF

The following is a short summary documenting how to build and run GPU-accelerated LASSCF (and similar) calculations using the `mrh` code on [Polaris](https://www.alcf.anl.gov/polaris) at the Argonne Leadership Computing Facility.

## Creating Python environment

A local virtual Python environment is derived from the `conda` environment provided by ALCF, which will require only a few additional modules to be installed. The Python environment can be created on a Polaris login node as follows.

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

module load conda

. ${WORKDIR}/my_env/bin/activate

export PYTHONPATH=${WORKDIR}/pyscf:$PYTHONPATH
export PYTHONPATH=${WORKDIR}/mrh/gpu:$PYTHONPATH
export PYTHONPATH=${WORKDIR}:$PYTHONPATH

# Ensure use of standalone OpenBlas library 
export LD_LIBRARY_PATH=${WORKDIR}/openblas/lib:$LD_LIBRARY_PATH
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
make CC=cc FC=ftn TARGET=ZEN USE_OPENMP=1
mkdir install
make CC=cc FC=ftn TARGET=ZEN USE_OPENMP=1 PREFIX=${WORKDIR}/OpenBLAS/install install 
```

### Installing PySCF

The following script is an example to install PySCF on Polaris. Building from source is not required, but it can help with resolving some software issues.

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

The build can be completed faster a Polaris compute node in an interactive job. During the build, git will attempt to clone some additional repos and this requires outbound access otherwise the build will fail. More info on the proxy settings is available [here]((https://docs.alcf.anl.gov/polaris/getting-started/#proxy)).

### Installing mrh

The mrh code can similarly be installed in a straightforward manner from source.

```bash
WORKDIR=/path/to/installation
cd ${WORKDIR}

git clone https://github.com/MatthewRHermes/mrh.git

cd mrh/lib
mkdir build
cd build

cmake .. -DBLAS_LIBRARIES="${WORKDIR}/OpenBLAS/install/lib/libopenblas.so" 

make -j 4
```

### Building gpu4mrh with CUDA backend 

Once `mrh` and `PySCF` have been installed and verified to work, it is straightforward to build and install the `gpu4mrh` package. The following build on Polaris uses GNU compilers for CPU and `nvcc` for GPU code.

```bash
cd mrh/gpu/src
make clean
make ARCH=polaris install
```
The generated `libgpu.so` library will be copied to `mrh/my_pyscf/gpu/libgpu.so`. When this library is not installed, there is a STUB `mrh/my_pyscf/gpu/libgpu.py` used instead that prints a helpful error message. For reference, the architecture file `gpu/src/arch/polaris` can be updated as needed for similar architectures.

### Submission script for batch job

The following is an example submission script for a PBS batch job that takes the PySCF input script as a single command-line argument. By default, all 4x GPUs on the Polaris compute node will be used. It's important to ensure only a single compute node is requested (i.e. MPI-support for multi-node jobs is work-in-progress).

``` bash
$ cat ./submit_polaris.sh
#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug
#PBS -A LASSCF_gpudev
#PBS -l filesystems=home:grand

INPUT="${1}"

WORKDIR=/path/to/installation

cd /path/to/test

. ${WORKDIR}/setup_env.sh

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=1
NTHREADS=32
NDEPTH=${NTHREADS}

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

MPI_ARGS="-n ${NTOTRANKS} --npernode ${NRANKS_PER_NODE} "
MPI_ARGS+=" --env OMP_NUM_THREADS=${NTHREADS} --env OMP_PROC_BIND=spread --env OMP_PLACES=cores "

export PYSCF_TMPDIR=/tmp
export PYSCF_MAX_MEMORY=160000

EXE="python ${INPUT} "

{ time mpiexec ${MPI_ARGS} ${EXE} ;} 2>&1 | tee screen.txt
```