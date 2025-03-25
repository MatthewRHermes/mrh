# MI250 @ JLSE

The following is a short summary documenting how to build and run GPU-accelerated LASSCF (and similar) calculations using the `mrh` code on the [AMD MI250](https://www.jlse.anl.gov/testbeds-list) nodes in the Argonne Joint Laboratory for System Evaluation (JLSE).

## Creating Python environment

A local virtual Python environment is derived from a Python environment provided by JLSE, which will require only a few additional modules to be installed. The Python environment can be created on an MI250 compute node as follows.

``` bash
WORKDIR=/path/to/installation
cd $WORKDIR

module load spack python

python -m venv --system-site-packages my_env

. ./my_env/bin/activate

pip install numpy
pip install scipy
pip install h5py
pip install pybind11
```

The `numpy`, `scipy`, `h5py`, and `pybind11` modules are required.

## Setting up software environment

With the virtual Python environment ready to go, the following helper script can be used to quickly initialize the software environment for all work related to the `PySCF` and `mrh` codes.

``` bash
$ cd /path/to/installation
$ cat setup_env.sh

WORKDIR=/path/to/installation

module load rocm
module load spack aomp

. ${WORKDIR}/my_env/bin/activate

# so mrh cmake picks up correct python
export PATH=${WORKDIR}/my_env/bin:$PATH

export PYTHONPATH=${WORKDIR}/pyscf:$PYTHONPATH
export PYTHONPATH=${WORKDIR}/mrh/gpu:$PYTHONPATH
export PYTHONPATH=${WORKDIR}:$PYTHONPATH

# Ensure use of standalone OpenBlas library
export LD_LIBRARY_PATH=${BASE}/openblas/lib:$LD_LIBRARY_PATH
#export LD_PRELOAD=${BASE}/openblas/lib/libopenblas.so
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
make CC=hipcc FC=flang TARGET=ZEN USE_OPENMP=1
mkdir install
make CC=hipcc FC=flang TARGET=ZEN USE_OPENMP=1 PREFIX=${WORKDIR}/OpenBLAS/install install 
```

### Installing PySCF

The following script is an example to install PySCF. Building from source is not required, but it can help with resolving some software issues.

``` bash
WORKDIR=/path/to/installation
cd ${WORKDIR}

. ./setup_env.sh

git clone https://github.com/pyscf/pyscf.git

cd ./pyscf/pyscf/lib
mkdir build
cd build

CXX=hipcc CC=hipcc FC=flang cmake .. -DDISABLE_DFT=OFF -DBLAS_LIBRARIES="${WORKDIR}/openblas/lib/libopenblas.so" -DBUILD_MARCH_NATIVE=ON

make -j 4
```

### Installing mrh

The mrh code can similarly be installed in a straightforward manner from source.

```bash
WORKDIR=/path/to/installation
cd ${WORKDIR}

git clone https://github.com/MatthewRHermes/mrh.git

cd mrh/lib
mkdir build
cd build

CXX=hipcc CC=hipcc FC=flang cmake .. -DBLAS_LIBRARIES="${WORKDIR}/openblas/lib/libopenblas.so " -DPYSCFLIB=${WORKDIR}/pyscf/pyscf/lib

make -j 4
```

### Building gpu4mrh with HIP backend

Once `mrh` and `PySCF` have been installed and verified to work, it is straightforward to build and install the `gpu4mrh` package. The following build on a MI250 node uses ROCM compilers for CPU and GPU code.

```bash
cd mrh/gpu/src
make clean
make ARCH=jlse-amd-mi250 install
```

The generated `libgpu.so` library will be copied to `mrh/my_pyscf/gpu/libgpu.so`. When this library is not installed, there is a STUB `mrh/my_pyscf/gpu/libgpu.py` used instead that prints a helpful error message. For reference, the architecture file `gpu/src/arch/jlse-amd-mi250` can be updated as needed for similar architectures.

### Submission script for batch job

The following is an example script to run interactive jobs on the compute node that takes the PySCF input script as a single command-line argument. By default, all 4x GPUs (8 GCDs) on the AMD MI250 compute node will be used. It's important to ensure only a single compute node is requested (i.e. MPI-support for multi-node jobs is work-in-progress).

*Note* This is for demonstration purposes as some CPU performance issues are still being resolved.

``` bash
$ cat ./run.sh
#!/bin/bash -l

INPUT="${1}"

WORKDIR=/path/to/installation

cd /path/to/test

. ${WORKDIR}/setup_env.sh

export OMP_NUM_THREADS=32
export OMP_PROC_BIND=close
export OMP_PLACES=cores

export LD_PRELOAD=/usr/lib64/libstdc++.so.6:/soft/compilers/rocm/rocm-6.3.2/lib/llvm/lib/libomp.so:${WORKDIR}/openblas/lib/libopenblas.so.0

export PYSCF_TMPDIR=/tmp
export PYSCF_MAX_MEMORY=160000

EXE="python ${INPUT} "

{ time ${EXE} ;} 2>&1 | tee screen.txt
```