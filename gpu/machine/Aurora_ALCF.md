# Aurora @ ALCF

The following is a short summary documenting how to build and run GPU-accelerated LASSCF (and similar) calculations using the `mrh` code on [Aurora](https://www.alcf.anl.gov/aurora) at the Argonne Leadership Computing Facility.

## Creating Python environment

A local virtual Python environment is derived from a Python environment provided by ALCF, which will require only a few additional modules to be installed. The Python environment can be created on a Aurora login node as follows.

``` bash linenums="1"
WORKDIR=/path/to/installation
cd $WORKDIR

module load thapi

python -m venv --system-site-packages ${PWD}/my_env

. ./my_env/bin/activate

pip install -i https://pypi.anaconda.org/intel/simple scipy
pip install pybind11
pip install h5py

pip uninstall impi-rt
```

The `scipy`, `h5py`, and `pybind11` modules are required, and `thapi` is a lightweight profiling tool, which we are expanding upon the Python environment of. Install scipy like this is needed for modest performance gain on CPU side (~2x in gemms) by leveraging Intel MKL libraries instead of OpenBLAS. Uninstalling the Intel MPI runtime module is needed to avoid conflicts with the MPI installed on Aurora (e.g. `mpiexec` for launching tasks).

## Setting up software environment

With the virtual Python environment ready to go, the following helper script can be used to quickly initialize the software environment for all work related to the `PySCF` and `mrh` codes.

``` bash linenums="1" title="setup_env.sh"
module restore

BASE=/path/to/installation

module load thapi

. ${WORKDIR}/my_env/bin/activate

module load cmake hdf5

export PYTHONPATH=${BASE}/pyscf:$PYTHONPATH
export PYTHONPATH=${BASE}/mrh/gpu:$PYTHONPATH
export PYTHONPATH=${BASE}:$PYTHONPATH

export PYSCF_EXT_PATH=${BASE}/pyscf-forge
```

The environment is then loaded in current shell or start of a batch job submission as follows.
```
$ . /path/to/installation/setup_env.sh
```

### Installing PySCF

The following script is an example to install PySCF on Aurora. Building from source is not required, but it can help with resolving some software issues.

``` bash linenums="1"
WORKDIR=/path/to/installation
cd ${WORKDIR}

. ./setup_env.sh

git clone https://github.com/pyscf/pyscf.git

cd ./pyscf/pyscf/lib
mkdir build
cd build

cmake .. -DDISABLE_DFT=OFF -DBUILD_MARCH_NATIVE=ON

make -j 4
```

The build can be completed faster an Aurora compute node in an interactive job. During the build, git will attempt to clone some additional repos and this requires outbound access otherwise the build will fail. More info on the proxy settings is available [here]((https://docs.alcf.anl.gov/polaris/getting-started/#proxy)).

If compilation appears to be stalled, then it may be related to an issue that hasn't been fully debugged (something to do with `bzip2`). A workaround is to disable building of tests for LibXC by adding the `-DBUILD_TESTING=OFF` option in the following section of `pyscf/pyscf/lib/CMakeLists.txt`.

```cmake linenums="1"
if(ENABLE_LIBXC AND BUILD_LIBXC)
  ExternalProject_Add(libxc
    #GIT_REPOSITORY https://gitlab.com/libxc/libxc.git
    #GIT_TAG master
    URL https://gitlab.com/libxc/libxc/-/archive/7.0.0/libxc-7.0.0.tar.gz
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_SHARED_LIBS=1
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
            -DCMAKE_INSTALL_LIBDIR:PATH=lib
            -DENABLE_FORTRAN=0 -DDISABLE_KXC=0 -DDISABLE_LXC=1
            -DCMAKE_C_CREATE_SHARED_LIBRARY=${C_LINK_TEMPLATE}
            -DENABLE_XHOST:STRING=${BUILD_MARCH_NATIVE}
            -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5 # remove when libxc update version min in next release
            -DBUILD_TESTING=OFF
  )   
  add_dependencies(xc_itrf libxc)
  add_dependencies(dft libxc)
  add_dependencies(pdft libxc)
endif() # ENABLE_LIBXC

```

### Installing Pyscf-Forge

The pyscf-forge code can be installed following same recipe as PySCF.

``` bash linenums="1"
WORKDIR=/path/to/installation
cd ${WORKDIR}

. ./setup_env.sh

git clone https://github.com/pyscf/pyscf-forge.git

cd ./pyscf-forge/pyscf/lib
mkdir build
cd build

cmake .. -DDISABLE_DFT=OFF -DBUILD_MARCH_NATIVE=ON

make -j 4
```



### Installing mrh

The mrh code can similarly be installed in a straightforward manner from source.

```bash linenums="1"
WORKDIR=/path/to/installation
cd ${WORKDIR}

git clone https://github.com/MatthewRHermes/mrh.git

cd mrh/lib
mkdir build
cd build

cmake .. 

make -j 4
```

### Building gpu4mrh with SYCL backend

Once `mrh` and `PySCF` have been installed and verified to work, it is straightforward to build and install the `gpu4mrh` package. The following build on Aurora uses oneAPI compilers for CPU and GPU code.

```bash linenums="1"
cd mrh/gpu/src
make clean
make ARCH=aurora install
```

The generated `libgpu.so` library will be copied to `mrh/my_pyscf/gpu/libgpu.so`. When this library is not installed, there is a STUB `mrh/my_pyscf/gpu/libgpu.py` used instead that prints a helpful error message. For reference, the architecture file `gpu/src/arch/aurora` can be updated as needed for similar architectures.

### Submission script for batch job

The following is an example submission script for a PBS batch job that takes the PySCF input script as a single command-line argument. By default, all 6x GPUs (12 tiles) on the Aurora compute node will be used. It's important to ensure only a single compute node is requested (i.e. MPI-support for multi-node jobs is work-in-progress).

``` bash linenums="1" title="submit_aurora.sh"
#!/bin/bash -l
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug
#PBS -A LASSCF_gpudev
#PBS -l filesystems=home:flare

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

MPI_ARGS="-n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} "
MPI_ARGS+=" --depth=${NDEPTH} --cpu-bind=depth "
MPI_ARGS+=" --env OMP_NUM_THREADS=${NTHREADS} --env OMP_PROC_BIND=spread --env OMP_PLACES=cores "

# Launch kernels to all 12 tiles
export ONEAPI_DEVICE_SELECTOR=level_zero:*.*

export PYSCF_TMPDIR=/tmp
export PYSCF_MAX_MEMORY=160000

EXE="python ${INPUT} "

{ time mpiexec ${MPI_ARGS} ${EXE} ;} 2>&1 | tee screen.txt
```