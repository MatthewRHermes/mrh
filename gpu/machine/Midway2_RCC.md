# Midway2 @ ALCF

The following is a short summary documenting how to build and run GPU-accelerated LASSCF (and similar) calculations using the `mrh` code on [Midway2](https://rcc.uchicago.edu/midway2) at the University of Chicago Research Computing Center.

## Creating Python environment

A local virtual Python environment is derived from the `conda` environment provided by RCC, which will require only a few additional modules to be installed. The Python environment can be created on a Polaris login node as follows.

``` bash
WORKDIR=/path/to/installation
cd $WORKDIR

module load python/anaconda-2021.05

python -m venv --system-site-packages my_env

. ./my_env/bin/activate

pip install nvtx
```

The required `pybind11` module is already available. The `nvtx` module is used to aid profiling with NVIDIA's tools.

## Setting up software environment

With the virtual Python environment ready to go, the following helper script can be used to quickly initialize the software environment for all work related to the `PySCF` and `mrh` codes. The `gcc/9.1.0` compiler is needed to resolve an OpenMP compilation issue seen with some of the MP2 files in PySCF. Unfortunately, we'll need to switch back to `gcc/8.2.0` when compiling the `gpu4mrh`.

``` bash
$ cd /path/to/installation
$ cat setup_env.sh

WORKDIR=/path/to/installation

module load python/anaconda-2021.05
module load cmake
module load mpich/3.3-cuda-aware
module switch gcc gcc/9.1.0
module load mkl

. ${WORKDIR}/my_env/bin/activate

export PYTHONPATH=${WORKDIR}/pyscf:$PYTHONPATH
export PYTHONPATH=${WORKDIR}/mrh/gpu:$PYTHONPATH
export PYTHONPATH=${WORKDIR}:$PYTHONPATH
```

The environment is then loaded in current shell or start of a batch job submission as follows.
```
$ . /path/to/installation/setup_env.sh
```
### CMake

In case you run into difficulty using one of the versions of `cmake` installed, it is straightforward to download and install a binary distribution.

``` bash
$ cd /path/to/installation

$ wget https://github.com/Kitware/CMake/releases/download/v3.11.0/cmake-3.11.0-Linux-x86_64.tar.gz
$ tar -xvf cmake-311.0-linux-x86_64.tar.gz

$ export PATH=/path/to/installation/cmake-3.11.0-linux-x86_64/bin:$PATH
```

### Installing PySCF

The following script is an example to install PySCF on the login node of Midway3. Building from source is not required, but it can help with resolving some software issues.

``` bash
WORKDIR=/path/to/installation
cd ${WORKDIR}

. ./setup_env.sh

git clone https://github.com/pyscf/pyscf.git

cd ./pyscf/pyscf/lib
mkdir build
cd build

CC=mpicc CXX=mpicxx cmake .. -DDISABLE_DFT=OFF -DBLAS_LIBRARIES="-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group  -liomp5 -lpthread -lm -ldl"
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

CC=mpicc CXX=mpicxx cmake .. -DDISABLE_DFT=OFF -DBLAS_LIBRARIES="-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group  -liomp5 -lpthread -lm -ldl" -DPYSCFLIB=/project/lgagliardi/knight/LASSCF_gpudev/soft-midway2/pyscf/pyscf/lib

make -j 4
```

### Building gpu4mrh with CUDA backend 

Once `mrh` and `PySCF` have been installed and verified to work, it is straightforward to build and install the `gpu4mrh` package. The following build on Midway2 uses GNU compilers for CPU and `nvcc` for GPU code. Switching back to the `gcc/8.2.0` compiler is required.

```bash
cd mrh/gpu/src
module switch gcc gcc/8.2.0
make clean
make ARCH=midway2 install
```
The generated `libgpu.so` library will be copied to `mrh/my_pyscf/gpu/libgpu.so`. When this library is not installed, there is a STUB `mrh/my_pyscf/gpu/libgpu.py` used instead that prints a helpful error message. For reference, the architecture file `gpu/src/arch/midway3` can be updated as needed for similar architectures.

### Submission script for batch job

The following is an example submission script for a PBS batch job that takes the PySCF input script as a single command-line argument. It's important to ensure only a single compute node is requested (i.e. MPI-support for multi-node jobs is work-in-progress).

``` bash
#!/bin/bash -l
#SBATCH --account=pi-lgagliardi
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 #number of gpus
#SBATCH --ntasks-per-node=32 
#SBATCH --error=error.%j.err
. project/lgagliardi/valayagarawal/scripts/setup_env.sh

# -- should probably replace this with job-specific tmp
export PYSCF_TMPDIR=~/tmp

ulimit -s unlimited
export PYSCF_MAX_MEMORY=64000
/usr/bin/time --verbose python $1

$ cat setup_env.sh

module load cmake
module load mpich/3.3-cuda-aware
module load gcc/8.2.0
. /home/valayagarawal/soft/my_env/bin/activate

export PYTHONPATH=/home/cjknight/LASSCF_gpudev/soft/pyscf:$PYTHONPATH
export PYTHONPATH=/home/cjknight/LASSCF_gpudev/soft/mrh/gpu:$PYTHONPATH
export PYTHONPATH=/home/cjknight/LASSCF_gpudev/soft:$PYTHONPATH
```