# Midway3 @ ALCF

The following is a short summary documenting how to build and run GPU-accelerated LASSCF (and similar) calculations using the `mrh` code on [Midway3](https://rcc.uchicago.edu/midway3) at the University of Chicago Research Computing Center.

## Creating Python environment

A local virtual Python environment is derived from the `conda` environment provided by RCC, which will require only a few additional modules to be installed. The Python environment can be created on a Polaris login node as follows.

``` bash
WORKDIR=/path/to/installation
cd $WORKDIR

module restore
module load python/anaconda-2021.05

conda activate
python -m venv --system-site-packages my_env

. ./my_env/bin/activate

pip install nvtx
```

The required `pybind11` module is already available. The `nvtx` module is used to aid profiling with NVIDIA's tools.

## Setting up software environment

With the virtual Python environment ready to go, the following helper script can be used to quickly initialize the software environment for all work related to the `PySCF` and `mrh` codes.

``` bash
$ cd /path/to/installation
$ cat setup_env.sh

WORKDIR=/path/to/installation

module load python/anaconda-2021.05
module load cmake mpich gcc cuda

. ${WORKDIR}/my_env/bin/activate

export PYTHONPATH=${WORKDIR}/pyscf:$PYTHONPATH
export PYTHONPATH=${WORKDIR}/mrh/gpu:$PYTHONPATH
export PYTHONPATH=${WORKDIR}:$PYTHONPATH

# Ensure use of MKL library 
MKL=/software/python-anaconda-2021.05-el8-x86_64/lib
export LD_PRELOAD=${MKL}/libmkl_def.so.1:${MKL}/libmkl_avx2.so.1:${MKL}/libmkl_core.so:${MKL}/libmkl_intel_lp64.so:${MKL}/libmkl_intel_thread.so:${MKL}/libiomp5.so
```

The environment is then loaded in current shell or start of a batch job submission as follows.
```
$ . /path/to/installation/setup_env.sh
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

CC=mpicc CXX=mpicxx cmake .. -DDISABLE_DFT=OFF
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

CC=mpicc CXX=mpicxx cmake .. 

make -j 4
```

### Building gpu4mrh with CUDA backend 

Once `mrh` and `PySCF` have been installed and verified to work, it is straightforward to build and install the `gpu4mrh` package. The following build on Midway3 uses GNU compilers for CPU and `nvcc` for GPU code.

```bash
cd mrh/gpu/src
make clean
make ARCH=midway3 install
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