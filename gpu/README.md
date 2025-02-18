# GPU-enabled LASSCF

The following is a short summary of documenting how to build and run LASSCF (and similar) calculations using the `mrh` code accelerated by a small `libGPU` library supporting multiple backends to better enable exploration of algorithms and strategies on different hardware.

# Building gpu-enabled LASSCF

The standalone gpu package consists of two primary components: a Python module for monkey-patching PySCF code and a C/C++ library for executing select functionality on CPUs and GPUs. 

- Monkey Patch Layer: mrh/gpu/gpu4mrh
- C/C++ Library: mrh/gpu/src
  
The organization of gpu4mrh very much follows that of [gpu4pyscf](https://github.com/pyscf/gpu4pyscf), but supports different functionality.

Several backends for the library are currently in development. The `Host` backend is primarily used today for debugging and preparing code for GPU offload using either `CUDA` or `OpenMP`. 

- Host: cpu-only backend parallelized with OpenMP
- CUDA: gpu-enabled backend using cuBLAS with some CPU/OpenMP operations
- OpenMP: gpu-enabled backend using OpenMP offload, currently with cuBLAS
- OpenMP-MKL: ToDo

The gpu package is not currently integrated in the cmake build system for mrh. At present, one needs to separately build the gpu package after successfully installing `mrh` and `PySCF` for their respective documention.

## Polaris @ ALCF

### Setting up environment

A local Python environment was created based on the `conda` environment provide by the ALCF, which required minimal additional modules to be installed. That environment was created as follows on a Polaris login node.
```
$ module load conda
$ conda activate
$ python -m venv --system-site-packages my_env

$ ./my_env/bin/activate

$ pip install nvtx
$ pip install pybind11
```
The `pybind11` module is currently the only requirement. The `nvtx` module is used for profiling with NVIDIA's tools.

The following helper script is then used for all work related to the `PySCF` and `mrh` codes.
```
$ cat setup_polaris.sh

WORKDIR=/lus/grand/projects/LASSCF_gpudev/knight/soft

module load cmake
module swap PrgEnv-nvhpc/8.3.3 PrgEnv-gnu
module load cudatoolkit-standalone

. ${WORKDIR}/my_env/bin/activate

export PYTHONPATH=${WORKDIR}/pyscf:$PYTHONPATH
export PYTHONPATH=${WORKDIR}/mrh/gpu:$PYTHONPATH
export PYTHONPATH=${WORKDIR}:$PYTHONPATH

# -- should probably replace this with job-specific tmp (and probably node-local?)
export PYSCF_TMPDIR=${WORKDIR}/../tmp
```

The environment is then loaded in current shell or start of a batch job submission as follows.
```
$ . ./setup_polaris.sh
```

### Installing PySCF

The following script is an example of that used to install PySCF on Polaris. The the build, git will attempt to clone some additional repos and this requires outbound access otherwise the build will fail. More info on the proxy settings is available [here]((https://docs.alcf.anl.gov/polaris/getting-started/#proxy)).
```
WORKDIR=/lus/grand/projects/LASSCF_gpudev/knight/soft
cd ${WORKDIR}

git clone https://github.com/pyscf/pyscf.git

# cmake requires outbound access to clone a couple repos

# proxy settings
export HTTP_PROXY="http://proxy-01.pub.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy-01.pub.alcf.anl.gov:3128"
export http_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export https_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export ftp_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"

cd ${WORKDIR}/pyscf/pyscf/lib
mkdir build
cd build

#cmake .. -DDISABLE_DFT=OFF -DBLAS_LIBRARIES="-lsci_gnu_82_mpi -lsci_gnu_82"

# need to figure out why using simple BLAS/LAPACK install and not cray_libsci...
cmake .. -DDISABLE_DFT=OFF -DBLAS_LIBRARIES="-L/home/knight/soft/polaris/lapack/lib -llapack -lrefblas"

make -j 16

```

### Installing mrh

The mrh code can similarly be installed in a straightforward manner.
```
WORKDIR=/lus/grand/projects/LASSCF_gpudev/knight/soft
cd ${WORKDIR}

git clone https://github.com/MatthewRHermes/mrh.git

cd mrh/lib
mkdir build
cd build

# same note as above, why using local install of blas & lapack??
cmake .. -DBLAS_LIBRARIES="-L/home/knight/soft/polaris/lapack/lib -llapack -lrefblas"

make -j 4
```


### Building libGPU with host backend 

Once `mrh` and `PySCF` have been installed and verified to work it is straightforward to build and install the gpu package after modifying an appropriate `architecture` file to set compiler flags. The following build uses the GNU compilers.

```
$ cd mrh/gpu/src
$ make clean
$ make ARCH=polaris-cpu-gnu install
```
The generated `libgpu.so` library will be installed into the specificed `INSTALL` directory, which currently resides in `mrh/my_pyscf/gpu/libgpu.so`. When this library is not installed, there is a STUB `mrh/my_pyscf/gpu/libgpu.py` that will be used instead. For reference, the architecture file `gpu/src/arch/polaris-cpu-gnu` is defined as follows.
```
$ cat ./arch/polaris-cpu-gnu
INSTALL = ../../my_pyscf/gpu

PYTHON_INC=$(shell python -m pybind11 --includes)

FC = ftn
FCFLAGS = -g -O3 -fopenmp
FCFLAGS += -fPIC

CXX = CC
CXXFLAGS = -g -O3 -fopenmp
CXXFLAGS += -fPIC
CXXFLAGS += $(PYTHON_INC)

CXXFLAGS += -D_USE_CPU
#CXXFLAGS += -D_SIMPLE_TIMER

LD = $(FC)
LDFLAGS = -fopenmp -fPIC -shared
LIB = -lstdc++

```

The `-D_USE_CPU` preprocessor flag is key here for enabling the Host backend and CPU-only runs. The `-D_SIMPLE_TIME` flag is enabling a very lightweight timer for select portions of the code useful only for developing the code. This timer is likely to be removed/altered as code development progresses.

When needed, the `libgpu.so` library can be uninstalled by running the `make ARCH=polaris-cpu-gnu clean` command (or similar). Input files attempting to use functions within the `libgpu` module will receive an error to help catch unintended usage.

### Building libGPU with CUDA backend

The environment setup by the script above provides the necessary CUDA software components. To build with the CUDA backend to enable runs on NVIDIA GPUs, one can simply build as follows.
```
$ cd mrh/gpu/src
$ make clean
$ make -f Makefile.nvcc ARCH=polaris-gnu-nvcc install
```
Use of the `Makefile.nvcc` Makefile is because two separate compilers are being used to compile source in this build: GNU compilers for CPU-only code and NVIDIA's nvcc compiler for GPU code. For reference, the architecture file `gpu/src/arch/polaris-gnu-nvcc` is defined as follows.
```
INSTALL = ../../my_pyscf/gpu

PYTHON_INC=$(shell python -m pybind11 --includes)

FC = ftn
FCFLAGS = -g -fopenmp -O3

CXX = CC
CXXFLAGS = -g -fPIC -fopenmp -O3
CXXFLAGS += -I$(NVIDIA_PATH)/cuda/include
CXXFLAGS += $(PYTHON_INC)

CXXFLAGS += -D_USE_GPU -D_GPU_CUDA
CXXFLAGS += -I/soft/compilers/cudatoolkit/cuda-11.4.4/include
CXXFLAGS += -D_SIMPLE_TIMER
CXXFLAGS += -D_CUDA_NVTX

CUDA_CXXFLAGS = $(PYTHON_INC)
CUDA_CXXFLAGS += -Xcompiler -fopenmp
CUDA_CXXFLAGS += -shared -Xcompiler -fPIC 
CUDA_CXXFLAGS += -D_USE_GPU -D_GPU_CUDA
CUDA_CXXFLAGS += -I/soft/compilers/cudatoolkit/cuda-11.4.4/include
CUDA_CXXFLAGS += -D_SIMPLE_TIMER
CUDA_CXXFLAGS += -D_CUDA_NVTX

LD = $(CXX)
LDFLAGS = -fPIC -shared
LIB = -lstdc++
LIB += -L/soft/compilers/cudatoolkit/cuda-11.4.4/lib64 -lcublas
LIB += -lnvToolsExt
```

The `-D_GPU_CUDA` flag is key to enabling support for the CUDA backend and running on NVIDIA GPUs. The `-D_CUDA_NVTX` flag is useful for profiling and enables using the nvtx API to start/stop profiling sections around certain code blocks. 

Work is in progress to build using NVIDIA's compilers for everything and being able to simply execute `make ARCH=polaris install`. 

### Building libGPU with OpenMPTarget backend

This is likely to be the main production backend for GPU-accelerated runs, but is very much work-in-progress. This documentation will be updated when simple cases of offloading a matrix-multiply in the get_jk() function using NVIDIA's cuBLAS or Intel's MKL have been validated. 

## Running GPU-enabled LASSCF calculations

The primary input-deck for development work thus far has been the `polymer_async` benchmark in the following directory.
```
mrh/examples/gpu_lasscf/polymer_async
```

Usage will evolve as the code develops, but the following is the `1_6-31g_inp_gpu_simple.py` input script for reference.
```
$ cat ./1_6-31g_inp_gpu_simple.py
import pyscf 
from gpu4mrh import patch_pyscf

from geometry_generator import generator
from pyscf import gto, scf, tools, mcscf, lib
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
from pyscf.mcscf import avas	

from mrh.my_pyscf.gpu import libgpu
gpu = libgpu.libgpu_init()

lib.logger.TIMER_LEVEL=lib.logger.INFO

nfrags=1
basis='6-31g'
outputfile='1_6-31g_out_gpu.log'
mol=gto.M(use_gpu=gpu, atom=generator(nfrags),basis=basis,verbose=5,output=outputfile)
mf=scf.RHF(mol)
mf=mf.density_fit()
mf.run()

ncas,nelecas,guess_mo_coeff = avas.kernel(mf, ['C 2p'])
#ncas,nelecas,guess_mo_coeff = avas.kernel(mf, ['C 2pz'])

las=LASSCF(mf, list((2,)*nfrags),list((2,)*nfrags), use_gpu=gpu)

frag_atom_list=[list(range(1+4*nfrag,3+4*nfrag)) for nfrag in range(nfrags)]
#print(frag_atom_list)
mo_coeff=las.set_fragments_(frag_atom_list, guess_mo_coeff)
las.kernel(mo_coeff)

libgpu.libgpu_destroy_device(gpu)
```

Key modifications to a "normal" LASSCF input file are as follows.
- `from gpu4mrh import patch_pyscf` : enable monkey patching for a select number of PySCF source files, such as updating the Molecule object to track a new `use_gpu` variable.
- `from mrh.my_pyscf.gpu import libgpu` : enable access to the gpu library interface functions 
- `gpu = libgpu.libgpu_init()` : initialize the gpu library and return a handle. This gpu handle is to be passed to a small number of functions (and likely made smaller in the future).
- `mol=gto.M(use_gpu=gpu, atom=...` : this is the key usage of the gpu handle by which most of the underlying code and algorithms can access the gpu library.
- `las=LASSCF(mf, list((2,)*nfrags),list((2,)*nfrags), use_gpu=gpu)` : this is currently required, but expected to be no longer necessary soon...
- `libgpu.libgpu_destroy_device(gpu)` : always good to clean up after ourselves and prevent out-of-memory issues in more complex workflows.

## Status

The `Host` and `CUDA` backends tested on Polaris currently yield consistent LASSCF energies and number of iterations for the asynchronous algorithm for the polymer\_async input-deck for nfrags= 1, 2, and 4. There is currently a 0.0253 energy difference for nfrags = 8 that is being debugged. Any differences observed should be reported as a bug.

*Last Updated : 9-08-2023*
