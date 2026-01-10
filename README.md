# mrh
GPL research code of Matthew R. Hermes

## WARNING!
The CSF-basis FCI solver and MC-PDFT and MC-DCFT modules (other than a few experimental features) have been *moved to [pyscf-forge](https://github.com/pyscf/pyscf-forge)* and are currently in the process of being removed from this project.

## Dependencies:
- PySCF, including all header files in pyscf/lib
- [pyscf-forge](https://github.com/pyscf/pyscf-forge)
- see `pyscf_version.txt` and `pyscf-forge_version.txt` for last-tested versions or commits

### Installation 

To use mrh, install the `master` branch through pip by running:

```bash
# Install PySCF and PySCF-Forge separatly (see note)

# Install the compatible PySCF version
pip install `curl -s https://raw.githubusercontent.com/GagliardiGroup/mrh/refs/heads/master/pyscf_version.txt | xargs`

# Install the compatible PySCF-Forge version
pip install `curl -s https://raw.githubusercontent.com/GagliardiGroup/mrh/refs/heads/master/pyscf-forge_version.txt | xargs`

# Install MRH
pip install git+https://github.com/MatthewRHermes/mrh.git
```

*NOTE:* Both mrh and pyscf-forge require PySCF as a build and runtime dependency. The pyscf-forge build script has a bug where the `pyscf/lib` directory is not corrected detected (it references a deleted temporary directory). This does not occur if PySCF is already installed.

### Developer Installation

Clone this repository and then tell pip to do an editable installation from the local directory.

```bash
git clone git@github.com:MatthewRHermes/mrh.git
pip install -e ./mrh
```

## Installing on macOS

OpenMP is required to build mrh. On macOS, CMake [cannot detect](https://stackoverflow.com/questions/46414660/macos-cmake-and-openmp) OpenMP when using the default compiler. The easiest work-around is to install GCC and OpenMP through [Homebrew](https://brew.sh/) and tell PySCF/pyscf-forge/mrh to [build with these compilers](https://pyscf.org/install.html#build-from-source-with-pip).

PySCF-Forge does not build correctly with the version of LibXC which is built or bundled with PySCF on macOS. It is easier to use a Homebrew-provided version.

See [brew.sh](https://brew.sh/) for Homebrew installation instructions and then install OpenMP and LibXC by running `brew install libomp libxc`.

Once installed, tell PySCF and related packages to use GCC 14 instead of macOS's Clang-based compiler by setting the `CMAKE_CONFIGURE_ARGS` environment variable.

```bash
# Set the compiler
export CMAKE_CONFIGURE_ARGS="-DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 -DBUILD_LIBXC=OFF"

# Set these to use the local libxc
export LIBRARY_PATH="/opt/homebrew/lib"
export C_INCLUDE_PATH="/opt/homebrew/include"
export CPLUS_INCLUDE_PATH="/opt/homebrew/lib"

# Enable a multi-threaded build (replace 12 with how many CPUs you have)
export CMAKE_BUILD_ARGS="-- -j12"
```

### Legacy Install Method
- cd /path/to/mrh/lib
- mkdir build ; cd build
- cmake ..
- make
- Add /path/to/mrh to Python's search path somehow. Examples:
    * "sys.path.append ('/path/to')" in Python script files
    * "export PYTHONPATH=/path/to:$PYTHONPATH" in the shell
    * "ln -s /path/to/mrh /path/to/python/site-packages/mrh" in the shell
- If you installed PySCF from source and the compilation still fails, try setting the path to the PySCF library directory manually:
`cmake -DPYSCFLIB=/full/path/to/pyscf/lib ..`

## Notes
- The dev branch is continuously updated. The master branch is updated every time I pull PySCF and confirm that everything still works. If you have some issue and you think it may be related to PySCF version mismatch, try using the master branch and the precise PySCF commit indicated above.
- If you are using Intel MKL as the BLAS library, you may need to enable the corresponding cmake option:
`cmake -DBLA_VENDOR=Intel10_64lp_seq ..`

## Acknowledgments
- This work is supported by the U.S. Department of Energy, Office of Basic Energy Sciences, Division of Chemical Sciences, Geosciences and Biosciences through the Nanoporous Materials Genome Center under award DE-FG02-17ER16362.

