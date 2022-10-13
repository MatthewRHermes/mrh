# mrh
GPL research code of Matthew R. Hermes

### DEPENDENCIES:

- PySCF, including all header files in pyscf/lib (most recently checked commit: 7b91b3ce5, v2.1.1)
    * This usually requires a PySCF installation which was downloaded as source (i.e., from github.com/pyscf/pyscf) and compiled
    * If you installed PySCF via pip, your compilation of mrh will probably fail
    * TODO: learn enough CMake to handle this automatically
- Python 3

### INSTALLATION:
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

### Notes
- The dev branch is continuously updated. The master branch is updated every time I pull PySCF and confirm that everything still works. If you have some issue and you think it may be related to PySCF version mismatch, try using the master branch and the precise PySCF commit indicated above.
- If you are using Intel MKL as the BLAS library, you may need to enable the corresponding cmake option:
`cmake -DBLA_VENDOR=Intel10_64lp_seq ..`

### ACKNOWLEDGMENTS:
- This work is supported by the U.S. Department of Energy, Office of Basic Energy Sciences, Division of Chemical Sciences, Geosciences and Biosciences through the Nanoporous Materials Genome Center under award DE-FG02-17ER16362.

