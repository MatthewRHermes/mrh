# mrh
GPL research code of Matthew R. Hermes

### INSTALLATION:
- cd /path/to/mrh/lib
- mkdir build ; cd build
- cmake ..
- make
- Add /path/to/mrh to Python's search path somehow. Examples:
    * "sys.path.append ('/path/to')" in Python script files
    * "export PYTHONPATH=/path/to:$PYTHONPATH" in the shell
    * "ln -s /path/to/mrh /path/to/python/site-packages/mrh" in the shell

### DEPENDENCIES:
- PySCF
- Python 3

### ACKNOWLEDGMENTS:
- This work is supported by the U.S. Department of Energy, Office of Basic Energy Sciences, Division of Chemical Sciences, Geosciences and Biosciences through the Nanoporous Materials Genome Center under award DE-FG02-17ER16362.

