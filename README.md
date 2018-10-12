# mrh
GPL research code of Matthew R. Hermes

### INSTALLATION:
- cd /path/to/mrh/lib
- mkdir build ; cd build
- cmake ..
- make
- Add /path/to/mrh to Python's search path somehow. Examples:
    * "sys.path.append ('/path/to/mrh')" in Python script files
    * "export PYTHONPATH=/path/to/mrh:$PYTHONPATH" in the shell
    * "ln -s /path/to/mrh /path/to/python/site-packages/mrh" in the shell

### DEPENDENCIES:
- PySCF
- Python 3

