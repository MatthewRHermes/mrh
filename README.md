# mrh
GPL research code of Matthew R. Hermes

Because ctypes.util.find_library function doesn't seem to work, you're going to need to find all the ctypes.CDLL
function calls and change the paths to the exact paths to your installation of the shared libraries.
Currently there is only one: libqcdmet.so in my_dmet/qcdmethelper.py. If you installed Sebastian Wouters' QC-DMET, you should
have this somewhere. At some point I'll actually put the source code for that library in here so you don't have to do that.

### DEPENDENCIES:
- PySCF
- CheMPS2
- QC-DMET

