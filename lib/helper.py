import sys, os
import numpy


def version_tuple(version):
    return tuple(map(int, version.split('.')))


''' This bit copied from PySCF '''
def load_library(libname):
# numpy 1.6 has bug in ctypeslib.load_library, see numpy/distutils/misc_util.py
    if version_tuple(numpy.__version__)[:2] == version_tuple('1.6'):
        if (sys.platform.startswith('linux') or
            sys.platform.startswith('gnukfreebsd')):
            so_ext = '.so'
        elif sys.platform.startswith('darwin'):
            so_ext = '.dylib'
        elif sys.platform.startswith('win'):
            so_ext = '.dll'
        else:
            raise OSError('Unknown platform')
        libname_so = libname + so_ext
        return ctypes.CDLL(os.path.join(os.path.dirname(__file__), libname_so))
    else:
        _loaderpath = os.path.dirname(__file__)
        return numpy.ctypeslib.load_library(libname, _loaderpath)

