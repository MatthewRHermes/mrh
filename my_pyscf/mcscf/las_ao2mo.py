import ctypes

from functools import reduce
import numpy
import h5py
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo import outcore

libmcscf = lib.load_library('libmcscf')

def _mem_usage(las, mo_coeff, ncore, ncas, nmo):
    ncore = las.ncore
    ncas = las.ncas
    nmo = mo_coeff.shape[-1]
    nocc = ncore + ncas
    outcore = basic = ncas**2*nmo**2*2 * 8/1e6
    incore = outcore + (ncore+ncas)*nmo**3*4/1e6
    return incore, outcore, basic
