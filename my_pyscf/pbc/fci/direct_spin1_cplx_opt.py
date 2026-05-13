import ctypes
import numpy as np

from pyscf import lib, __config__
from pyscf.fci import direct_spin1

from mrh.lib.helper import load_library
from mrh.my_pyscf.pbc.fci.direct_spin1_cplx import _unpack
from mrh.my_pyscf.pbc.fci.direct_spin1_cplx import FCISolver as direct_spin1_cplx_FCISolver

libpbcfci = load_library('libpbc_fci_contract_opt')

# Author: Bhavnesh Jangid

# I am defining a global variable for the threads, what I have seen
# in my local computer defining the number of threads to 1 gives the best performance for 
# the direct_spin1_cplx_opt.FCISolver however on the supercomputer it seems the nest optimization 
# is not working when the number of threads is set to 1.

contract_2e_threads = getattr(__config__, 'pbc_contract_2e_threads', None)

def contract_2e(eri, fcivec, norb, nelec, link_index=None):
    direct_spin1.contract_2e.__doc__ + '''
    args:
        eri: np.ndarray (norb, norb, norb, norb)
            2-electron integrals in the MO basis (complex)
        fcivec: np.ndarray (ndet,)
            FCI vector to be contracted (complex)
        norb: int
            number of orbitals
        nelec: int or tuple
            number of electrons
        link_index: tuple (Currently stored without trilidx)
            link index for the FCI vector
    returns:
        h2e_fcivec: np.ndarray (ndet,)
            the result of contracting eri with fcivec, i.e. h2e_fcivec = eri * fcivec
    '''
    link_indexa, link_indexb = _unpack(norb, nelec, link_index)
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    assert fcivec.size == na*nb
    assert eri.dtype == fcivec.dtype == np.complex128
    fcivec = np.asarray(fcivec, dtype=np.complex128, order='C')
    eri = np.asarray(eri, dtype=np.complex128, order='C')
    out_CI = np.zeros_like(fcivec)

    # Currently the strb_blksize is hard coded to 240. But that makes it so slow for the 
    # AS > (8e, 8o), hence I am setting the strb_blksize here. Currently I am setting the nb to less
    # than 1000, which would require atmost of 1.5-2.0 GB memory for (12e, 12o) AS. Later on based
    # on the memory available, we can set the nb and strb_blksize accordingly.
    if nb < 1000: 
        strb_blksize = nb
    else: 
        x = int(np.ceil(nb / 1000.0))
        strb_blksize = int(np.ceil(nb / x))

    with lib.with_omp_threads(contract_2e_threads):
        libpbcfci.FCIcontract_2es1_zgemm_blksize(eri.ctypes.data_as(ctypes.c_void_p),
                                    fcivec.ctypes.data_as(ctypes.c_void_p),
                                    out_CI.ctypes.data_as(ctypes.c_void_p),
                                    ctypes.c_int(norb),
                                    ctypes.c_int(na), ctypes.c_int(nb),
                                    ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                                    link_indexa.ctypes.data_as(ctypes.c_void_p),
                                    link_indexb.ctypes.data_as(ctypes.c_void_p),
                                    ctypes.c_int(strb_blksize))
    return out_CI.view(direct_spin1.FCIvector)

class FCISolver(direct_spin1_cplx_FCISolver):
    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None):
        return contract_2e(eri, fcivec, norb, nelec, link_index)
