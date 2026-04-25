import ctypes
import numpy as np

from pyscf.fci import direct_spin1

from mrh.my_pyscf.pbc.fci.direct_spin1_cplx import _unpack, FCISolver as direct_spin1_cplx_FCISolver
from mrh.lib.helper import load_library


libpbcfci = load_library('libpbc_fci_contract_nosym')


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
    libpbcfci.FCIcontract_2es1_zgemm(eri.ctypes.data_as(ctypes.c_void_p),
                                fcivec.ctypes.data_as(ctypes.c_void_p),
                                out_CI.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(norb),
                                ctypes.c_int(na), ctypes.c_int(nb),
                                ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                                link_indexa.ctypes.data_as(ctypes.c_void_p),
                                link_indexb.ctypes.data_as(ctypes.c_void_p))
    return out_CI.view(direct_spin1.FCIvector)


class FCISolver(direct_spin1_cplx_FCISolver):
    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None):
        return contract_2e(eri, fcivec, norb, nelec, link_index)
