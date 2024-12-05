import numpy as np
import ctypes
from pyscf.fci.direct_spin1 import libfci, FCIvector
from pyscf.fci.direct_nosym import _unpack

# Variant of pyscf.fci.direct_nosym.contract_1e that contemplates spin-separated Hamiltonians

def contract_1e (h1e, fcivec, norb, nelec, link_index=None):
    ''' Variant of pyscf.fci.direct_nosym.contract_1e that contemplates a spin-separated h1e '''
    h1e = np.ascontiguousarray(h1e)
    fcivec = np.asarray(fcivec, order='C')
    link_indexa, link_indexb = _unpack(norb, nelec, link_index)
            
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    assert fcivec.size == na*nb
    assert fcivec.dtype == h1e.dtype == np.float64
    ci1 = np.zeros_like(fcivec)

    libfci.FCIcontract_a_1e_nosym(h1e[0].ctypes.data_as(ctypes.c_void_p),
                                  fcivec.ctypes.data_as(ctypes.c_void_p),
                                  ci1.ctypes.data_as(ctypes.c_void_p),
                                  ctypes.c_int(norb),
                                  ctypes.c_int(na), ctypes.c_int(nb),
                                  ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                                  link_indexa.ctypes.data_as(ctypes.c_void_p),
                                  link_indexb.ctypes.data_as(ctypes.c_void_p))
    libfci.FCIcontract_b_1e_nosym(h1e[1].ctypes.data_as(ctypes.c_void_p),
                                  fcivec.ctypes.data_as(ctypes.c_void_p),
                                  ci1.ctypes.data_as(ctypes.c_void_p),
                                  ctypes.c_int(norb),
                                  ctypes.c_int(na), ctypes.c_int(nb),
                                  ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                                  link_indexa.ctypes.data_as(ctypes.c_void_p),
                                  link_indexb.ctypes.data_as(ctypes.c_void_p))
    return ci1.view(FCIvector)


