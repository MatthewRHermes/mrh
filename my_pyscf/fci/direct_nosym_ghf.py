import numpy as np
import ctypes
from pyscf.fci.direct_spin1 import libfci, FCIvector
from pyscf.fci.direct_nosym import _unpack
from pyscf.fci.addons import civec_spinless_repr_generator, _unpack_nelec

# Variant of pyscf.fci.direct_nosym.contract_1e that contemplates purely spinless operators

def contract_1e (h1e, fcivec, norb, nelec, link_index=None):
    ''' Variant of pyscf.fci.direct_nosym.contract_1e that contemplates a purely spinless operator '''
    h1e = np.ascontiguousarray(h1e)
    neleca, nelecb = _unpack_nelec (nelec)
    ci0_r = [fcivec,]
    nelec_r = [nelec,]
    if (neleca<norb) and (nelecb>0):
        ci0_r.append (0)
        nelec_r.append ((neleca+1,nelecb-1))
    if (nelecb<norb) and (neleca>0):
        ci0_r.append (0)
        nelec_r.append ((neleca-1,nelecb+1))
    gen_fcivec, _, pack = civec_spinless_repr_generator (ci0_r, norb, nelec_r)
    fcivec = next (gen_fcivec ())
    norb = 2*norb
    nelec = (sum (nelec), 0)

    fcivec = np.asarray(fcivec, order='C')
    link_indexa, link_indexb = _unpack(norb, nelec, link_index)
            
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    assert fcivec.size == na*nb
    assert fcivec.dtype == h1e.dtype == np.float64
    ci1 = np.zeros_like(fcivec)

    libfci.FCIcontract_a_1e_nosym(h1e.ctypes.data_as(ctypes.c_void_p),
                                  fcivec.ctypes.data_as(ctypes.c_void_p),
                                  ci1.ctypes.data_as(ctypes.c_void_p),
                                  ctypes.c_int(norb),
                                  ctypes.c_int(na), ctypes.c_int(nb),
                                  ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                                  link_indexa.ctypes.data_as(ctypes.c_void_p),
                                  link_indexb.ctypes.data_as(ctypes.c_void_p))

    norb = norb // 2
    ci1_p = ci1_m = 0
    ci1_0 = pack (ci1, (neleca, nelecb)).view(FCIvector)
    if (neleca<norb) and (nelecb>0):
        ci1_p = pack (ci1, (neleca+1, nelecb-1)).view(FCIvector)
    if (nelecb<norb) and (neleca>0):
        ci1_m = pack (ci1, (neleca-1, nelecb+1)).view(FCIvector)

    return ci1_p, ci1_0, ci1_m


