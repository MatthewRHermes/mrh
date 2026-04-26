import ctypes
import numpy as np

from pyscf.fci import direct_spin1, cistring

from mrh.my_pyscf.pbc.fci.direct_spin1_cplx import _unpack, FCISolver as direct_spin1_cplx_FCISolver
from mrh.lib.helper import load_library

libpbcfci = load_library('libpbc_fci_contract_opt')

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

def contract_2e_spin0(eri, fcivec, norb, nelec, link_index=None):
    '''
    See above contract_2e for more details.

    This is the optimized contract_2e function for the spin0 case.
    Note, in the spin0 case, I can only take advantage of the symmetry of the HC term, since the 
    eri doesn't have the full 8 fold symmetries.
    '''

    link_indexa, link_indexb = _unpack(norb, nelec, link_index)
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]

    # Some Sanity Checks
    assert na == nb, 'You should use the direct_spin1, this is not the singlet spin case'
    assert nlinka == nlinkb
    assert fcivec.size == na * na
    assert eri.dtype == fcivec.dtype == np.complex128

    fcivec = np.asarray(fcivec, dtype=np.complex128, order='C')
    eri = np.asarray(eri, dtype=np.complex128, order='C')

    # TODO: Symmetry check for the incoming fcivec.
    out_CI = np.zeros_like(fcivec)
    
    # Choosing the strb_blksize.
    # TODO: use the mem to decide this.
    if nb < 1000:
        strb_blksize = nb
    else:
        x = int(np.ceil(nb / 1000.0))
        strb_blksize = int(np.ceil(nb / x))

    libpbcfci.FCIcontract_2es1_zgemm_spin0_blksize(
            eri.ctypes.data_as(ctypes.c_void_p),
            fcivec.ctypes.data_as(ctypes.c_void_p),
            out_CI.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(norb),
            ctypes.c_int(na),
            ctypes.c_int(nlinka),
            link_indexa.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(strb_blksize),
        )
    out_CI = out_CI.reshape(na, na)
    out_CI = out_CI + out_CI.T
    return out_CI.ravel().view(direct_spin1.FCIvector)

class FCISolverSpin0(direct_spin1_cplx_FCISolver):

    def get_init_guess(self, norb, nelec, nroots, hdiag):
        '''
        In case of singlet case, I would need to symmetrize the initial guess CI coefficients matrix, 
        which is originally generated for the spin1 case, to make it suitable for the spin0 case.
        '''
        init_ci = super().get_init_guess(norb, nelec, nroots, hdiag)
        neleca, nelecb = direct_spin1._unpack_nelec(nelec)
        assert neleca == nelecb
        na = cistring.num_strings(norb, neleca)
        if nroots > 1:
            sym_init_ci = []
            for ci in init_ci:
                ci = np.asarray(ci, dtype=np.complex128, order='C')
                ci_mat = ci.reshape(na, na)
                # Symmetrize the CI coefficients matrix to make it suitable for the spin0 case.
                ci_mat = 0.5 * (ci_mat + ci_mat.T) # Note it's not conjugate.
                sym_init_ci.append(ci_mat.ravel())
            return np.array(sym_init_ci, dtype=init_ci.dtype)
        else:
            ci = np.asarray(init_ci, dtype=np.complex128, order='C')
            ci_mat = ci.reshape(na, na)
            ci_mat = 0.5 * (ci_mat + ci_mat.T)
            return ci_mat.ravel().view(direct_spin1.FCIvector)

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None):
        return contract_2e_spin0(eri, fcivec, norb, nelec, link_index)
