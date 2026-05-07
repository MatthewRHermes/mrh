import ctypes
import numpy as np

from pyscf import lib
from pyscf.fci import cistring

from mrh.lib.helper import load_library
from mrh.my_pyscf.pbc.fci.addons import _unpack, _unpack_nelec

libpbcrdm = load_library('libpbc_fci_rdms')

# Author: Bhavnesh Jangid

# In this file, there are function to compute the spin-separated 1-RDMs and 2-RDMs for a 
# complex FCI vector.
## TODO List:
# 1. Spin-summed excitation operator implementation in backend C code, which would be useful
#    for computing optimizing the code.

def reorder_rdm(dm1, dm2, inplace=True):
    norb = dm1.shape[0]
    if not inplace:
        dm2 = dm2.copy()
    for r in range(norb):
        dm2[:, r, r, :] -= dm1.conj().T
    return dm1, dm2

def make_rdm1_spin1(fname, cibra, ciket, norb, nelec, link_index=None):
    '''
    Wrapper function for backend C function, to compute the spin-separated 1-RDMs.
    '''
    assert (cibra is not None and ciket is not None)
    cibra = np.asarray(cibra, order='C')
    ciket = np.asarray(ciket, order='C')
    dtype = ciket.dtype
    link_indexa, link_indexb = _unpack(norb, nelec, link_index)
    na,nlinka = link_indexa.shape[:2]
    nb,nlinkb = link_indexb.shape[:2]
    assert (cibra.size == na*nb), '{} {} {}'.format (cibra.size, na, nb)
    assert (ciket.size == na*nb), '{} {} {}'.format (ciket.size, na, nb)
    rdm1 = np.empty((norb,norb), dtype=dtype, order='C')
    fn = getattr(libpbcrdm, fname)
    fn(rdm1.ctypes.data_as(ctypes.c_void_p),
       cibra.ctypes.data_as(ctypes.c_void_p),
       ciket.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(norb),
       ctypes.c_int(na), ctypes.c_int(nb),
       ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
       link_indexa.ctypes.data_as(ctypes.c_void_p),
       link_indexb.ctypes.data_as(ctypes.c_void_p))
    return rdm1.conj().T

def make_rdm12_spin1(fname, cibra, ciket, norb, nelec, link_index=None, symm=0):
    '''
    Wrapper function for backend C function, to compute the spin-separated 1-RDMs and 2-RDMs.
    args:
        fname: str, name of the C function to call
            either 'FCIrdm12kern_a_cplx' or 'FCIrdm12kern_b_cplx' or 'FCIrdm12kern_ab_cplx' 
             on the spin block of the 2-RDM to be computed
        cibra: np.ndarray of shape (na*nb, )
            complex FCI bra vector
        ciket: np.ndarray of shape (na*nb, )
            complex FCI ket vector
        norb: int
            number of orbitals (ncas * nkpts)
        nelec: tuple of ints
            number of alpha and beta electrons (nelecasa, nelecb)
        link_index: tuple of np.ndarray
            link indices for alpha and beta strings
        symm: int
            symmetry sector to compute the RDMs for, default is 0 (no symmetry)
    returns:
        rdm1: np.ndarray of shape (norb, norb)
            spin-separated 1-RDM (either alpha or beta depending on fname)
        rdm2: np.ndarray of shape (norb, norb, norb, norb)
            spin-separated 2-RDM (either alpha-alpha, beta-beta or alpha-beta depending on fname)
    '''
    assert (cibra is not None and ciket is not None)
    cibra = np.asarray(cibra, order='C')
    ciket = np.asarray(ciket, order='C')
    link_indexa, link_indexb = _unpack(norb, nelec, link_index)
    na,nlinka = link_indexa.shape[:2]
    nb,nlinkb = link_indexb.shape[:2]

    assert (cibra.size == na*nb)
    assert (ciket.size == na*nb)

    rdm1 = np.empty((norb,norb), dtype=np.complex128, order='C')
    rdm2 = np.empty((norb,norb,norb,norb), dtype=np.complex128, order='C')

    # In case I don't set the one of the OMP_THREADS or MKL_NUM_THREADS env variables to one
    # there is nested parallelization which slows down the code.
    with lib.with_omp_threads(1):
        libpbcrdm.FCIrdm12_drv_cplx(getattr(libpbcrdm, fname),
                            rdm1.ctypes.data_as(ctypes.c_void_p),
                            rdm2.ctypes.data_as(ctypes.c_void_p),
                            cibra.ctypes.data_as(ctypes.c_void_p),
                            ciket.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb),
                            ctypes.c_int(na), ctypes.c_int(nb),
                            ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                            link_indexa.ctypes.data_as(ctypes.c_void_p),
                            link_indexb.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(symm))
    return rdm1.conj().T, rdm2

def make_rdm1s_py(fcivec, norb, nelec, link_index=None):
    '''
    Python implementation of spin-separated 1-RDMs for a complex FCI vector.
    args:
        fcivec: np.ndarray of shape (na*nb, )
            complex FCI vector
        norb: int
            number of orbitals (ncas * nkpts)
        nelec: tuple of ints
            number of alpha and beta electrons (na, nb)
        link_index: tuple of np.ndarray
            link indices for alpha and beta strings
    returns:
        rdm1a, rdm1b: np.ndarray of shape (norb, norb)
            spin-separated 1-RDMs for alpha and beta spins
    '''
    dtype = fcivec.dtype
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa, link_indexb = _unpack(norb, nelec, link_index)

    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)

    civec = np.asarray(fcivec).reshape(na, nb)
    
    rdm1a = np.zeros((norb, norb), dtype=dtype)
    rdm1b = np.zeros((norb, norb), dtype=dtype)

    for a0, tab in enumerate(link_indexa):
        for p, i, a1, sign in tab:
            if sign == 0: break
            rdm1a[p, i] += sign * np.vdot(civec[a0, :], civec[a1, :])

    for b0, tab in enumerate(link_indexb):
        for p, i, b1, sign in tab:
            if sign == 0: break
            rdm1b[p, i] += sign * np.vdot(civec[:, b0], civec[:, b1])

    return (rdm1a, rdm1b)

def make_rdm12s_py(fcivec, norb, nelec, link_index=None, reorder=True):
    '''
    Python implementation of spin-separated 1-RDMs and 2-RDMs for a
    complex FCI vector.
    See above function for arguments. 
    Returns:
        rdm1a, rdm1b: np.ndarray of shape (norb, norb)
            spin-separated 1-RDMs for alpha and beta spins
        rdm2aa, rdm2ab, rdm2bb: np.ndarray of shape (norb, norb, norb, norb)
            spin-separated 2-RDMs for alpha-alpha, alpha-beta and beta-beta spins
    '''
    
    fcivec /= np.linalg.norm(fcivec)
    na, nb = nelec
    na_str = cistring.num_strings(norb, na)
    nb_str = cistring.num_strings(norb, nb)
    fcivec = np.asarray(fcivec).reshape(na_str, nb_str)
    link_indexa, link_indexb = _unpack(norb, nelec, link_index)
    
    # Initializing the arrays:
    dtype = fcivec.dtype
    dm1a = np.zeros((norb, norb), dtype=dtype)
    dm1b = np.zeros((norb, norb), dtype=dtype)
    dm2aa = np.zeros((norb, norb, norb, norb), dtype=dtype)
    dm2bb = np.zeros((norb, norb, norb, norb), dtype=dtype)
    dm2ab = np.zeros((norb, norb, norb, norb), dtype=dtype)

    # alpha, alpha block of dm2
    for ia, tab in enumerate(link_indexa):
        t1a = np.zeros((norb, norb, nb_str), dtype=dtype)
        for p, q, ja, sign in tab:
            if sign == 0:
                continue
            t1a[q, p, :] += sign * fcivec[ja, :]
        dm1a += np.einsum("B,qpB->pq", fcivec[ia, :].conj(), t1a, optimize=True)
        dm2aa += np.einsum("qpB,srB->pqsr", t1a.conj(), t1a, optimize=True)
    dm2aa = dm2aa.transpose(0, 2, 1, 3).conj() 

    # beta, beta block of dm2
    for ib, tab in enumerate(link_indexb):
        t1b = np.zeros((norb, norb, na_str), dtype=dtype)
        for p, q, ja, sign in tab:
            if sign == 0:
                continue
            t1b[q, p, :] += sign * fcivec[:, ja]
        dm1b += np.einsum("B,qpB->pq", fcivec[:, ib].conj(), t1b, optimize=True)
        dm2bb += np.einsum("qpB,srB->pqsr", t1b.conj(), t1b, optimize=True)
    dm2bb = dm2bb.transpose(0, 2, 1, 3).conj()

    # alpha, beta block of dm2
    beta_tabs = []
    for ib, tabb in enumerate(link_indexb):
        ops = []
        for r, s, jb, sgn_b in tabb:
            if sgn_b == 0:
                break
            ops.append((r, s, jb, sgn_b))
        beta_tabs.append(ops)

    for ia, taba in enumerate(link_indexa):
        Ta = np.zeros((norb, norb, nb_str), dtype=dtype)
        for p, q, ja, sgn_a in taba:
            if sgn_a == 0:
                break
            Ta[q, p, :] += sgn_a * fcivec[ja, :]
        cbra_row = fcivec[ia, :].conj()

        for ib, ops in enumerate(beta_tabs):
            cbra = cbra_row[ib]
            if cbra == 0:
                continue
            for r, s, jb, sgn_b in ops:
                dm2ab[:, :, s, r] += cbra * sgn_b * Ta[:, :, jb]

    if reorder:
        dm1a, dm2aa = reorder_rdm(dm1a, dm2aa)
        dm1b, dm2bb = reorder_rdm(dm1b, dm2bb)

    dm2aa = dm2aa.transpose(0, 2, 1, 3).conj()
    dm2bb = dm2bb.transpose(0, 2, 1, 3).conj()
    return (dm1a, dm1b), (dm2aa, dm2ab, dm2bb)

