import ctypes
import numpy as np

from pyscf import lib, __config__
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

def _reorder_rdm_tdm(dm1, dm2, inplace=True, tdm=False):
    norb = dm1.shape[0]
    if not inplace:
        dm2 = dm2.copy()
    for r in range(norb):
        if tdm:
            dm2[:, r, r, :] -= dm1.T
        else:
            dm2[:, r, r, :] -= dm1.conj().T
    return dm1, dm2


def reorder_rdm(dm1, dm2, inplace=True):
    return _reorder_rdm_tdm(dm1, dm2, inplace=inplace, tdm=False)


def reorder_tdm(tdm1, tdm2, inplace=True):
    return _reorder_rdm_tdm(tdm1, tdm2, inplace=inplace, tdm=True)


def _make_rdm1_tdm1_spin1(fname, cibra, ciket, norb, nelec,
                           link_index=None, tdm=False):
    r'''
    Call a complex spin-resolved 1-RDM or transition-1-RDM kernel.
    '''
    assert (cibra is not None and ciket is not None)

    cibra = np.asarray(cibra, order='C')
    ciket = np.asarray(ciket, order='C')
    if cibra.dtype is not np.complex128 or ciket.dtype is not np.complex128:
        cibra = cibra.astype(np.complex128)
        ciket = ciket.astype(np.complex128)

    dtype = ciket.dtype
    link_indexa, link_indexb = _unpack(norb, nelec, link_index)
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    assert cibra.size == na * nb, '{} {} {}'.format(cibra.size, na, nb)
    assert ciket.size == na * nb, '{} {} {}'.format(ciket.size, na, nb)

    dm1 = np.empty((norb, norb), dtype=np.complex128, order='C')
    fn = getattr(libpbcrdm, fname)
    fn(dm1.ctypes.data_as(ctypes.c_void_p),
       cibra.ctypes.data_as(ctypes.c_void_p),
       ciket.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(norb),
       ctypes.c_int(na), ctypes.c_int(nb),
       ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
       link_indexa.ctypes.data_as(ctypes.c_void_p),
       link_indexb.ctypes.data_as(ctypes.c_void_p))
    return dm1.T if tdm else dm1.conj().T


def make_rdm1_spin1(fname, cibra, ciket, norb, nelec, link_index=None):
    r'''Wrapper for a complex spin-resolved ordinary 1-RDM kernel.'''
    return _make_rdm1_tdm1_spin1(fname, cibra, ciket, norb, nelec, 
                                 link_index, tdm=False)

def trans_rdm1_spin1(fname, cibra, ciket, norb, nelec, link_index=None):
    r'''Wrapper for a complex spin-resolved transition 1-RDM kernel.'''
    return _make_rdm1_tdm1_spin1(fname, cibra, ciket, norb, nelec, 
                                 link_index, tdm=True)

def _make_rdm12_tdm12_spin1(fname, cibra, ciket, norb, nelec,
                             link_index=None, symm=0, tdm=False):
    r'''Call a complex spin-resolved 1-/2-RDM or transition-RDM kernel.'''
    assert cibra is not None and ciket is not None
    cibra = np.asarray(cibra, dtype=np.complex128, order='C')
    ciket = np.asarray(ciket, dtype=np.complex128, order='C')
    link_indexa, link_indexb = _unpack(norb, nelec, link_index)
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    assert cibra.size == na * nb
    assert ciket.size == na * nb

    dm1 = np.empty((norb, norb), dtype=np.complex128, order='C')
    dm2 = np.empty((norb, norb, norb, norb),
                   dtype=np.complex128, order='C')
    # Follow the current PySCF thread setting unless explicitly overridden.
    with lib.with_omp_threads(getattr(__config__, "pbc_rdm_threads", lib.num_threads())):
        libpbcrdm.FCIrdm12_drv_cplx(
            getattr(libpbcrdm, fname),
            dm1.ctypes.data_as(ctypes.c_void_p),
            dm2.ctypes.data_as(ctypes.c_void_p),
            cibra.ctypes.data_as(ctypes.c_void_p),
            ciket.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(norb),
            ctypes.c_int(na), ctypes.c_int(nb),
            ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
            link_indexa.ctypes.data_as(ctypes.c_void_p),
            link_indexb.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(symm))
    return (dm1.T if tdm else dm1.conj().T), dm2


def make_rdm12_spin1(fname, cibra, ciket, norb, nelec, link_index=None,
                     symm=0):
    r'''Wrapper for a complex spin-resolved ordinary 1-/2-RDM kernel.'''
    return _make_rdm12_tdm12_spin1(fname, cibra, ciket, norb, nelec, 
                                   link_index, symm, tdm=False)


def trans_rdm12_spin1(fname, cibra, ciket, norb, nelec, link_index=None,
                      symm=0):
    r'''Wrapper for a complex spin-resolved transition 1-/2-RDM kernel.'''
    return _make_rdm12_tdm12_spin1(fname, cibra, ciket, norb, nelec, 
                                   link_index, symm, tdm=True)

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


def make_tdm1s_py(cibra, ciket, norb, nelec, link_index=None):
    r'''Python implementation of spin-separated transition 1-RDMs.

    ``tdm1[p,q] = <cibra|q^\dagger p|ciket>``.  Neither CI vector is
    normalized or modified.
    '''
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa, link_indexb = _unpack(norb, nelec, link_index)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    cibra = np.asarray(cibra).reshape(na, nb)
    ciket = np.asarray(ciket).reshape(na, nb)
    dtype = np.result_type(cibra.dtype, ciket.dtype)

    tdm1a = np.zeros((norb, norb), dtype=dtype)
    tdm1b = np.zeros((norb, norb), dtype=dtype)
    for a0, tab in enumerate(link_indexa):
        for a, i, a1, sign in tab:
            if sign == 0:
                break
            tdm1a[i, a] += sign * np.vdot(cibra[a1, :], ciket[a0, :])
    for b0, tab in enumerate(link_indexb):
        for a, i, b1, sign in tab:
            if sign == 0:
                break
            tdm1b[i, a] += sign * np.vdot(cibra[:, b1], ciket[:, b0])
    return tdm1a, tdm1b


def _make_tdm12_same_spin_py(cibra, ciket, norb, link_index,
                              other_size, reorder):
    dtype = np.result_type(cibra.dtype, ciket.dtype)
    tdm1 = np.zeros((norb, norb), dtype=dtype)
    tdm2 = np.zeros((norb, norb, norb, norb), dtype=dtype)
    for str0, tab in enumerate(link_index):
        t1bra = np.zeros((norb, norb, other_size), dtype=dtype)
        t1ket = np.zeros_like(t1bra)
        for a, i, str1, sign in tab:
            if sign == 0:
                break
            t1bra[i, a, :] += sign * cibra[str1, :]
            t1ket[i, a, :] += sign * ciket[str1, :]
        tdm1 += np.einsum(
            'B,iaB->ai', cibra[str0, :].conj(), t1ket, optimize=True)
        tdm2 += np.einsum(
            'iaB,jbB->aijb', t1bra.conj(), t1ket, optimize=True)
    if reorder:
        tdm1, tdm2 = reorder_tdm(tdm1, tdm2, inplace=True)
    return tdm1, tdm2


def _make_tdm2ab_py(cibra, ciket, norb, link_indexa, link_indexb):
    nb = ciket.shape[1]
    dtype = np.result_type(cibra.dtype, ciket.dtype)
    tdm2ab = np.zeros((norb, norb, norb, norb), dtype=dtype)
    beta_tabs = []
    for tab in link_indexb:
        ops = []
        for a, i, str1, sign in tab:
            if sign == 0:
                break
            ops.append((a, i, str1, sign))
        beta_tabs.append(ops)

    for stra, tab in enumerate(link_indexa):
        t1keta = np.zeros((norb, norb, nb), dtype=dtype)
        for a, i, str1, sign in tab:
            if sign == 0:
                break
            t1keta[i, a, :] += sign * ciket[str1, :]
        bra_row = cibra[stra, :].conj()
        for strb, ops in enumerate(beta_tabs):
            for a, i, str1, sign in ops:
                tdm2ab[:, :, i, a] += (
                    bra_row[strb] * sign * t1keta[:, :, str1])
    return tdm2ab


def make_tdm12s_py(cibra, ciket, norb, nelec, link_index=None,
                    reorder=True):
    r'''Python implementation of spin-separated transition 1- and 2-RDMs.

    Returns ``(tdm1a, tdm1b), (tdm2aa, tdm2ab, tdm2ba, tdm2bb)`` with
    ``tdm2[p,q,r,s] = <cibra|p^\dagger r^\dagger s q|ciket>`` when
    ``reorder`` is true.
    '''
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa, link_indexb = _unpack(norb, nelec, link_index)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    cibra = np.asarray(cibra).reshape(na, nb)
    ciket = np.asarray(ciket).reshape(na, nb)

    tdm1a, tdm2aa = _make_tdm12_same_spin_py(
        cibra, ciket, norb, link_indexa, nb, reorder)
    tdm1b, tdm2bb = _make_tdm12_same_spin_py(
        cibra.T, ciket.T, norb, link_indexb, na, reorder)
    tdm2ab = _make_tdm2ab_py(
        cibra, ciket, norb, link_indexa, link_indexb)
    tdm2ba = _make_tdm2ab_py(
        ciket, cibra, norb, link_indexa, link_indexb)
    tdm2ba = tdm2ba.transpose(3, 2, 1, 0).conj()

    return (tdm1a, tdm1b), (tdm2aa, tdm2ab, tdm2ba, tdm2bb)
