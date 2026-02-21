import os
import ctypes
import numpy as np
from pyscf import lib
from pyscf.fci import cistring
from pyscf.fci.addons import _unpack_nelec
from mrh.lib.helper import load_library

libpbcrdm = load_library('libpbc_fci_rdms')

def reorder_rdm(dm1, dm2, inplace=True):
    norb = dm1.shape[0]
    if not inplace:
        dm2 = dm2.copy()
    for r in range(norb):
        dm2[:, r, r, :] -= dm1.conj().T
    return dm1, dm2

def make_rdm1_spin1(fname, cibra, ciket, norb, nelec, link_index=None):
    '''
    Wrapper function for backend c function, to compute the spin-separated 1-RDMs.
    '''
    assert (cibra is not None and ciket is not None)
    cibra = np.asarray(cibra, order='C')
    ciket = np.asarray(ciket, order='C')
    dtype = ciket.dtype
    if link_index is None:
        neleca, nelecb = _unpack_nelec(nelec)
        link_indexa = link_indexb = cistring.gen_linkstr_index(range(norb), neleca)
        if neleca != nelecb:
            link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    else:
        link_indexa, link_indexb = link_index
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
    assert (cibra is not None and ciket is not None)
    cibra = np.asarray(cibra, order='C')
    ciket = np.asarray(ciket, order='C')
    if link_index is None:
        neleca, nelecb = _unpack_nelec(nelec)
        link_indexa = link_indexb = cistring.gen_linkstr_index(range(norb), neleca)
        if neleca != nelecb:
            link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    else:
        link_indexa, link_indexb = link_index

    na,nlinka = link_indexa.shape[:2]
    nb,nlinkb = link_indexb.shape[:2]

    assert (cibra.size == na*nb)
    assert (ciket.size == na*nb)

    rdm1 = np.empty((norb,norb), dtype=np.complex128, order='C')
    rdm2 = np.empty((norb,norb,norb,norb), dtype=np.complex128, order='C')

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

def make_rdm1s_cplx(fcivec, norb, nelec, link_index=None):
    dtype = fcivec.dtype
    neleca, nelecb = _unpack_nelec(nelec)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    C = np.asarray(fcivec).reshape(na, nb)

    if link_index is None:
        linka = cistring.gen_linkstr_index(range(norb), neleca)
        linkb = cistring.gen_linkstr_index(range(norb), nelecb)
    else:
        linka, linkb = link_index

    rdm1a = np.zeros((norb, norb), dtype=dtype)
    rdm1b = np.zeros((norb, norb), dtype=dtype)

    for a0, tab in enumerate(linka):
        for p, i, a1, sign in tab:
            if sign == 0:
                break
            rdm1a[p, i] += sign * np.vdot(C[a0, :], C[a1, :])

    for b0, tab in enumerate(linkb):
        for p, i, b1, sign in tab:
            if sign == 0:
                break
            rdm1b[p, i] += sign * np.vdot(C[:, b0], C[:, b1])

    return (rdm1a, rdm1b)

def make_rdm12s_cplx(fcivec, norb, nelec, link_index=None, reorder=True):
    
    fcivec /= np.linalg.norm(fcivec)
    na, nb = nelec
    na_str = cistring.num_strings(norb, na)
    nb_str = cistring.num_strings(norb, nb)
    fcivec = np.asarray(fcivec).reshape(na_str, nb_str)
    if link_index is None:
        link_indexa = cistring.gen_linkstr_index(range(norb), na)
        link_indexb = cistring.gen_linkstr_index(range(norb), nb)

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
        
    
    '''
    Few Checks:
    1. Hermiticity
    2. Traces
    3. 1-RDM from 2-RDM
    '''
    # assert np.max(np.abs(dm2aa - dm2aa.conj().transpose(2,3,0,1))) < 1e-8
    # assert np.max(np.abs(dm2bb - dm2bb.conj().transpose(2,3,0,1))) < 1e-8
    # dm1a_from_dm2aa =  np.einsum("pqrq->pr", dm2aa, optimize=True) / (na - 1)
    # dm1b_from_dm2bb =  np.einsum("pqrq->pr", dm2bb, optimize=True) / (nb - 1)
    # assert np.max(np.abs(dm1a_from_dm2aa - dm1a.conj().T)) < 1e-8, "1-RDM a diff: {}".format(np.max(np.abs(dm1a_from_dm2aa - dm1a.conj().T)))
    # assert np.max(np.abs(dm1b_from_dm2bb - dm1b.conj().T)) < 1e-8, "1-RDM b diff: {}".format(np.max(np.abs(dm1b_from_dm2bb - dm1b.conj().T)))

    dm2aa = dm2aa.transpose(0, 2, 1, 3).conj()
    dm2bb = dm2bb.transpose(0, 2, 1, 3).conj()
    return (dm1a, dm1b), (dm2aa, dm2ab, dm2bb)

def make_rdm1_cplx(fcivec, norb, nelec, link_index=None):
    (dm1a, dm1b) = make_rdm1s_cplx(fcivec, norb, nelec, link_index=link_index)
    rdm1 = dm1a + dm1b
    return rdm1.conj().T

def make_rdm12_cplx(fcivec, norb, nelec, link_index=None, reorder=True):
    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = \
        make_rdm12s_cplx(fcivec, norb, nelec, link_index=link_index, reorder=reorder)
    rdm1 = dm1a + dm1b
    dm2ba = dm2ab.conj().transpose(1,0,3,2)
    rdm2 = dm2aa + dm2bb + dm2ab + dm2ba
    rdm2 = 0.5*(rdm2 + rdm2.conj().transpose(1,0,3,2))
    return rdm1.conj().T, rdm2
