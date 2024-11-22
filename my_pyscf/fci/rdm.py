import numpy as np
import math
from pyscf.fci import direct_spin1
from pyscf.fci.addons import _unpack_nelec
from mrh.my_pyscf.fci.addons import add_doubly_occupied_orbital
from mrh.my_pyscf.fci.addons import add_singly_occupied_bottom_orbital

def trans_rdm13hs (cibra_alpha, cibra_beta, ciket, norb, nelec, link_index=None):
    ''' Evaluate the one-half- and three-half-particle transition density matrices between ci
    vectors in different Hilbert spaces: <cibra|r'p'q|ciket> and <cibra|r'|ciket>, where |cibra>
    has the same number of orbitals but one additional electron of the same spin as r compared to
    |ciket>.

    Args:
        cibra_alpha: ndarray or None
            CI vector in neleca+1 Hilbert space. Must not be None if cibra_beta is None.
        cibra_beta: ndarray or None
            CI vector in nelecb+1 Hilbert space. Must not be None if cibra_alpha is None.
        ciket: ndarray
            CI vector in (norb,nelec) Hilbert space.
        norb: integer
            Number of spatial orbitals
        nelec: integer or sequence of length 2
            Number of electrons in the ket Hilbert space

    Kwargs:
        link_index: tuple of length 2 of "linkstr" type ndarray
            linkstr arrays for the (neleca+1,nelecb+1) electrons in norb+1 orbitals Hilbert space.
            See pyscf.fci.gen_linkstr_index for the shape of "linkstr".

    Returns:
        tdm1hs: tuple of length 2 of ndarray of shape (norb,)
            Spin-separated one-half-particle transition density matrix. The first element is
            between cibra_alpha and ciket, and the second is between cibra_beta and ciket. If
            either is None, the corresponding array elements are zero.
        tdm3hs: tuple of length 4 of ndarray of shape (norb,norb,norb)
            Spin-separated three-half-particle transition density matrix. The first two elements
            are between cibra_alpha and ciket, and the next two are between cibra_beta and ciket.
            If either is None, the corresponding array elements are zero.
    '''
    assert ((cibra_alpha is not None) or (cibra_beta is not None))
    neleca, nelecb = _unpack_nelec (nelec)
    if link_index is not None:
        link_indexa, link_indexb = direct_spin1._unpack (link_index)
        errmsg = ("For the half-particle transition density matrix functions, the linkstr must "
                  "be for (neleca+1,nelecb+1) electrons occupying norb+1 orbitals.")
        assert (link_indexa.shape[1]==(neleca+1)*(norb-neleca)+neleca+1), errmsg
        assert (link_indexb.shape[1]==(nelecb+1)*(norb-nelecb)+nelecb+1), errmsg
    ciket = add_doubly_occupied_orbital (ciket, norb, nelec)
    cibra = 0
    fac = 0.0
    if cibra_alpha is not None:
        fac += 1.0
        cibra = cibra + add_singly_occupied_bottom_orbital (cibra_alpha,norb,(neleca+1,nelecb),1)
    if cibra_beta is not None:
        fac += 1.0
        cibra = cibra + add_singly_occupied_bottom_orbital (cibra_beta,norb,(neleca,nelecb+1),0)
    fac = math.sqrt (fac)
    cibra /= fac
    tdm1hs, tdm3hs = direct_spin1.trans_rdm12s (cibra, ciket, norb+1, (neleca+1,nelecb+1),
                                                link_index=link_index)
    tdm1hs = (-fac*tdm1hs[0][0,1:], -fac*tdm1hs[1][0,1:])
    tdm3hs = (-fac*tdm3hs[0][1:,0,1:,1:], -fac*tdm3hs[1][1:,0,1:,1:],
              -fac*tdm3hs[1][1:,0,1:,1:], -fac*tdm3hs[1][1:,0,1:,1:])
    return tdm1hs, tdm3hs

def trans_rdm1hs (cibra_alpha, cibra_beta, ciket, norb, nelec, link_index=None):
    ''' Evaluate the one-half-particle transition density matrix between ci vectors in different
    Hilbert spaces: <cibra|r'|ciket>, where |cibra> has the same number of orbitals but one
    additional electron of the same spin as r compared to |ciket>.

    Args:
        cibra_alpha: ndarray or None
            CI vector in neleca+1 Hilbert space. Must not be None if cibra_beta is None.
        cibra_beta: ndarray or None
            CI vector in nelecb+1 Hilbert space. Must not be None if cibra_alpha is None.
        ciket: ndarray
            CI vector in (norb,nelec) Hilbert space.
        norb: integer
            Number of spatial orbitals 
        nelec: integer or sequence of length 2
            Number of electrons in the ket Hilbert space

    Kwargs:
        link_index: tuple of length 2 of "linkstr" type ndarray
            linkstr arrays for the (neleca+1,nelecb+1) electrons in norb+1 orbitals Hilbert space.
            See pyscf.fci.gen_linkstr_index for the shape of "linkstr".

    Returns:
        tdm1ha: ndarray of shape (norb,)
            Spin-up one-half-particle transition density matrix between cibra_alpha and ciket.
            Elements are zero if cibra_alpha is None.
        tdm1hb: ndarray of shape (norb,)
            Spin-down one-half-particle transition density matrix between cibra_beta and ciket.
            Elements are zero if cibra_beta is None.
    '''
    assert ((cibra_alpha is not None) or (cibra_beta is not None))
    neleca, nelecb = _unpack_nelec (nelec)
    if link_index is not None:
        link_indexa, link_indexb = direct_spin1._unpack (link_index)
        errmsg = ("For the half-particle transition density matrix functions, the linkstr must "
                  "be for (neleca+1,nelecb+1) electrons occupying norb+1 orbitals.")
        assert (link_indexa.shape[1]==(neleca+1)*(norb-neleca)+neleca+1), errmsg
        assert (link_indexb.shape[1]==(nelecb+1)*(norb-nelecb)+nelecb+1), errmsg
    ciket = add_doubly_occupied_orbital (ciket, norb, nelec)
    cibra = 0
    fac = 0.0
    if cibra_alpha is not None:
        fac += 1.0
        cibra = cibra + add_singly_occupied_bottom_orbital (cibra_alpha,norb,(neleca+1,nelecb),1)
    if cibra_beta is not None:
        fac += 1.0
        cibra = cibra + add_singly_occupied_bottom_orbital (cibra_beta,norb,(neleca,nelecb+1),0)
    fac = math.sqrt (fac)
    cibra /= fac
    tdm1ha, tdm1hb = direct_spin1.trans_rdm1s (cibra, ciket, norb+1, (neleca+1,nelecb+1),
                                               link_index=link_index)
    tdm1ha = -fac*tdm1ha[0,1:]
    tdm1hb = -fac*tdm1hb[0,1:]
    return tdm1ha, tdm1hb


