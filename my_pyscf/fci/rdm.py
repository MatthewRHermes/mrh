import numpy as np
import math
from scipy import linalg
from pyscf.fci import direct_spin1, rdm
from pyscf.fci.addons import _unpack_nelec
from mrh.my_pyscf.fci import dummy

def _trans_rdm1hs (cibra, ciket, norb, nelec, spin=0, link_index=None):
    '''Evaluate the one-half-particle transition density matrix between ci vectors in different
    Hilbert spaces: <cibra|r'|ciket>, where |cibra> has the same number of orbitals but one
    additional electron of the same spin as r compared to |ciket>.

    Args:
        cibra: ndarray
            CI vector in (norb,nelec+1) Hilbert space
        ciket: ndarray
            CI vector in (norb,nelec) Hilbert space
        norb: integer
            Number of spatial orbitals 
        nelec: integer or sequence of length 2
            Number of electrons in the ket Hilbert space

    Kwargs:
        link_index: tuple of length 2 of "linkstr" type ndarray
            linkstr arrays for the nelec+1 electrons in norb+1 orbitals Hilbert space.
            See pyscf.fci.gen_linkstr_index for the shape of "linkstr".

    Returns:
        tdm1h: ndarray of shape (norb,)
            One-half-particle transition density matrix between cibra and ciket.
    '''
    nelec_ket = _unpack_nelec (nelec)
    nelec_bra = list (_unpack_nelec (nelec))
    nelec_bra[spin] += 1
    linkstr = direct_spin1._unpack (norb+1, nelec_bra, link_index)
    errmsg = ("For the half-particle transition density matrix functions, the linkstr must "
              "be for nelec+1 electrons occupying norb+1 orbitals.")
    for i in range (2): assert (linkstr[i].shape[1]==(nelec_bra[i]*(norb-nelec_bra[i]+2))), errmsg
    ciket = dummy.add_orbital (ciket, norb, nelec_ket, occ_a=(1-spin), occ_b=spin)
    cibra = dummy.add_orbital (cibra, norb, nelec_bra, occ_a=0, occ_b=0)
    fn = ('FCItrans_rdm1a', 'FCItrans_rdm1b')[spin]
    return rdm.make_rdm1_spin1 (fn, cibra, ciket, norb+1, nelec_bra, link_index)[-1,:-1]

def trans_rdm1ha (cibra, ciket, norb, nelec, link_index=None):
    '''Half-electron spin-up case of:\n''' + _trans_rdm1hs.__doc__
    return _trans_rdm1hs (cibra, ciket, norb, nelec, spin=0, link_index=link_index)

def trans_rdm1hb (cibra, ciket, norb, nelec, link_index=None):
    '''Half-electron spin-down case of:\n''' + _trans_rdm1hs.__doc__
    return _trans_rdm1hs (cibra, ciket, norb, nelec, spin=1, link_index=link_index)

def _trans_rdm13hs (cibra, ciket, norb, nelec, spin=0, link_index=None, reorder=True):
    ''' Evaluate the one-half- and three-half-particle transition density matrices between ci
    vectors in different Hilbert spaces: <cibra|r'p'q|ciket> and <cibra|r'|ciket>, where |cibra>
    has the same number of orbitals but one additional electron of the same spin as r compared to
    |ciket>.

    Args:
        cibra: ndarray
            CI vector in (norb,nelec+1) Hilbert space
        ciket: ndarray
            CI vector in (norb,nelec) Hilbert space
        norb: integer
            Number of spatial orbitals 
        nelec: integer or sequence of length 2
            Number of electrons in the ket Hilbert space

    Kwargs:
        link_index: tuple of length 2 of "linkstr" type ndarray
            linkstr arrays for the nelec+1 electrons in norb+1 orbitals Hilbert space.
            See pyscf.fci.gen_linkstr_index for the shape of "linkstr".

    Returns:
        tdm1h: ndarray of shape (norb,)
            One-half-particle transition density matrix between cibra and ciket.
        (tdm3ha, tdm3hb): ndarrays of shape (norb,norb,norb,)
            Three-half-particle transition density matrix between cibra and ciket, spin-up and
            spin-down cases of the full electron.
    '''
    nelec_ket = _unpack_nelec (nelec)
    nelec_bra = list (_unpack_nelec (nelec))
    nelec_bra[spin] += 1
    linkstr = direct_spin1._unpack (norb+1, nelec_bra, link_index)
    errmsg = ("For the half-particle transition density matrix functions, the linkstr must "
              "be for nelec+1 electrons occupying norb+1 orbitals.")
    for i in range (2): assert (linkstr[i].shape[1]==(nelec_bra[i]*(norb-nelec_bra[i]+2))), errmsg
    ciket = dummy.add_orbital (ciket, norb, nelec_ket, occ_a=(1-spin), occ_b=spin)
    cibra = dummy.add_orbital (cibra, norb, nelec_bra, occ_a=0, occ_b=0)
    fn_par = ('FCItdm12kern_a', 'FCItdm12kern_b')[spin]
    fn_ab = 'FCItdm12kern_ab'
    tdm1h, tdm3h_par = rdm.make_rdm12_spin1 (fn_par, cibra, ciket, norb+1, nelec_bra, link_index, 2)
    if reorder: tdm1h, tdm3h_par = rdm.reorder_rdm (tdm1h, tdm3h_par, inplace=True)
    if spin:
        tdm3ha = rdm.make_rdm12_spin1 (fn_ab, ciket, cibra, norb+1, nelec_bra, link_index, 0)[1]
        tdm3ha = tdm3ha.transpose (3,2,1,0)
        tdm3hb = tdm3h_par
    else:
        tdm3ha = tdm3h_par
        tdm3hb = rdm.make_rdm12_spin1 (fn_ab, cibra, ciket, norb+1, nelec_bra, link_index, 0)[1]

    return tdm1h[-1,:-1], (tdm3ha[:-1,-1,:-1,:-1], tdm3hb[:-1,-1,:-1,:-1])

def trans_rdm13ha (cibra, ciket, norb, nelec, link_index=None):
    '''Half-electron spin-up case of:\n''' + _trans_rdm1hs.__doc__
    return _trans_rdm13hs (cibra, ciket, norb, nelec, spin=0, link_index=link_index)

def trans_rdm13hb (cibra, ciket, norb, nelec, link_index=None):
    '''Half-electron spin-down case of:\n''' + _trans_rdm1hs.__doc__
    return _trans_rdm13hs (cibra, ciket, norb, nelec, spin=1, link_index=link_index)


