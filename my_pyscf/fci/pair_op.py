import numpy as np
from mrh.my_pyscf.fci import dummy
from pyscf.fci import direct_spin1, direct_uhf
from pyscf.fci.addons import _unpack_nelec

def contract_pair_op (op, cre, spin, ci, norb, nelec, link_index=None):
    ''' Act a pair creation or destruction operator on a CI vector '''
    s1 = int (spin>1)
    s2 = int (spin>0)
    ndum = 2 - (spin%2)
    nelec_ket = _unpack_nelec (nelec)
    occ_a, occ_b = int ((spin<2) and cre), int (spin>0 and cre)
    nelecd = [nelec_ket[0], nelec_ket[1]]
    for i in range (ndum):
        ci = dummy.add_orbital (ci, norb+i, nelecd, occ_a=occ_a, occ_b=occ_b)
        nelecd[0] += occ_a
        nelecd[1] += occ_b
    linkstr = direct_spin1._unpack (norb+ndum, nelecd, link_index)
    errmsg = ("For the pair-creation (pair-destruction) operator functions, the linkstr must "
              "be for nelec+2 (nelec) electrons occupying norb+1/norb+2 (ab/other spin case) "
              "orbitals.")
    assert (linkstr[0].shape[1]==(nelecd[0]*(norb+ndum-nelecd[0]+1))), errmsg
    assert (linkstr[1].shape[1]==(nelecd[1]*(norb+ndum-nelecd[1]+1))), errmsg
    hdum = np.zeros ([norb+ndum,]*4, dtype=op.dtype)
    if not cre: op = op.T
    if spin==1: op = op*.5
    hdum[:-ndum,-1,:-ndum,-ndum] = op[:,:]
    hdum += hdum.transpose (1,0,3,2)
    heff = [np.zeros_like (hdum) for i in range (3)]
    heff[spin] = hdum
    hci = direct_uhf.contract_2e (heff, ci, norb+ndum, nelecd, link_index=link_index)
    norb = norb + ndum
    occ_a, occ_b = int ((spin<2) and not cre), int (spin>0 and not cre)
    for i in range (ndum):
        norb -= 1
        nelecd[0] -= occ_a
        nelecd[1] -= occ_b
        hci = dummy.read_orbital (hci, norb, nelecd, occ_a=occ_a, occ_b=occ_b)
    return hci



