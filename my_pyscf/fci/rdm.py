import numpy as np
from pyscf.fci import direct_spin1
from pyscf.fci.addons import _unpack_nelec
from mrh.my_pyscf.fci.addons import add_doubly_occupied_orbital
from mrh.my_pyscf.fci.addons import add_singly_occupied_orbital

def trans_rdm13hs (cibra_alpha, cibra_beta, ciket, norb, nelec, link_index=None):
    ''' Evaluate <cibra|r'p'q|ciket> '''
    ciket = add_doubly_occupied_orbital (ciket, norb, nelec)
    cibra = 0
    if cibra_alpha is not None:
        cibra = cibra + add_singly_occupied_orbital (cibra_alpha, norb, nelec, 1)
    if cibra_beta is not None:
        cibra = cibra + add_singly_occupied_orbital (cibra_beta, norb, nelec, 0)
    neleca, nelecb = _unpack_nelec (nelec)
    tdm1hs, tdm3hs = direct_spin1.trans_rdm12s (cibra, ciket, norb+1, (neleca+1,nelecb+1),
                                                link_index=link_index)
    tdm1hs = (-tdm1hs[0][0], -tdm1s[1][0])
    tdm3hs = (-tdm3hs[0][:,0], -tdm3hs[0][:,0], -tdm3hs[0][:,0], -tdm3hs[0][:,0])
    return tdm1hs, tdm3hs

def trans_rdm1hs (cibra_alpha, cibra_beta, ciket, norb, nelec, link_index=None):
    ''' Evaluate <cibra|p'|ciket> '''
    ciket = add_doubly_occupied_orbital (ciket, norb, nelec)
    cibra = 0
    if cibra_alpha is not None:
        cibra = cibra + add_singly_occupied_orbital (cibra_alpha, norb, nelec, 1)
    if cibra_beta is not None:
        cibra = cibra + add_singly_occupied_orbital (cibra_beta, norb, nelec, 0)
    neleca, nelecb = _unpack_nelec (nelec)
    tdm1ha, tdm1hb = direct_spin1.trans_rdm1s (cibra, ciket, norb+1, (neleca+1,nelecb+1),
                                               link_index=link_index)
    tdm1ha = tdm1ha[0]
    tdm1hb = tdm1hb[0]
    return tdm1ha, tdm1hb


