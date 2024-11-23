import numpy as np
from mrh.my_pyscf.fci import dummy
from pyscf.fci import direct_spin1
from pyscf.fci.addons import _unpack_nelec

def contract_1he (h1he, cre, spin, ci, norb, nelec, link_index=None):
    neleca, nelecb = _unpack_nelec (nelec)
    nelec_bra = [neleca, nelecb]
    ket_occ = [0,0]
    bra_occ = [0,0]
    if cre: 
        ket_occ[spin] = 1
        nelec_bra[spin] += 1
    else:
        bra_occ[spin] = 1
        nelec_bra[spin] -= 1
    nelecd = [neleca + ket_occ[0], nelecb + ket_occ[1]]
    ci = dummy.add_orbital (ci, norb, nelec, occ_a=ket_occ[0], occ_b=ket_occ[1])
    f1e = np.zeros ((norb+1,norb+1), dtype=h1he.dtype)
    f1e[-1,:-1] = h1he[:]
    f1e += f1e.T
    hci = direct_spin1.contract_1e (f1e, ci, norb+1, nelecd, link_index=link_index)
    return dummy.read_orbital (hci, norb, nelec_bra, occ_a=bra_occ[0], occ_b=bra_occ[1])

def absorb_h1he (h1he, h3he, cre, spin, norb, nelec, fac=1):
    neleca, nelecb = _unpack_nelec (nelec)
    nelecd = [neleca, nelecb]
    if cre: nelecd[spin] += 1
    f1e = np.zeros ((norb+1,norb+1), dtype=h1he.dtype)
    f1e[-1,:-1] = h1he[:]
    f1e += f1e.T
    f2e = np.zeros ((norb+1,norb+1,norb+1,norb+1), dtype=h3he.dtype)
    if cre:
        f2e[:-1,-1,:-1,:-1] = h3he[:,:,:] # pph
    else:
        f2e[:-1,:-1,-1,:-1] = h3he[:,:,:] # phh
    f2e += f2e.transpose (1,0,3,2)
    f2e += f2e.transpose (2,3,0,1)
    return direct_spin1.absorb_h1e (f1e, f2e, norb+1, nelecd, fac=fac)

def contract_3he (h3heff, cre, spin, ci, norb, nelec, link_index=None):
    neleca, nelecb = _unpack_nelec (nelec)
    ket_occ = [0,0]
    bra_occ = [0,0]
    if cre: 
        ket_occ[spin] = 1
    else:
        bra_occ[spin] = 1
    nelecd = [neleca + ket_occ[0], nelecb + ket_occ[1]]
    ci = dummy.add_orbital (ci, norb, nelec, occ_a=ket_occ[0], occ_b=ket_occ[1])
    hci = direct_spin1.contract_2e (h3heff, ci, norb+1, nelecd, link_index=link_index)
    return dummy.read_orbital (hci, norb, nelec, occ_a=bra_occ[0], occ_b=bra_occ[1])

