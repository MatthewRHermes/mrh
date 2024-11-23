from mrh.my_pyscf.fci.addons import add_empty_orbital
from mrh.my_pyscf.fci.addons import add_singly_occupied_orbital
from mrh.my_pyscf.fci.addons import read_dummy_orbital
from pyscf.fci import direct_spin1
from pyscf.fci.addons import _unpack_nelec

def contract_1he (h1he, cre, spin, ci, norb, nelec, link_index=None):
    neleca, nelecb = _unpack_nelec (nelec)
    nelecd = [neleca, nelecb]
    if cre: ci = add_empty_orbital (ci, norb, nelec)
    else: 
        ci = add_singly_occupied_orbital (ci, norb, nelec, spin)
        nelecd[spin] += 1
    f1e = np.zeros ((norb+1,norb+1), dtype=h1he.dtype)
    f1e[-1,:-1] = h1he[:]
    f1e += f1e.T
    hci = direct_spin1.contract_1e (f1e, ci, norb+1, nelecd, link_index=link_index)
    return read_dummy_orbital (hci, norb, nelec, spin=spin, occ=(not cre))

def absorb_h1he (h1he, h3he, cre, spin, norb, nelec, fac=1):
    neleca, nelecb = _unpack_nelec (nelec)
    nelecd = [neleca, nelecb]
    if cre: ci = add_empty_orbital (ci, norb, nelec)
    else: 
        ci = add_singly_occupied_orbital (ci, norb, nelec, spin)
        nelecd[spin] += 1
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
    nelecd = [neleca, nelecb]
    if cre: ci = add_empty_orbital (ci, norb, nelec)
    else: 
        ci = add_singly_occupied_orbital (ci, norb, nelec, spin)
        nelecd[spin] += 1
    hci = direct_spin1.contract_2e (h3heff, ci, norb+1, nelecd, link_index=link_index)
    return read_dummy_orbital (hci, norb, nelec, spin=spin, occ=(not cre))

