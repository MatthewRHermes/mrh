### Adapted from github.com/hczhai/fci-siso/blob/master/fcisiso.py ###

import numpy as np
import copy
from pyscf.data import nist

def get_jk(mol, dm0):
    hso2e = mol.intor('int2e_p1vxp1', 3).reshape(3, mol.nao, mol.nao, mol.nao, mol.nao)
    vj = np.einsum('yijkl,lk->yij', hso2e, dm0)
    vk = np.einsum('yijkl,jk->yil', hso2e, dm0)
    vk += np.einsum('yijkl,li->ykj', hso2e, dm0)
    return vj, vk

def get_jk_amfi(mol, dm0):
    nao = mol.nao_nr()
    aoslice = mol.aoslice_by_atom()
    vj = np.zeros((3, nao, nao))
    vk = np.zeros((3, nao, nao))
    atom = copy.copy(mol)

    for ia in range(mol.natm):
        b0, b1, p0, p1 = aoslice[ia]
        atom._bas = mol._bas[b0:b1]
        vj1, vk1 = get_jk(atom, dm0[p0:p1, p0:p1])
        vj[:, p0:p1, p0:p1] = vj1
        vk[:, p0:p1, p0:p1] = vk1
    return vj, vk

def compute_hso_amfi(mol, dm0): 
    alpha2 = nist.ALPHA ** 2
    aoslice = mol.aoslice_by_atom()
    nao = mol.nao_nr()
    hso_1e = np.zeros((3,nao,nao))
    for i in range(mol.natm):
        si, sf, ai, af = aoslice[i]
        slices = (si, sf, si, sf)
        mol.set_rinv_origin(mol.atom_coord(i))
        atom_1e = mol.intor('int1e_prinvxp', comp=3, shls_slice=slices)
        hso_1e[:,ai:af,ai:af] = - atom_1e * (mol.atom_charge(i))

    vj, vk = get_jk_amfi(mol, dm0)
    hso_2e = vj - vk * 1.5
    
    hso = (alpha2 / 4) * (hso_1e + hso_2e)
    return hso

def compute_hso(mol, dm0, amfi=True):  
    alpha2 = nist.ALPHA ** 2
    
    if amfi:
        hso = compute_hso_amfi(mol, dm0)
    
    else:
        hso_1e = mol.intor('int1e_prinvxp', comp=3)
        vj, vk = get_jk(mol, dm0)
        hso_2e = vj - vk * 1.5
        hso = (alpha2 / 4) * (hso_1e + hso_2e)
    return hso * 1j

