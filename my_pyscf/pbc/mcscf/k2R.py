import numpy as np

from pyscf import lib
from pyscf.pbc.tools import k2gamma

# Author: Bhavnesh Jangid

'''
In this file there are some helper function to transform the k-point mo_coeff to wannier 
orbitals in real-space.
'''

get_phase = k2gamma.get_phase
group_by_conj_pairs = k2gamma.group_by_conj_pairs

def get_mo_coeff_k2R(kmf, mo_coeff_kpts, ncore, ncas, kmesh=None):
    '''
    Get the k-point mo_coeff from the real-space mo_coeff_R
    Reference: Also see pyscf/pbc/tools/k2gamma.py for the phase factor 
    and transformation details.
    args:
        kmf : pbc.scf.KRHF or pbc.scf.KROHF
            k-point mean-field object.
        mo_coeff_kpts : list of np.arrays or np.ndarray [(nao, nmo),]*Nk
            molecular orbitals at each k-point.
            nmo : number of molecular orbita at each k-point
        ncore : int
            number of core orbitals (This will be same for each k-point)
        ncas : int
            number of active orbitals (This will be same for each k-point)
        kmesh : list or tuple of 3 ints
            kmesh used in the k-CASSCF calculation. This is needed to get the phase
            factor for the transformation. If None, it will be inferred from the kpts.
            Note: for a given number of kpts, there can be multiple kmesh. that can lead to 
            different number NR than Nk. So it is better to provide the kmesh.
    returns:
        scell : pyscf.pbc.gto.Cell
            Supercell object
        phase : np.ndarray (NR, Nk)
            Phase factor for the transformation.
        mo_coeff_R : np.ndarray (nao*NR, ncas*Nk)
            Active space orbitals in wannier space.
        mo_phase : np.ndarray (Nk, ncas, ncas*Nk)
            Phase factors for the transformation.
    '''

    cell = kmf.cell
    kpts = kmf.kpts
    dtype = mo_coeff_kpts[0].dtype

    mo_coeff_k = np.array([mo[:, ncore:ncore+ncas] 
                           for mo in mo_coeff_kpts], dtype=dtype)
    
    mo_energy_k = np.hstack([kmf.mo_energy[k][ncore:ncore+ncas] 
                             for k in range(len(kpts))], dtype=dtype)

    E_g = np.hstack(mo_energy_k)
    C_k = np.asarray(mo_coeff_k)
    Nk, nao, nmo = C_k.shape

    scell, phase = get_phase(cell, kpts, kmesh)

    NR = phase.shape[0]

    assert Nk==NR, "Please use the kmesh in the k-CASSCF"
    k_conj_groups = group_by_conj_pairs(cell, kpts, return_kpts_pairs=False)
    k_phase = np.eye(Nk, dtype=dtype)
    r2x2 = np.array([[1., 1j], 
                     [1., -1j]]) * 0.5**0.5
    pairs = [[k, k_conj] for k, k_conj in k_conj_groups
             if k_conj is not None and k != k_conj]
    for idx in np.array(pairs):
        k_phase[idx[:,None],idx] = r2x2

    # Complex supercell (real-space) MOs
    # mo_coeff_R = np.einsum('Rk,kum,kS->RuSm', phase, C_k, k_phase)
    # mo_coeff_R = mo_coeff_R.reshape(nao*NR, NR*nmo)
    mo_coeff_R = np.dot(phase, (C_k[:, :, :, None] * k_phase[:, None, None, :]).reshape(Nk, -1)).reshape(NR, nao, nmo, NR)
    mo_coeff_R = mo_coeff_R.transpose(0, 1, 3, 2).reshape(NR*nao, NR*nmo)

    # sort by energy
    E_sort_idx = np.argsort(E_g, kind='stable')
    E_g = E_g[E_sort_idx]
    mo_coeff_R = mo_coeff_R[:, E_sort_idx]

    # mo_phase can be computed with the (possibly complex) mo_coeff_R
    s_k = kmf.get_ovlp(kpts=kpts)
    s_k_g = np.einsum('kuv,Rk->kuRv', s_k, phase.conj()).reshape(Nk,nao,NR*nao)
    # mo_phase = lib.einsum('kum,kuv,vi->kmi', C_k.conj(), s_k_g, mo_coeff_R)
    mo_phase = lib.einsum('kum,kui->kmi', C_k.conj(), np.dot(s_k_g, mo_coeff_R))
    return scell, phase, mo_coeff_R, mo_phase
