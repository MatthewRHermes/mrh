import numpy as np
import scipy.linalg
from functools import reduce
from pyscf import lib
from pyscf.lib import cartesian_prod
from pyscf.pbc.tools import k2gamma

get_phase = k2gamma.get_phase
group_by_conj_pairs = k2gamma.group_by_conj_pairs

'''
Transform the k-point mo_coeff from the k-space to real-space.
'''

def _basis_transformation(mat, mo_coeff):
    return reduce(np.dot, (mo_coeff.conj().T, mat, mo_coeff))

def _check_orthonormality(mo_coeff, s):
    # ovlp = np.einsum('um,uv,vn->mn', mo_coeff.conj(), s, mo_coeff)
    ovlp = _basis_transformation(s, mo_coeff)
    return np.allclose(ovlp, np.eye(mo_coeff.shape[1]))

def get_real_space_trans(cell, kmesh):
    latt_vec = cell.lattice_vectors()
    rmesh = cartesian_prod((np.arange(kmesh[0]), 
                                np.arange(kmesh[1]),
                                np.arange(kmesh[2])))
    return np.einsum('nu, uv -> nv', rmesh, latt_vec)

def k2R(kmf, mo_coeff_k, kmesh, abs_kpts):
    '''
    nact_k: 
        number of active space orbitals at unit cell (i.e. at each k-point)
    nact: nk * nac_k
        total number of active space orbitals.
    nao_k: 
        number of atomic orbitals at unit cell (i.e. at each k-point)
    nao: nk * nao_k
        total number of atomic orbitals.
    Args:
        kmf : pbc.scf.KRHF or pbc.scf.KROHF
            k-point mean-field object.
        mo_coeff_k : list of np.ndarray [mo_active,]*nk
            Active space orbitals at each k-point.
    Returns:
        mo_coeff_R : np.ndarray [nao, nact]
            Active space orbitals in real-space.
    '''
    cell = kmf.cell
    mo_coeff_k = np.array(mo_coeff_k)
    r_abs_mesh = get_real_space_trans(cell, kmesh)
    Nk, Nao, Nmo = mo_coeff_k.shape
    NR = r_abs_mesh.shape[0]  
    phase = np.exp(1j*np.einsum('Ru, ku -> Rk', r_abs_mesh, abs_kpts))
    mo_act = np.einsum('Rk, kum -> Rukm', phase, mo_coeff_k) / np.sqrt(NR)
    mo_act = mo_act.reshape((NR, Nao, Nk*Nmo))
    return mo_act

def actmo_k2R(kmf, mo_coeff_guess, ncore, ncas, kpts, kmesh=None):
    '''
    '''
    cell = kmf.cell
    sk = kmf.get_ovlp(kpts=kpts)
    mo_coeff_k = [mo[:, ncore:ncore+ncas] for mo in mo_coeff_guess]
    mo_energy_k = np.hstack([kmf.mo_energy[k][ncore:ncore+ncas] for k in range(len(kpts))])

    # Sanity Check
    actshape = mo_coeff_k[0].shape
    assert all([mo.shape == actshape for mo in mo_coeff_k])

    # Phase
    scell, phase = get_phase(cell, kpts, kmesh)

    mo_coeff_k = np.asarray(mo_coeff_k)
    nk, nao, nact = mo_coeff_k.shape
    nR = phase.shape[0]
    
    # kphase
    k_conj_groups = group_by_conj_pairs(cell, kpts, return_kpts_pairs=False)
    k_phase = np.eye(nk, dtype=np.complex128)
    r2x2 = np.array([[1., 1j], [1., -1j]]) * .5**.5
    pairs = np.array([[k, k_conj] for k, k_conj in k_conj_groups
             if k_conj is not None and k != k_conj])
    
    for idx in pairs:
        k_phase[idx[:,None],idx] = r2x2

    # Transform AO indices
    mo_coeff = np.einsum('Rk,kum,kh->Ruhm', phase, mo_coeff_k, k_phase)
    mo_coeff = mo_coeff.reshape(nk*nao, nk*nact)

    # Pure imaginary orbitals to real
    cR_max = abs(mo_coeff.real).max(axis=0)
    mo_coeff[:,cR_max < 1e-5] *= -1j

    # TODO: Sort by mo by energy ?
    E_sort_idx = np.argsort(mo_energy_k, kind='stable')
    E_g = mo_energy_k[E_sort_idx]

    cI_max = abs(mo_coeff.imag).max(axis=0)
    if cI_max.max() < 1e-5:
        mo_coeff = mo_coeff.real[:,E_sort_idx]
    # Will need to fix this.
    else:
        mo_coeff = mo_coeff[:,E_sort_idx]
        s = scell.pbc_intor('int1e_ovlp')
        
        # For degenerated MOs, the transformed orbitals in super cell may not be
        # real. Construct a sub Fock matrix in super-cell to find a proper
        # transformation that makes the transformed MOs real.
        # MO energy degeneracy check
        E_k_degen = abs(E_g[1:] - E_g[:-1]) < 1e-3
        degen_mask = np.append(False, E_k_degen) | np.append(E_k_degen, False)
        degen_mask[cI_max < 1e-5] = False
        if np.any(E_k_degen):
            csc = reduce(np.dot, (mo_coeff.conj().T, s, mo_coeff))
            f = np.dot(csc * E_sort_idx, csc.conj().T)
            assert (abs(f.imag).max() < 1e-4)
            e, u = scipy.linalg.eigh(f.real, type='hermitian')
            mo_coeff = np.dot(mo_coeff, u)
        
        assert (abs(reduce(np.dot, (mo_coeff.conj().T, s, mo_coeff))
                    - np.eye(nR*nact)).max() < 1e-5)

    # Overlap between k-point unitcell and gamma-point supercell
    skg = np.einsum('kuv,Rk->kuRv', sk, phase.conj())
    skg = skg.reshape(nk,nao,nR*nao)
    cI_max = abs(mo_coeff.imag).max(axis=0)

    print('Max imaginary part in act mo (R-space): ', cI_max.max())
    mo_phase = np.einsum('kum,kuv,vi->kmi', mo_coeff_k.conj(), skg, mo_coeff)
    return scell, phase, mo_coeff, mo_phase

def actmo_R2k(kmf, mo_coeff_R, mo_phase, phase, kpts):
    '''
    Inverse of actmo_k2R
    '''
    nk = len(kpts)
    nR = phase.shape[0]
    nmo = mo_coeff_R.shape[1]
    nmo_k = nmo // nk
    sk = kmf.get_ovlp(kpts=kpts)
    nact_k = mo_phase.shape[1]
    nmo_k = mo_coeff_R.shape[0] // nR
    nact = nact_k * nk

    assert mo_coeff_R.shape[1] == nmo
    assert np.prod(mo_phase.shape[:2]) == nact
    mo_phase = mo_phase.reshape(-1, nact)
    
    # Do the transformation
    print(mo_coeff_R.shape, mo_phase.shape, phase.shape)
    mo_coeff_k_ = np.einsum('pq, qu -> pu', mo_coeff_R.conj().T, mo_phase).reshape(nR, nmo_k, nk, nact_k)
    mo_coeff_k = np.einsum('Rk,Rukm->kum', phase.conj(), mo_coeff_k_)

    # Sanity Check
    for k in range(nk):
        assert _check_orthonormality(mo_coeff_k[k], sk[k])

    return mo_coeff_k


def get_mo_coeff_k2R(kmf, mo_coeff_guess, ncore, ncas, kmesh=None):
    '''
    Get the k-point mo_coeff from the real-space mo_coeff_R
    '''
    cell = kmf.cell
    kpts = kmf.kpts
    sk = kmf.get_ovlp(kpts=kpts)
    mo_coeff_k = [mo[:, ncore:ncore+ncas] for mo in mo_coeff_guess]
    mo_energy_k = np.hstack([kmf.mo_energy[k][ncore:ncore+ncas] for k in range(len(kpts))])

    # Sanity Check
    actshape = mo_coeff_k[0].shape
    assert all([mo.shape == actshape for mo in mo_coeff_k])
    scell, phase = get_phase(cell, kpts, kmesh)

    E_g = np.hstack(mo_energy_k)
    C_k = np.asarray(mo_coeff_k)
    Nk, Nao, Nmo = C_k.shape
    NR = phase.shape[0]

    k_conj_groups = group_by_conj_pairs(cell, kpts, return_kpts_pairs=False)
    k_phase = np.eye(Nk, dtype=np.complex128)
    r2x2 = np.array([[1., 1j], [1., -1j]]) * .5**.5
    pairs = [[k, k_conj] for k, k_conj in k_conj_groups
             if k_conj is not None and k != k_conj]
    for idx in np.array(pairs):
        k_phase[idx[:,None],idx] = r2x2

    # complex supercell (real-space) MOs
    mo_coeff_R = np.einsum('Rk,kum,kh->Ruhm', phase, C_k, k_phase)
    mo_coeff_R = mo_coeff_R.reshape(Nao*NR, Nk*Nmo)

    # sort by energy
    E_sort_idx = np.argsort(E_g, kind='stable')
    E_g = E_g[E_sort_idx]
    mo_coeff_R = mo_coeff_R[:, E_sort_idx]

    # mo_phase can be computed with the (possibly complex) mo_coeff_R
    s_k = cell.pbc_intor('int1e_ovlp', kpts=kpts)
    s_k_g = np.einsum('kuv,Rk->kuRv', s_k, phase.conj()).reshape(Nk,Nao,NR*Nao)
    mo_phase = lib.einsum('kum,kuv,vi->kmi', C_k.conj(), s_k_g, mo_coeff_R)

    return scell, phase, mo_coeff_R, mo_phase

from pyscf.pbc.tools.k2gamma import to_supercell_ao_integrals

convert_kao_to_rao = to_supercell_ao_integrals