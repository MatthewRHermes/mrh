import numpy as np
from mrh.my_pyscf.dmet.basistransformation import BasisTransform

'''
This module contains the helper functions for DMET-PDFT.
Basically, it assembles the full space mo_coeffs from the embedded
CASSCF calculations.
'''

'''
Point to note: DMET-PDFT hasn't been tested for 
1. MultiState PDFT
2. Hybrid functionals
3. DMRG Solvers
'''

get_basis_transform = BasisTransform._get_basis_transformed

def assemble_mo(mf, ao2eo, ao2co, mc_mo_coeff):
    '''
    Assemble the mo_coeff to run the PDFT with the dmet_mf object.
    args:
        mf: RHF/ROHF object
            mean-field object for the full system
        ao2eo: np.array (nao, neo)
            transformation matrix from the full system to the embedded system. Note that
            nao: number of orbitals in the full system
            neo: number of orbitals in the embedded system
            ncore: number of core orbitals from the environment. (Don't get confuse with the ncore of mcscf)
            nao = neo + ncore
        ao2co: np.array (nao, ncore)
            transformation matrix from the full system to the core space
        mc_mo_coeff: np.array (neo, neo)
            mo_coeff for the embedded CASSCF calculation
    returns:
        mo_coeff: np.ndarray
            mo_coeff for the full system
    '''

    dm = mf.make_rdm1()
    s = mf.get_ovlp()
    
    cor2ao = ao2co.T @ s

    if dm.ndim > 2:
        dm = dm[0] + dm[1]
    
    # Generate the core density matrix and using that transform the ao2co
    # to the canonical basis.
    core_dm = get_basis_transform(dm, cor2ao.T)
    e, eigvec = np.linalg.eigh(core_dm)
    sorted_indices = np.argsort(e)[::-1]
    eigvec_sorted = eigvec[:, sorted_indices]  
    ao2co = ao2co @ eigvec_sorted
    core_nelec = int(round(np.sum(e[sorted_indices] > 1e-10)))
    assert core_nelec % 2 == 0, "Core nelec should be even., Something went wrong."
    ao2eo = ao2eo @ mc_mo_coeff

    ncore = core_nelec//2
    neo = ao2eo.shape[1]

    # Now we can assemble the full space mo_coeffs.
    mo_coeff = np.empty_like(mf.mo_coeff)
    mo_coeff[:, :ncore] = ao2co[:, :ncore]
    mo_coeff[:, ncore:ncore+neo] = ao2eo
    mo_coeff[:, ncore+neo:] = ao2co[:, ncore:]
    return mo_coeff

