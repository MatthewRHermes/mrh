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

def assemble_mo(mf, trans_coeff, mc_mo_coeff):
    '''
    Assemble the mo_coeff
    args:
        mf: RHF/ROHF object
            mean-field object for the full system
        trans_coeff: dict
            transformation coefficients
        mc_mo_coeff: np.ndarray
            mo_coeff for the embedded CASSCF calculation
    returns:
        mo_coeff: np.ndarray
            mo_coeff for the full system
    '''

    dm = mf.make_rdm1()
    s = mf.get_ovlp()
    ao2co = trans_coeff['ao2co']
    ao2eo = trans_coeff['ao2eo']

    cor2ao = ao2co.T @ s

    if dm.ndim > 2:
        core_dm = [get_basis_transform(dm[0], cor2ao.T),get_basis_transform(dm[1], cor2ao.T)] 
        core_dm = core_dm[0] + core_dm[1]
    else:
        core_dm = get_basis_transform(dm, cor2ao.T)
    
    e, eigvec = np.linalg.eigh(core_dm)
    sorted_indices = np.argsort(e)[::-1]
    eigvec_sorted = eigvec[:, sorted_indices]  
    ao2co = ao2co @ eigvec_sorted

    ao2eo = ao2eo @ mc_mo_coeff
    core_nelec = int(round(trans_coeff['core_nelec']))
    ncore = core_nelec//2
    emb_size = ao2eo.shape[1]

    mo_coeff = np.empty_like(mf.mo_coeff)
    mo_coeff[:, :ncore] = ao2co[:, :ncore]
    mo_coeff[:, ncore:ncore+emb_size] = ao2eo
    mo_coeff[:, ncore+emb_size:] = ao2co[:, ncore:]
    return mo_coeff

