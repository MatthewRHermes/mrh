import numpy as np
from functools import reduce
from pyscf import gto, scf, dft, lo, lib, mcscf
from pyscf.csf_fci import csf_solver
from mrh.my_pyscf.dmet._dmet import _DMET 

# Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>

def _energy_contribution(mydmet, dmet_mf, verbose=None):
    log = lib.logger.new_logger(mydmet, verbose)
    ao2eo = mydmet.ao2eo
    ao2co = mydmet.ao2co
    core_energy = mydmet._get_core_contribution(ao2eo, ao2co)
    log.info('DMET energy contribution:')
    log.info('Total Energy  %.7f', dmet_mf.e_tot)
    log.info('Emb. Energy   %.7f', dmet_mf.e_tot - core_energy)
    log.info('Core Energy   %.7f', core_energy)
    return None

def get_fragment_mf(mf, lo_method='meta_lowdin', bath_tol=1e-6, density_fit=True,
                    atmlst=None, atmlabel=None, verbose=None,**kwargs):
    get_fragment_mf.__doc__ = _get_dmet_fragment.__doc__
    mydmet = _DMET(mf, lo_method=lo_method, bath_tol=bath_tol, density_fit=density_fit, atmlst=atmlst, atmlabel=atmlabel, verbose=verbose, **kwargs)
    dmet_mf = mydmet.kernel()
    # Contributions.
    _energy_contribution(mydmet, dmet_mf, mf.verbose)
    return dmet_mf, mydmet

def _get_dmet_fragment(mf, lo_method='meta_lowdin', bath_tol=1e-6, density_fit=True,
                        atmlst=None, atmlabel=None, verbose=None,**kwargs):
    '''
    Get the DMET Mean Field object
    Args:
        mf : SCF object
            SCF object for the molecule
        lo_method : str
            Localization method
        bath_tol : float
            Bath tolerance
        atmlst : list
            List of atom indices
        atmlabel : list
            List of atom labels
        verbose : int
            Print level
    Returns:
        total_energy : float
            Total energy
        core_energy : float
            core energy
        dmet_mf : SCF object
            DMET mean-field object
        ao2eo : np.array
            Transformation matrix from AO to EO
        ao2co : np.array
            Core coefficient matrix
    '''

    # If the provided mean-field is UHF or DFT, convert it to RHF.
    # Also, if it periodic kmf, convert it to RHF.

    if isinstance(mf, dft.rks.KohnShamDFT) or isinstance(mf, scf.uhf.UHF):
        mf = mf.to_rhf()
    elif hasattr(mf, 'kpts'):
        raise NotImplementedError("Use pDMET code")

    return get_fragment_mf(mf, lo_method, bath_tol, density_fit, atmlst, atmlabel, verbose, **kwargs)

runDMET = _get_dmet_fragment
if __name__ == '__main__':
    mol = gto.Mole(basis='6-31G', spin=1, charge=0, verbose=4, max_memory = 10000)
    mol.atom='''
    P  -5.64983   3.02383   0.00000
    H  -4.46871   3.02383   0.00000
    H  -6.24038   2.19489   0.59928
    '''
    mol.build()

    np.set_printoptions(precision=4)

    mf = scf.ROHF(mol).density_fit()
    mf.kernel()
    
    dmet_mf, trans_coeff = _get_dmet_fragment(mf, lo_method='lowdin', bath_tol=1e-10, atmlst=[0, ])
    
    assert abs((mf.e_tot - dmet_mf.e_tot)) < 1e-7, "Something went wrong."
    
    ao2eo = trans_coeff['ao2eo']
    orblst = getorbindex(mol, ao2eo, lo_method='meta-lowdin',
                     ao_label=['P 3s', 'P 3p', 'H 1s'], activespacesize=6, s=mf.get_ovlp())
    
    mc = mcscf.CASSCF(dmet_mf, 6, 7)
    mo = mc.sort_mo(orblst)
    mc.fcisolver  = csf_solver(mol, smult=2)
    mc.kernel(mo)


