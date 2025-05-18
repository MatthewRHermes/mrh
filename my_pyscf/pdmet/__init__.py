import numpy as np
from functools import reduce
from pyscf import lo, lib
from pyscf.pbc import gto, scf, dft
from mrh.my_pyscf.pdmet._pdmet import _pDMET 

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

def get_fragment_mf(mf, lo_method='meta_lowdin', bath_tol=1e-4, 
                    atmlst=None, atmlabel=None, verbose=None,density_fit=False,**kwargs):
    get_fragment_mf.__doc__ = _get_dmet_fragment.__doc__
    
    mydmet = _pDMET(mf, lo_method=lo_method, bath_tol=bath_tol, atmlst=atmlst, atmlabel=atmlabel, verbose=verbose, density_fit=density_fit,**kwargs)
    dmet_mf = mydmet.kernel()
    # Contributions.
    _energy_contribution(mydmet, dmet_mf, mf.verbose)
    return dmet_mf, mydmet

def _get_dmet_fragment(mf, lo_method='meta_lowdin', bath_tol=1e-4, 
                        atmlst=None, atmlabel=None, density_fit=False,**kwargs):
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
    elif not hasattr(mf, 'kpts'):
        raise NotImplementedError("Use molecular DMET code")

    return get_fragment_mf(mf, lo_method, bath_tol, atmlst, atmlabel,density_fit=density_fit, **kwargs)

runpDMET = _get_dmet_fragment 


if __name__ == '__main__':
    from pyscf.pbc import gto, scf, df
    
    cell = gto.Cell(basis = 'gth-SZV',pseudo = 'gth-pade', a = np.eye(3) * 12, max_memory = 5000)
    cell.atom = '''
    N 0 0 0
    N 0 0 1.1
    '''
    cell.verbose = 4
    cell.build()

    # Integral generation
    gdf = df.GDF(cell)
    gdf._cderi_to_save = 'N2.h5'
    gdf.build()

    # SCF: Note: use the density fitting object to build the SCF object
    mf = scf.RHF(cell).density_fit()
    mf.exxdiv = None
    mf.with_df._cderi = 'N2.h5'
    mf.kernel()

    np.set_printoptions(precision=4)
    
    dmet_mf, ao2eo, ao2co = runpDMET(mf, lo_method='meta-lowdin', bath_tol=1e-10, atmlst=[0, 1])
    
    print("DMET:", dmet_mf.e_tot)
    print("Total Difference", mf.e_tot - dmet_mf.e_tot)
    assert abs((mf.e_tot - dmet_mf.e_tot)) < 1e-7, "Something went wrong."

    from pyscf import mcscf
    from pyscf.csf_fci import csf_solver

    mc = mcscf.CASSCF(dmet_mf,8,10)
    solver  = csf_solver(cell, smult=1)
    solver.nroots = 2
    mcscf.state_average_mix_(mc,[solver,], [0.5, 0.5])
    mc.kernel(dmet_mf.mo_coeff)


