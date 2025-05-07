from pyscf import lo
import numpy as np
from DMET.my_pyscf.dmet.basistransformation import BasisTransform

# Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>

get_basis_transform = BasisTransform._get_basis_transformed

class Localization:
    '''
    Localization class for DMET.
    '''
    def __init__(self, mf, lo_method='meta-lowdin', ao2lo=None, **kwargs):
        '''
        Args:
            mf : SCF object
                SCF object for the molecule
            lo : str
                Localization method
            verbose : int
                Print level
        Returns:
            localized orbitals: np.array of shape N*N (ao*lo)
                Where N is the number of basis functions.
                These orbitals will be used to transform the ao to lo basis.
        '''
        self.mf = mf
        self.lo_method = lo_method
        self.ao2lo = ao2lo if ao2lo is not None else self.get_localized_orbitals()
            
    def get_localized_orbitals(self):
        '''
        Perform the localization/orthogonalization of the orbitals.
        '''
        mf = self.mf
        lo_method = self.lo_method
        ovlp = mf.get_ovlp()
        ao2lo = lo.orth_ao(mf, method=lo_method, s=ovlp)
        
        return ao2lo

    def localized_rdm1(self, ao2lo=None):
        '''
        Transform the dm from ao to lo basis.
        Args:
            ao2lo : np.array of shape N*N (ao*lo)
                Transformation matrix from ao to lo basis.
        Returns:
            dm_lo : np.array of shape N*N (lo*lo)
                Transformed dm in lo basis.
        '''
        if ao2lo is None:
            ao2lo = self.ao2lo

        dm = self.mf.make_rdm1()
        s = self.mf.get_ovlp()
        
        lo2ao = ao2lo.T @ s
        
        

        if dm.ndim > 2:
            dm_lo = np.asarray([get_basis_transform(dm_, lo2ao.T) for dm_ in dm])
            nelecs = np.trace(dm_lo[0]) + np.trace(dm_lo[1])
        else:
            dm_lo = get_basis_transform(dm, lo2ao.T)
            nelecs = np.trace(dm_lo)
        
        # Sanity Check
        assert (nelecs- self.mf.mol.nelectron) < 1e-7,\
        "Localization has some problem, nelectrons are not converged."

        return dm_lo