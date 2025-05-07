from pyscf import lo
import numpy as np
from DMET.my_pyscf.dmet import basistransformation as bt

# Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>

_get_basis_transformed = bt.BasisTransform._get_basis_transformed

class Localization:
    '''
    Localization class for DMET
    '''
    def __init__(self, kmf, lo_method='meta-lowdin', ao2lo=None, **kwargs):
        '''
        Args:
            kmf : SCF object in PBC
                SCF object for the cell
            lo : str
                Orthonormalization/Localization method
            verbose : int
                Print level
        Returns:
            localized orbitals: np.array
                n*n; where n is the number of basis functions
        '''
        self.kmf = kmf
        self.lo_method = lo_method
        self.ao2lo = ao2lo if ao2lo is not None else self.get_localized_orbitals()
            
    def get_localized_orbitals(self):
        '''
        Get the localized orbitals
        '''
        kmf = self.kmf
        lo_method = self.lo_method
        cell = kmf.cell
        ovlp = kmf.get_ovlp()

        ao2lo = lo.orth_ao(kmf, method=lo_method, s=ovlp)

        return ao2lo

    def localized_rdm1(self, ao2lo=None):
        '''
        Get the localized 1-RDM
        '''
        if ao2lo is None:
            ao2lo = self.ao2lo

        kmf = self.kmf
        dm = kmf.make_rdm1()
        s = kmf.get_ovlp()
       
        lo2ao = ao2lo.T @ s

        
        
        if dm.ndim > 2:
            dm_lo = np.asarray([_get_basis_transformed(dm_, lo2ao.T) for dm_ in dm])
            nelecs = np.trace(dm_lo[0]) + np.trace(dm_lo[1])
        else:
            dm_lo = _get_basis_transformed(dm, lo2ao.T)
            nelecs = np.trace(dm_lo)
       
        # Sanity Check
        assert (nelecs- kmf.cell.nelectron) < 1e-7,\
        "Localization has some problem, nelectrons are not converged."

        return dm_lo