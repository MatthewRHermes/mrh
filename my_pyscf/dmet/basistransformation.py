import numpy as np
from functools import reduce
from pyscf import ao2mo

# Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>

class BasisTransform:
    '''
    Basis transformation class for DMET
    '''
    def __init__(self, mf, ao2eo, ao2co, **kwargs):
        '''
        Args:
            mf : SCF object
                SCF object for the molecule
            ao2eo : np.array
                Localized orbitals
            loc_rdm1 : np.array
                Localized 1-RDM
            mask_frag : list
                Fragment mask
            mask_env : list
                Environment mask
        '''
        self.mf = mf
        self.ao2eo = ao2eo
        self.ao2co = ao2co

    @staticmethod
    def _get_basis_transformed(operator, basis):
        '''
        Transform the operator to the new basis
        Args:
            operator : np.array
                Operator in the old basis
            basis : np.array
                Transformation matrix
        Returns:
            operator : np.array
                Operator in the new basis
        '''
        assert operator.ndim == 2, "Operator should be a 2D array"
        operator = reduce(np.dot, (basis.T, operator, basis))
        return operator

    def _get_fock_transformed(self, ao2eo=None):
        '''
        Fock matrix transformation
        Args:
            ao2eo : np.array nao * neo
                Transformation matrix from AO to EO
        Returns:
            fock : np.array neo * neo
                Transformed Fock matrix in AO basis
        '''
        if ao2eo is None:
            ao2eo = self.ao2eo
        
        fock = self.mf.get_fock()

        get_basis_transform = BasisTransform._get_basis_transformed

        if hasattr(fock, "focka"):
            focka = fock.focka
            fockb = fock.fockb
            fock_frag =  np.array([get_basis_transform(f, ao2eo) for f in (focka, fockb)])
           
        else:
            fock_frag = get_basis_transform(fock, ao2eo)
        
        return fock_frag

    def _get_cderi_transformed(self, mo):
        """
        Transforms CDERI integrals from AO to MO basis.
        Lpq---> Lij 
        Args:
           mo: np.array (nao*neo)
        Returns:
            Transformed CDERI integrals (Lij).
        """
        nmo = mo.shape[-1]
        mf = self.mf
        
        Lij = np.empty((mf.with_df.get_naoaux(), nmo * (nmo + 1) // 2), dtype=mo.dtype)
        
        ijmosym, mij_pair, moij, ijslice = ao2mo.incore._conc_mos(mo, mo, compact=True)
        b0 = 0
        for eri1 in mf.with_df.loop():
            b1 = b0 + eri1.shape[0]
            eri2 = Lij[b0:b1]
            eri2 = ao2mo._ao2mo.nr_e2(eri1, moij, ijslice, aosym='s2', mosym=ijmosym, out=eri2)
            b0 = b1
        return Lij

    def _get_eri_transformed(self, ao2eo=None):
        '''
        ERI transformation
        Args:
            ao2eo : np.array nao * neo
                Transformation matrix from AO to EO
        Returns:
            eri : 
                Transformed ERI in EO basis
        '''
        if ao2eo is None:
            ao2eo = self.ao2eo
        if hasattr(self.mf, 'with_df'):
            # This is condition, where for entire system, density_fitting is on, but
            # for the fragment, we want to use 4-index ERI for various reasons.
            eri = self.mf.with_df.ao2mo(ao2eo, compact=False)
        else:
            eri = ao2mo.general(self.mf._eri, [ao2eo, ao2eo, ao2eo, ao2eo], compact=False)
        return eri