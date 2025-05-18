from functools import reduce
from pyscf import ao2mo, lib
import numpy as np
import h5py

class BasisTransform:
    '''
    Basis transformation class for DMET
    '''
    def __init__(self, kmf, ao2eo, ao2co, **kwargs):
        '''
        Args:
            kmf : SCF object in PBC
                SCF object for the cell
            ao2eo : np.array
                Localized orbitals
            loc_rdm1 : np.array
                Localized 1-RDM
            mask_frag : list
                Fragment mask
            mask_env : list
                Environment mask
        '''
        self.kmf = kmf
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

        kmf = self.kmf
        ao2eo = self.ao2eo
        
        fock = self.kmf.get_fock()
        _get_basis_transformed = self._get_basis_transformed
        if hasattr(fock, "focka"):
            focka = fock.focka
            fockb = fock.fockb
            fock_frag =  np.array([_get_basis_transformed(f, ao2eo) for f in (focka, fockb)])
        else:
            fock_frag = _get_basis_transformed(fock, ao2eo)
        
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
        assert mo.ndim == 2, "MO_coeff should be a 2D array"

        nmo = mo.shape[-1]
        kmf = self.kmf
        fsgdf = kmf.with_df._cderi
        neo = mo.shape[1]
        naux = kmf.with_df.get_naoaux ()
        neo_pair = neo * (neo + 1) // 2
        mem_eris = 8*(neo_pair*naux)/ 1e6
        mem_av = kmf.cell.max_memory - lib.current_memory ()[0]
        is_enough_mem = mem_av > 2. * mem_eris
        assert is_enough_mem, "Not enough memory for ERI transformation."
       
        Lij = np.empty((kmf.with_df.get_naoaux(), nmo * (nmo + 1) // 2), dtype=mo.dtype)
        
        ijmosym, mij_pair, moij, ijslice = ao2mo.incore._conc_mos(mo, mo, compact=True)
        b0 = 0
        for eri1 in kmf.with_df.loop():
            b1 = b0 + eri1.shape[0]
            eri2 = Lij[b0:b1]
            eri2 = ao2mo._ao2mo.nr_e2(eri1, moij, ijslice, aosym='s4', mosym=ijmosym, out=eri2)
            b0 = b1

        cderi_file = fsgdf.replace(".h5", "_df.h5") if fsgdf.endswith(".h5") else fsgdf + "_df"

        old_gdf = h5py.File(fsgdf, 'r')
        new_gdf = h5py.File(cderi_file, 'w')
        for key in old_gdf.keys():
            if key == 'j3c':
                new_gdf['j3c/0/0'] = Lij
            else:
                old_gdf.copy(old_gdf[key], new_gdf, key)
        old_gdf.close()
        new_gdf.close()
        return cderi_file


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
        kmf = self.kmf
        if ao2eo is None:
            ao2eo = self.ao2eo

        neo = ao2eo.shape[1]
        naux = kmf.with_df.get_naoaux ()
        neo_pair = neo * (neo + 1) // 2
        mem_eris = 8*(neo_pair*neo_pair)/ 1e6 
        mem_av = kmf.cell.max_memory - lib.current_memory ()[0]
        is_enough_mem = mem_av > 2. * mem_eris
        assert is_enough_mem, "Not enough memory for ERI transformation. \
        To solve the CAS problem in embedding space, 4-index integrals are kept in memory."
        eri = self.kmf.with_df.ao2mo(ao2eo)
        return eri
