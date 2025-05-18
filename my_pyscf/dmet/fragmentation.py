import numpy as np

# Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>

# TODO: Fragmentation based on orbindexes, atomlables.

class Fragmentation:
    '''
    Fragmentation class for DMET
    '''
    def __init__(self, mf, atmlst=None, atmlabel=None, **kwargs):
        '''
        Args:
            mf : SCF object
                SCF object for the molecule
            atmlst : list
                List of atom indices
            atmlabel : list
                List of atom labels
        '''
        assert atmlst is not None or atmlabel is not None,\
        "Fragmentation requires either atmlst or atmlabel"
        
        self.mf = mf
        self.mol = mf.mol
        self.fraginfo = atmlst 
        if atmlst is None and atmlabel is not None:
            raise NotImplementedError("Yet to be implemented")

    def _fragmentation_by_atmlist(self):
        '''
        Get the Fragmentation Information based on atom list,
        which is atom index in the XYZ file.
        Remember that the atom index starts from 0.
        Return:
            mask:  1D list of size nao
                containing the mask for the fragment orbitals in localized basis.
        '''
        mol = self.mol
        fraginfo = self.fraginfo
        nao = mol.nao

        aoslices = mol.aoslice_by_atom()[:,2:]
        frag_idx = [i for a in fraginfo for i in range(aoslices[a][0], aoslices[a][1])]

        mask = np.zeros(nao, dtype=bool)
        mask[frag_idx] = True
        return mask

    def _environment(self, mask):
        '''
        Get the mask for the environment
        Return:
            mask:  1D list of size nao
                containing the mask for the environment orbitals in localized basis.
        '''
        mask_env = ~mask
        return mask_env

    def get_fragments(self, atmlst=None, atmlabel=None):
        '''
        Get the mask for the fragments
        '''
        if atmlst is not None:
            mask_frag = self._fragmentation_by_atmlist()
            mask_env = self._environment(mask_frag)
        elif atmlabel is not None:
            mask_frag = self._fragmentation_by_atmlabels()
            mask_env = self._environment(mask_frag)
        return mask_frag, mask_env
