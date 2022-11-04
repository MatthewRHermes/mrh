import numpy as np
from scipy import linalg
from pyscf import lib

class LASFragmenter (object):
    '''Construct an impurity subspace for a specific "fragment" of a LASSCF calculation defined
    as the union of a set of trial orbitals for a particular localized active subspace and the
    AO basis of a specified collection of atoms.

    Constructor args:
        las : object of :class:`LASCINoSymm`
            Mined for basic configuration info about the problem: `mole` object, _scf, ncore,
            ncas_sub, with_df, mo_coeff. The last is copied at construction and should be any
            othornormal basis (las.mo_coeff.conj ().T @ mol.get_ovlp () @ las.mo_coeff = a unit
            matrix)
        frag_id : integer or None
            identifies which active subspace is associated with this fragment
        frag_atom : list of integer
            identifies which atoms are identified with this fragment

    Calling args:
        mo_coeff : ndarray of shape (nao,nmo)
            Contains MO coefficients
        dm1 : ndarray of shape (nao,nao)
            State-averaged spin-summed 1-RDM in the AO basis
        veff : ndarray of shape (nao,nao)
            State-averaged spin-symmetric effective potential in the AO basis
        fock1 : ndarray of shape (nmo,nmo)
            First-order effective Fock matrix
        
    Returns:
        fo_coeff : ndarray of shape (nao, *)
            Orbitals defining an unentangled subspace containing the frag_idth set of active
            orbitals and the AOs of frag_atom

    '''

    def __init__(self, las, frag_id, frag_atom, schmidt_thresh=1e-8):
        self.mol = las.mol
        self._scf = las._scf
        self.ncore, self.ncas_sub, self.ncas = las.ncore, las.ncas_sub, las.ncas
        self.oo_coeff = las.mo_coeff.copy ()
        self.with_df = getattr (las, 'with_df', None)
        self.frag_id = frag_id
        self.frag_atom = frag_atom
        self.schmidt_thresh = schmidt_thresh

        # Convenience
        self.las0 = self.nlas = self.las1 = 0
        if frag_id is not None:
            self.las0 = self.ncore + sum(self.ncas_sub[:frag_id])
            self.nlas = self.ncas_sub[frag_id]
            self.las1 = self.las0 + self.nlas
        self.ncas = sum (self.ncas_sub)
        self.nocc = self.ncore + self.ncas
        self.s0 = self.mol.get_ovlp ()

        # Orthonormal AO basis for frag_atom
        ao_offset = self.mol.offset_ao_by_atom ()
        frag_orb = [orb for atom in frag_atom
                    for orb in list (range (ao_offset[atom,2], ao_offset[atom,3]))]
        self.nao_frag = len (frag_orb)
        s0 = self.s0[frag_orb,:][:,frag_orb]
        ao_coeff = self.oo_coeff[frag_orb,:]
        s1 = ao_coeff.conj ().T @ s0 @ ao_coeff
        w, u = linalg.eigh (-s1) # negative: sort from largest to smallest
        self.ao_coeff = self.oo_coeff @ u[:,:self.nao_frag]
        self.sao_coeff = self.s0 @ self.ao_coeff

    def __call__(self, mo_coeff, dm1, veff, fock1):
        # TODO: gradient/hessian orbitals
        # TODO: how to handle active/active rotations
        fo_coeff, eo_coeff = self._get_orthnorm_frag (mo_coeff)
        fo_coeff, eo_coeff = self._schmidt (mo_coeff, fo_coeff, eo_coeff)
        return fo_coeff

    def _get_orthnorm_frag (self, mo_coeff):
        '''Get an orthonormal basis spanning the union of the frag_idth active space and
        ao_coeff, projected orthogonally to all other active subspaces.

        Args:
            mo_coeff : ndarray of shape (nao,nmo)
                Contains MO coefficients

        Returns:
            fo_coeff : ndarray of shape (nao,*)
                Contains frag_idth active orbitals plus ao_coeff approximately projected onto the
                inactive/external space
            eo_coeff : ndarray of shape (nao,*)
                Contains complementary part of the inactive/external space
        '''
        # TODO: edge case for no active orbitals
        fo_coeff = mo_coeff[:,self.las0:self.las1]
        idx = np.ones (mo_coeff.shape[1], dtype=np.bool_)
        idx[self.ncore:self.nocc] = False
        mo_basis = mo_coeff[:,idx]

        s1 = mo_basis.conj ().T @ self.sao_coeff
        u, svals, vh = linalg.svd (s1, full_matrices=True)
        idx = np.zeros (u.shape[1], dtype=np.bool_)
        idx[:len(svals)][np.abs (svals)>1e-8] = True
        fo_coeff = np.append (fo_coeff, mo_basis @ u[:,idx], axis=-1)

        eo_coeff = mo_basis @ u[:,~idx]

        return fo_coeff, eo_coeff

    def _schmidt (self, mo_coeff, fo_coeff, eo_coeff):
        '''Do the Schmidt decomposition of the inactive determinant

        Args:
            mo_coeff : ndarray of shape (nao,nmo)
                Contains MO coefficients
            fo_coeff : ndarray of shape (nao,*)
                Contains fragment-orbital coefficients
            eo_coeff : ndarray of shape (nao,*)
                Contains environment-orbital coefficients

        Returns:
            fbo_coeff : ndarray of shape (nao,*)
                Contains fragment and bath orbital coefficients
            ueo_coeff : ndarray of shape (nao,*)
                Contains unentangled inactive/external environment orbital coefficients
        '''
        nf = fo_coeff.shape[1] - self.nlas
        # TODO: edge case for eo_coeff.shape[1] < fo_coeff.shape[1]
        mo_core = mo_coeff[:,:self.ncore]
        dm_core = mo_core @ mo_core.conj ().T
        try:
            s1 = eo_coeff.conj ().T @ self.s0 @ dm_core @ self.s0 @ fo_coeff[:,self.nlas:]
        except IndexError as e:
            print (eo_coeff.shape, self.s0.shape, dm_core.shape, fo_coeff.shape)
            raise (e)
        u, svals, vh = linalg.svd (s1)
        svals = svals[:min(len(svals),nf)]
        idx = np.zeros (u.shape[1], dtype=np.bool_)
        idx[:len(svals)][np.abs(svals)>self.schmidt_thresh] = True
        eo_coeff = eo_coeff @ u
        fbo_coeff = np.append (fo_coeff, eo_coeff[:,idx], axis=-1)
        ueo_coeff = eo_coeff[:,~idx]
        print ("nf =",nf,"nb =",np.count_nonzero(idx),"nc =",np.count_nonzero(~idx))
        return fbo_coeff, ueo_coeff


if __name__=='__main__':
    from mrh.tests.lasscf.c2h4n4_struct import structure as struct
    from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
    from pyscf import scf

    mol = struct (3.0, 3.0, '6-31g', symmetry=False)
    mol.verbose = lib.logger.INFO
    mol.output = __file__+'.log'
    mol.build ()

    mf = scf.RHF (mol).run ()
    mc = LASSCF (mf, (4,4), ((3,1),(1,3)), spin_sub=(3,3))
    frag_atom_list = (list (range (3)), list (range (7,10)))
    mo_coeff = mc.localize_init_guess (frag_atom_list, mf.mo_coeff)
    mc.kernel (mo_coeff)
    fo_coeff = LASFragmenter (mc, 0, frag_atom_list[0]) (mo_coeff, None, None, None) 

    from pyscf.tools import molden
    molden.from_mo (mol, __file__+'.molden', fo_coeff)







