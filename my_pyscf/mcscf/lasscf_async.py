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
            First-order effective Fock matrix in the MO basis
        
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
        self.soo_coeff = self.s0 @ self.oo_coeff

        # Orthonormal AO basis for frag_atom
        # For now, I WANT these to overlap for different atoms. That is why I am pretending that
        # self.s0 is block-diagonal (so that <*|f>.(<f|f>^-1).<f|*> ~> P_f <f|f> P_f).
        ao_offset = self.mol.offset_ao_by_atom ()
        frag_orb = [orb for atom in frag_atom
                    for orb in list (range (ao_offset[atom,2], ao_offset[atom,3]))]
        self.nao_frag = len (frag_orb)
        ovlp_frag = self.s0[frag_orb,:][:,frag_orb] # <f|f> in the comment above
        proj_oo = self.oo_coeff[frag_orb,:] # P_f . oo_coeff in the comment above
        s1 = proj_oo.conj ().T @ ovlp_frag @ proj_oo
        w, u = linalg.eigh (-s1) # negative: sort from largest to smallest
        self.frag_umat = u[:,:self.nao_frag]

    def __call__(self, mo_coeff, dm1, veff, fock1):
        # TODO: gradient/hessian orbitals
        # TODO: how to handle active/active rotations

        # Everything in the orth-AO basis
        oo, soo = self.oo_coeff, self.soo_coeff
        mo = soo.conj ().T @ mo_coeff
        dm1 = soo.conj ().T @ dm1 @ soo
        veff = oo.conj ().T @ veff @ oo
        fock1 = mo @ fock1 @ mo.conj ().T

        fo, eo = self._get_orthnorm_frag (mo)
        fo, eo = self._a2i_gradorbs (fo, eo, fock1)
        nf = fo.shape[1]
        fo, eo = self._schmidt (fo, eo, mo)
        fo, eo = self._ia2x_gradorbs (fo, eo, mo, fock1)
        fo, eo = self._hessorbs (fo, eo, 2*nf, fock1)
        fo_coeff = self.oo_coeff @ fo
        return fo_coeff

    def _get_orthnorm_frag (self, mo):
        '''Get an orthonormal basis spanning the union of the frag_idth active space and
        ao_coeff, projected orthogonally to all other active subspaces.

        Args:
            mo : ndarray of shape (nmo,nmo)
                Contains MO coefficients in self.oo_coeff basis

        Returns:
            fo : ndarray of shape (nmo,*)
                Contains frag_idth active orbitals plus frag_orb approximately projected onto the
                inactive/external space in self.oo_coeff basis
            eo : ndarray of shape (nmo,*)
                Contains complementary part of the inactive/external space in self.oo_coeff basis
        '''
        # TODO: edge case for no active orbitals
        fo = mo[:,self.las0:self.las1]
        idx = np.ones (mo.shape[1], dtype=np.bool_)
        idx[self.ncore:self.nocc] = False
        uo = mo[:,idx]

        s1 = uo.conj ().T @ self.frag_umat
        u, svals, vh = linalg.svd (s1, full_matrices=True)
        idx = np.zeros (u.shape[1], dtype=np.bool_)
        idx[:len(svals)][np.abs (svals)>1e-8] = True
        fo = np.append (fo, uo @ u[:,idx], axis=-1)

        eo = uo @ u[:,~idx]

        return fo, eo

    def _a2i_gradorbs (self, fo, eo, fock1):
        '''Augment fragment-orbitals with environment orbitals coupled by the gradient to the
        active space

        Args:
            fo : ndarray of shape (nmo,*)
                Contains fragment-orbital coefficients in self.oo_coeff basis
            eo : ndarray of shape (nmo,*)
                Contains environment-orbital coefficients in self.oo_coeff basis
            fock : ndarray of shape (nmo,nmo)
                First-order effective Fock matrix in self.oo_coeff basis


        Returns:
            fo : ndarray of shape (nmo,*)
                Same as input, except with self.nlas additional gradient-coupled env orbs
            eo : ndarray of shape (nmo,*)
                Same as input, less the orbitals added to fo
        '''
        iGa = eo.conj ().T @ (fock1-fock1.T) @ fo[:,:self.nlas]
        u, svals, vh = linalg.svd (iGa, full_matrices=True)
        fo = np.append (fo, eo @ u[:,:self.nlas] @ vh[:self.nlas,:])
        eo = e0 @ u[:,self.nlas:]
        return fo, eo

    def _schmidt (self, fo, eo, mo):
        '''Do the Schmidt decomposition of the inactive determinant

        Args:
            fo : ndarray of shape (nao,*)
                Contains fragment-orbital coefficients in self.oo_coeff basis
            eo : ndarray of shape (nao,*)
                Contains environment-orbital coefficients in self.oo_coeff basis
            mo : ndarray of shape (nao,nmo)
                Contains MO coefficients in self.oo_coeff basis

        Returns:
            fbo : ndarray of shape (nao,*)
                Contains fragment and bath orbital coefficients
            ueo : ndarray of shape (nao,*)
                Contains unentangled inactive/external environment orbital coefficients
        '''
        nf = fo.shape[1] - self.nlas
        # TODO: edge case for eo.shape[1] < fo.shape[1]
        mo_core = mo[:,:self.ncore]
        dm_core = mo_core @ mo_core.conj ().T
        try:
            s1 = eo.conj ().T @ dm_core @ fo[:,self.nlas:]
        except IndexError as e:
            print (eo.shape, dm_core.shape, fo.shape)
            raise (e)
        u, svals, vh = linalg.svd (s1)
        svals = svals[:min(len(svals),nf)]
        idx = np.zeros (u.shape[1], dtype=np.bool_)
        idx[:len(svals)][np.abs(svals)>self.schmidt_thresh] = True
        eo = eo @ u
        fbo = np.append (fo, eo[:,idx], axis=-1)
        ueo = eo[:,~idx]
        print ("nf =",nf,"nb =",np.count_nonzero(idx),"nc =",np.count_nonzero(~idx))
        return fbo, ueo

    def _ia2x_gradorbs (self, mo, fo, eo, fock1):
        '''Replace virtual orbitals in fragment space with gradient-coupled virtual orbitals '''

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







