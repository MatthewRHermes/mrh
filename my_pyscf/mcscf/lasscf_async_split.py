import numpy as np
from scipy import linalg
from pyscf import lib

class LASImpurityOrbitalCallable (object):
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
        dm1s : ndarray of shape (2,nao,nao)
            State-averaged spin-separated 1-RDM in the AO basis
        veff : ndarray of shape (2,nao,nao)
            State-averaged spin-separated effective potential in the AO basis
        fock1 : ndarray of shape (nmo,nmo)
            First-order effective Fock matrix in the MO basis
        
    Returns:
        fo_coeff : ndarray of shape (nao, *)
            Orbitals defining an unentangled subspace containing the frag_idth set of active
            orbitals and the AOs of frag_atom
        nelec_fo : 2-tuple of integers
            Number of electrons (spin-up, spin-down) in the impurity subspace
    '''

    def __init__(self, las, frag_id, frag_atom, schmidt_thresh=1e-8, nelec_int_thresh=1e-4):
        self.mol = las.mol
        self._scf = las._scf
        self.ncore, self.ncas_sub, self.ncas = las.ncore, las.ncas_sub, las.ncas
        self.oo_coeff = las.mo_coeff.copy ()
        self.with_df = getattr (las, 'with_df', None)
        self.frag_id = frag_id
        self.frag_atom = frag_atom
        self.schmidt_thresh = schmidt_thresh
        self.nelec_int_thresh = nelec_int_thresh

        # Convenience
        self.las0 = self.nlas = self.las1 = 0
        if frag_id is not None:
            self.las0 = self.ncore + sum(self.ncas_sub[:frag_id])
            self.nlas = self.ncas_sub[frag_id]
            self.las1 = self.las0 + self.nlas
        self.ncas = sum (self.ncas_sub)
        self.nocc = self.ncore + self.ncas
        self.s0 = self._scf.get_ovlp ()
        self.hcore = self.oo_coeff.conj ().T @ self._scf.get_hcore () @ self.oo_coeff
        self.soo_coeff = self.s0 @ self.oo_coeff
        self.log = lib.logger.new_logger (las, las.verbose)

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

    def __call__(self, mo_coeff, dm1s, veff, fock1):
        # TODO: how to handle active/active rotations
        self.log.info ("nmo = %d", mo_coeff.shape[1])

        # Everything in the orth-AO basis
        oo, soo = self.oo_coeff, self.soo_coeff
        mo = soo.conj ().T @ mo_coeff
        dm1s_ = [0,0]
        veff_ = [0,0]
        dm1s_[0] = soo.conj ().T @ dm1s[0] @ soo
        dm1s_[1] = soo.conj ().T @ dm1s[1] @ soo
        veff_[0] = oo.conj ().T @ veff[0] @ oo
        veff_[1] = oo.conj ().T @ veff[1] @ oo
        dm1s = dm1s_
        veff = veff_
        fock1 = mo @ fock1 @ mo.conj ().T

        fo, eo = self._get_orthnorm_frag (mo)
        self.log.info ("nfrag before gradorbs = %d", fo.shape[1])
        fo, eo, fock1 = self._a2i_gradorbs (fo, eo, fock1, veff, dm1s)
        nf = fo.shape[1]
        self.log.info ("nfrag after gradorbs 1 = %d", fo.shape[1])
        fo, eo = self._schmidt (fo, eo, mo)
        self.log.info ("nfrag after schmidt = %d", fo.shape[1])
        fo, eo = self._ia2x_gradorbs (fo, eo, mo, fock1, 2*nf)
        self.log.info ("nfrag after gradorbs 2 = %d", fo.shape[1])
        nelec_fo = self._get_nelec_fo (fo, dm1s)
        self.log.info ("nelec in fragment = %d, %d", nelec_fo[0], nelec_fo[1])
        fo_coeff = self.oo_coeff @ fo

        return fo_coeff, nelec_fo

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

    def _a2i_gradorbs (self, fo, eo, fock1, veff, dm1s):
        '''Augment fragment-orbitals with environment orbitals coupled by the gradient to the
        active space

        Args:
            fo : ndarray of shape (nmo,*)
                Contains fragment-orbital coefficients in self.oo_coeff basis
            eo : ndarray of shape (nmo,*)
                Contains environment-orbital coefficients in self.oo_coeff basis
            fock1 : ndarray of shape (nmo,nmo)
                First-order effective Fock matrix in self.oo_coeff basis
            veff : ndarray of shape (2,nmo,nmo)
                State-averaged spin-separated effective potential in self.oo_coeff basis
            dm1s : ndarray of shape (2,nmo,nmo)
                State-averaged spin-separated 1-RDM in self.oo_coeff basis

        Returns:
            fo : ndarray of shape (nmo,*)
                Same as input, except with self.nlas additional gradient-coupled env orbs
            eo : ndarray of shape (nmo,*)
                Same as input, less the orbitals added to fo
            fock1 : ndarray of shape (nmo,nmo)
                Same as input, after an approximate step towards optimizing the active orbitals
        '''
        iGa = eo.conj ().T @ (fock1-fock1.T) @ fo[:,:self.nlas]
        if not iGa.size: return fo, eo, fock1
        u, svals, vh = linalg.svd (iGa, full_matrices=True)
        fo = np.append (fo, eo @ u[:,:self.nlas] @ vh[:self.nlas,:], axis=1)
        eo = eo @ u[:,self.nlas:]
        mo = np.append (fo, eo, axis=1)

        # Get an estimated active-orbital relaxation step size
        ao, uo = fo[:,:self.nlas], fo[:,self.nlas:]
        uGa = uo.conj ().T @ (fock1-fock1.T) @ ao
        u, uGa, vh = linalg.svd (uGa, full_matrices=False)
        uo = uo @ u[:,:self.nlas]
        ao = ao @ vh[:self.nlas,:].conj ().T
        f0 = self.hcore[None,:,:] + veff
        f0_aa = (np.dot (f0, ao) * ao[None,:,:]).sum (1)
        f0_uu = (np.dot (f0, uo) * uo[None,:,:]).sum (1)
        f0_ua = (np.dot (f0, ao) * uo[None,:,:]).sum (1)
        dm1s_aa = (np.dot (dm1s, ao) * ao[None,:,:]).sum (1)
        dm1s_uu = (np.dot (dm1s, uo) * uo[None,:,:]).sum (1)
        dm1s_ua = (np.dot (dm1s, ao) * uo[None,:,:]).sum (1)
        uHa = ((f0_aa*dm1s_uu) + (f0_uu*dm1s_aa) - (2*f0_ua*dm1s_ua)).sum (0)
        uXa = (u * ((-uGa/uHa)[None,:])) @ vh # x = -b/A
        kappa1 = np.zeros ((mo.shape[1], mo.shape[1]), dtype=mo.dtype)
        kappa1[self.nlas:fo.shape[1],:self.nlas] = uXa
        kappa1 -= kappa1.T 
        kappa1 = mo @ kappa1 @ mo.conj ().T

        # approximate update to fock1
        tdm1 = -np.dot (dm1s, kappa1)
        tdm1 += tdm1.transpose (0,2,1)
        fock1 += f0[0]@tdm1[0] + f0[1]@tdm1[1]
        # TODO: missing Coulomb potential update?

        return fo, eo, fock1

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
        s1 = eo.conj ().T @ dm_core @ fo[:,self.nlas:]
        if not s1.size: return fo, eo
        u, svals, vh = linalg.svd (s1)
        svals = svals[:min(len(svals),nf)]
        idx = np.zeros (u.shape[1], dtype=np.bool_)
        idx[:len(svals)][np.abs(svals)>self.schmidt_thresh] = True
        eo = eo @ u
        fbo = np.append (fo, eo[:,idx], axis=-1)
        ueo = eo[:,~idx]
        return fbo, ueo

    def _ia2x_gradorbs (self, fo, eo, mo, fock1, ntarget):
        '''Augment fragment space with gradient/Hessian orbs

        Args:
            fo : ndarray of shape (nao,*)
                Contains fragment-orbital coefficients in self.oo_coeff basis
            eo : ndarray of shape (nao,*)
                Contains environment-orbital coefficients in self.oo_coeff basis
            mo : ndarray of shape (nao,nmo)
                Contains MO coefficients in self.oo_coeff basis
            fock1 : ndarray of shape (nmo,nmo)
                First-order effective Fock matrix in self.oo_coeff basis
            ntarget : integer
                Desired number of fragment orbitals when all is said and done

        Returns:
            fo : ndarray of shape (nmo,*)
                Same as input, except with additional gradient/Hessian-coupled env orbs
            eo : ndarray of shape (nmo,*)
                Same as input, less the orbitals added to fo
        '''

        # Split environment orbitals into inactive and external
        eSi = eo.conj ().T @ mo[:,:self.ncore]
        if not eSi.size: return fo, eo
        u, svals, vH = linalg.svd (eSi, full_matrices=True)
        ni = np.count_nonzero (svals>0.5)
        eo = eo @ u
        eio = eo[:,:ni]
        exo = eo[:,ni:]
 
        # Separate SVDs to avoid re-entangling fragment to environment
        svals_i = svals_x = np.zeros (0)
        if eio.shape[1]:
            eGf = eio.conj ().T @ (fock1-fock1.T) @ fo
            u_i, svals_i, vh = linalg.svd (eGf, full_matrices=True)
            eio = eio @ u_i
        if exo.shape[1]:
            eGf = exo.conj ().T @ (fock1-fock1.T) @ fo
            u_x, svals_x, vh = linalg.svd (eGf, full_matrices=True)
            exo = exo @ u_x
        eo = np.append (eio, exo, axis=1)
        svals = np.append (svals_i, svals_x)
        idx = np.argsort (-np.abs (svals))
        eo = eo[:,idx]
        
        # Augment fo
        nadd = min (u.shape[1], ntarget-fo.shape[1])
        fo = np.append (fo, eo[:,:nadd], axis=1)
        eo = eo[:,nadd:]

        return fo, eo

    def _get_nelec_fo (self, fo, dm1s):
        neleca = (dm1s[0] @ fo).ravel ().dot (fo.conj ().ravel ())
        neleca_err = neleca - int (round (neleca))
        nelecb = (dm1s[1] @ fo).ravel ().dot (fo.conj ().ravel ())
        nelecb_err = nelecb - int (round (nelecb))
        if any ([x>self.nelec_int_thresh for x in (neleca_err, nelecb_err)]):
            raise RuntimeError (
                "Non-integer number of electrons in impurity! (neleca,nelecb)={}".format (
                    (neleca,nelecb)))
        neleca = int (round (neleca))
        nelecb = int (round (nelecb))
        return neleca, nelecb

if __name__=='__main__':
    from mrh.tests.lasscf.c2h4n4_struct import structure as struct
    from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
    from pyscf import scf

    mol = struct (3.0, 3.0, 'cc-pvdz', symmetry=False)
    mol.verbose = lib.logger.INFO
    mol.output = __file__+'.log'
    mol.build ()

    mf = scf.RHF (mol).run ()
    mc = LASSCF (mf, (4,4), ((3,1),(1,3)), spin_sub=(3,3))
    frag_atom_list = (list (range (3)), list (range (7,10)))
    mo_coeff = mc.localize_init_guess (frag_atom_list, mf.mo_coeff)
    mc.max_cycle_macro = 1
    mc.kernel (mo_coeff)

    print ("Kernel done")
    ###########################
    from mrh.my_pyscf.mcscf.lasci import get_grad_orb
    dm1s = mc.make_rdm1s ()
    veff = mc.get_veff (dm1s=dm1s)
    fock1 = get_grad_orb (mc, hermi=0)
    ###########################
    get_imporbs_0 = LASImpurityOrbitalCallable (mc, 0, frag_atom_list[0])
    fo_coeff, nelec_fo = get_imporbs_0 (mc.mo_coeff, dm1s, veff, fock1)

    from pyscf.tools import molden
    molden.from_mo (mol, __file__+'.molden', fo_coeff)







