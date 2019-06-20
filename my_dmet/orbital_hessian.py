import numpy as np
from pyscf import ao2mo
from mrh.util.basis import represent_operator_in_basis, is_basis_orthonormal
from scipy import linalg

class HessianCalculator (object):
    ''' Calculate elements of an orbital-rotation Hessian corresponding to particular orbital ranges in a CASSCF or
    LASSCF wave function. This is not designed to be efficient in orbital optimization; it is designed to collect
    slices of the Hessian stored explicitly for some kind of spectral analysis. '''

    def __init__(self, mf, oneRDMs, twoCDM, ao2amo):
        ''' Args:
                mf: pyscf.scf.hf object (or at any rate, an object with the member attributes get_hcore, get_jk, and get_ovlp that
                    behaves like a pyscf scf object)
                    Must also have an orthonormal complete set of orbitals in mo_coeff (they don't have to be optimized in any way,
                    they just have to be complete and orthonormal)
                oneRDMs: ndarray or list of ndarray with overall shape (2,nao,nao)
                    spin-separated one-body density matrix in the AO basis.
                twoCDM: ndarray or list of ndarray with overall shape (*,ncas,ncas,ncas,ncas) 
                    two-body cumulant density matrix of or list of two-body cumulant density matrices for
                    the active orbital space(s). Has multiple elements in LASSCF (in general)
                ao2amo: ndarray of list of ndarray with overall shape (*,nao,ncas)
                    molecular orbital coefficients for the active space(s)
        '''
        self.scf = mf
        self.oneRDMs = np.asarray (oneRDMs)
        if self.oneRDMs.ndim == 2:
            self.oneRDMs /= 2
            self.oneRDMs = np.asarray ([self.oneRDMs, self.oneRDMs])

        mo = self.scf.mo_coeff
        moH = mo.conjugate ().T
        moHS = moH @ self.scf.get_ovlp ()
        self.mo = mo
        self.moH = moH 
        self.moHS = moHS
        self.nao, self.nmo = mo.shape

        if isinstance (twoCDM, (list,tuple,)):
            self.twoCDM = twoCDM
        elif twoCDM.ndim == 5:
            self.twoCDM = [t.copy () for t in twoCDM]
        else:
            self.twoCDM = [twoCDM]
        self.nas = len (self.twoCDM)
        self.ncas = [t.shape[0] for t in self.twoCDM]
 
        if isinstance (ao2amo, (list,tuple,)):
            self.mo2amo = ao2amo
        elif ao2amo.ndim == 3:
            self.mo2amo = [l for l in ao2amo]
        else:
            self.mo2amo = [ao2amo]
        self.mo2amo = [self.moHS @ ao2a for ao2a in mo2amo]
        assert (len (self.mo2amo) == self.nas) "Same number of mo2amo's and twoCDM's required"

        for t, a, n in zip (self.twoCDM, self.mo2amo, self.ncas):
            assert (t.shape == (n, n, n, n)), "twoCDM array size problem"
            assert (a.shape == (self.nmo, n)), "mo2amo array size problem"
            assert (is_basis_orthonormal (a)), 'problem putting active orbitals in orthonormal basis'       

        # Precalculate the fock matrix 
        vj, vk = self.scf.get_jk (dm=self.oneRDMs)
        fock = self.get_hcore + vj[0] + vj[1]
        self.fock = [fock - vk[0], fock - vk[1]]

        # Put 1rdm and fock in the orthonormal basis
        self.fock = [moH @ f @ mo for f in self.fock]
        self.oneRDMs = [moH @ D @ mo for D in self.oneRDMs]

    def __call__(self, p, q, r, s):
        ''' The Hessian E2^pr_qs is F2^pr_qs - F2^qr_ps - F2^ps_qr + F2^qs_pr. 
        Since the orbitals are segmented into separate ranges, you cannot necessarily just calculate
        one of these and transpose. '''
        # Put the orbital ranges in the orthonormal basis for fun and profit
        p = self.moHS @ p
        q = self.moHS @ q
        r = self.moHS @ r
        s = self.moHS @ s
        hess  = self._get (p, q, r, s)
        hess -= self._get (q, p, r, s)
        hess -= self._get (p, q, s, r)
        hess += self._get (q, p, s, r)
        return hess / 4

    def _get_eri (self, orbs_list, compact=False):
        ''' Get eris for the orbital ranges in orbs_list from (in order of preference) the stored _eri tensor on self.scf, the stored density-fitting object
        on self.scf, or on-the-fly using PySCF's ao2mo module '''
        if isinstance (orbs_list, np.ndarray) and orbs_list.ndim == 2:
            orbs_list = [orbs_list, orbs_list, orbs_list, orbs_list]
        # Tragically, I have to go back to the AO basis to interact with PySCF's eri modules. This is the greatest form of racism.
        orbs_list = [self.mo @ o for o in orbs_list]
        if self.scf._eri is not None:
            eri = ao2mo.incore.general (self.scf._eri, orbs_list, compact=compact) 
        elif self.with_df is not None:
            eri = self.with_df.ao2mo (orbs_list, compact=compact)
        else:
            eri = ao2mo.outcore.general_iofree (self.scf.mol, orbs_list, compact=compact)
        norbs = [o.shape[1] for o in orbs_list]
        if not compact: eri = eri.reshape (*norbs)
        return eri

    def _get_1perm (self, p, q, r, s):
        ''' This calculates one of the terms F2^pr_qs '''

        # Easiest term: 2 f^p_r D^q_s
        f_pr = [p.conjugate ().T @ f @ r for f in self.fock]
        D_qs = [q.conjugate ().T @ D @ s for D in self.oneRDMs]
        hess = 2 * sum ([np.multiply.outer (f, D) for f, D in zip (f_pr, D_qs)])
        hess = hess.transpose (0,2,1,3) # 'pr,qs->pqrs'

        # Generalized Fock matrix terms: delta_qr (F^p_s + F^s_p)
        ovlp_qr = q.conjugate ().T @ r
        if np.amax (np.abs (ovlp_qr)) > 1e-8: # skip if there is no overlap between the q and r ranges
            gf_ps = self._get_gfock (p, s)
            gf_ps += gf_ps.T
            hess += np.multiply.outer (ovlp_qr, gf_ps).transpose (2,0,1,3) # 'qr,ps->pqrs'

        # Explicit CDM contributions:  2 v^pu_rv l^qu_sv  +  2 v^pr_uv (l^qs_uv + l^qv_us)        
        for t, a, n in zip (self.twoCDM, self.mo2amo, self.ncas):
            a2q = a.conjugate ().T @ q
            a2s = a.conjugate ().T @ s
            # If either q or s has no weight on the current active space, skip
            if np.amax (np.abs (a2q)) < 1e-8 or np.amax (np.abs (a2s)) < 1e-8:
                continue
            eri = self._get_eri ([p, r, u, v])
            thess  = np.tensordot (eri, t, axes=((2,3),(2,3)))
            eri = self._get_eri ([p, v, r, u])
            thess += np.tensordot (eri, t + t.transpose (0,1,3,2), axes=((1,3),(1,3)))
            thess = np.tensordot (thess, a2q, axes=(2,0)) # 'prab,aq->prbq' (tensordot always puts output indices in order of the arguments)
            thess = np.tensordot (thess, a2s, axes=(2,0)) # 'prbq,bs->prqs'
            hess += 2 * thess.transpose (0, 2, 1, 3) # 'prqs->pqrs'

        # Weirdo split-coulomb and split-exchange terms
        hess += 4 * self._get_splitc (p, q, r, s, self.oneRDMs[0] + self.oneRDMs[1])
        for dm in self.oneRDMs:
            hess -= 2 * self._get_splitx (p, q, r, s, dm)

        return hess

    def _get_gfock (self, p, q):
        ''' Calculate the "generalized fock matrix" for orbital ranges p and q '''
        gfock = sum ([f @ D for f, D in zip (self.fock, self.oneRDMs)])
        gfock = p.conjugate ().T @ gfock @ q
        for t, a, n in zip (self.twoCDM, self.mo2amo, self.ncas):
            a2q = a.conjugate ().T @ q
            # If q has no weight on the current active space, skip
            if np.amax (np.abs (a2q)) < 1e-8:
                continue
            eri = self._get_eri ([p, a, a, a])
            gfock += np.tensordot (eri, t, axes=((1,2,3),(1,2,3))) @ a2q
        return gfock 

    def _get_splitc (self, p, q, r, s, dm)
        ''' v^pr_uv D^q_u D^s_v
        It shows up because some of the cumulant decompositions put q and s on different 1rdm factors '''
        u = self._get_entangled (q, dm)
        v = self._get_entangled (s, dm)
        if u.shape[1] == 0 or v.shape[1] == 0: return 0
        eri = self._get_eri ([p,u,r,v])
        D_uq = u.conjugate ().T @ dm @ q
        D_vs = v.conjugate ().T @ dm @ s
        hess = np.tensordot (eri,  D_uq, axes=(1,0)) # 'purv,uq->prvq'
        hess = np.tensordot (hess, D_vs, axes=(2,0)) # 'prvq,vs->prqs'
        return hess.transpose (0,2,1,3) # 'prqs->pqrs'

    def _get_splitx (self, p, q, r, s, dm)
        ''' (v^pv_ru + v^pr_vu) g^q_u g^s_v
        It shows up because some of the cumulant decompositions put q and s on different 1rdm factors 
        Pay VERY CLOSE ATTENTION to the order of the indices! Remember p-q, r-s are the degrees of freedom
        and the contractions should resemble an exchange diagram from mp2! (I've exploited elsewhere
        the fact that v^pv_ru = v^pu_rv because that's just complex conjugation, but I wrote it as v^pv_ru here
        to make the point because you cannot swap u and v in the other one.)'''
        u = self._get_entangled (q, dm)
        v = self._get_entangled (s, dm)
        if u.shape[1] == 0 or v.shape[1] == 0: return 0
        eri = self._get_eri ([p,r,v,u]) + self._get_eri ([p,v,r,u]).transpose (0,2,1,3)
        D_uq = u.conjugate ().T @ dm @ q
        D_vs = v.conjugate ().T @ dm @ s
        hess = np.tensordot (eri,  D_uq, axes=(3,0)) # 'prvu,uq->prvq'
        hess = np.tensordot (hess, D_vs, axes=(2,0)) # 'prvq,vs->prqs'
        return hess.transpose (0,2,1,3) # 'prqs->pqrs'

    def _get_entangled (self, p, dm)
        ''' Do SVD of a 1-rdm to get a small number of orbitals that you need to actually pay attention to
        when computing splitc and splitx eris '''
        q = linalg.qr (p)[0]
        qH = q.conjugate ().T
        lvec, sigma, rvec = linalg.svd (qH @ dm @ p)
        idx = np.abs (sigma) > 1e-8
        return q @ lvec[:,idx]


