import time
import numpy as np
from pyscf import ao2mo
from pyscf.mcscf.mc1step import gen_g_hop
from mrh.util.basis import represent_operator_in_basis, is_basis_orthonormal, measure_basis_olap, orthonormalize_a_basis, get_complementary_states
from mrh.util.rdm import get_2CDM_from_2RDM
from scipy import linalg
from itertools import product

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
        self._eri_kernel = self._eri_kernel_basis = None 

        mo = self.scf.mo_coeff
        Smo = self.scf.get_ovlp () @ mo
        moH = mo.conjugate ().T
        moHS = moH @ self.scf.get_ovlp ()
        self.mo = mo
        self.Smo = Smo 
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
        self.mo2amo = [self.moHS @ ao2a for ao2a in self.mo2amo]
        assert (len (self.mo2amo) == self.nas), "Same number of mo2amo's and twoCDM's required"

        for t, a, n in zip (self.twoCDM, self.mo2amo, self.ncas):
            assert (t.shape == (n, n, n, n)), "twoCDM array size problem"
            assert (a.shape == (self.nmo, n)), "mo2amo array size problem"
            assert (is_basis_orthonormal (a)), 'problem putting active orbitals in orthonormal basis'       

        # Precalculate the fock matrix 
        vj, vk = self.scf.get_jk (dm=self.oneRDMs)
        fock = self.scf.get_hcore () + vj[0] + vj[1]
        self.fock = [fock - vk[0], fock - vk[1]]

        # Put 1rdm and fock in the orthonormal basis
        self.fock = [moH @ f @ mo for f in self.fock]
        self.oneRDMs = [moHS @ D @ Smo for D in self.oneRDMs]

    def __call__(self, *args, **kwargs):
        diagx1 = kwargs['diagx1'] if 'diagx1' in kwargs else False
        diagx2 = kwargs['diagx2'] if 'diagx2' in kwargs else False
        if 'diagx' in kwargs: diagx1 = diagx2 = kwargs['diagx']
        if len (args) == 0:
            ''' If no orbital ranges are passed, return the full mo-basis Hessian '''
            return self._call_fullrange (self.mo)
        elif len (args) == 1:
            ''' If one orbital range is passed, return the Hessian with all four indices in that range'''
            return self._call_fullrange (args[0])
        elif len (args) == 2:
            ''' If two orbital ranges are passed, I assume that you are asking ONLY for the diagonal elements'''
            return self._call_diag (args[0], args[1], diagx=diagx1)
        elif len (args) == 3:
            ''' No interpretation; raise an error '''
            raise RuntimeError ("Can't interpret 3 orbital ranges; pass 0, 1, 2, or 4")
        elif len (args) == 4:
            ''' If all four orbital ranges are passed, return the Hessian so specified. No permutation symmetry can be exploited. '''
            return self._call_general (args[0], args[1], args[2], args[3], diagx1=diagx1, diagx2=diagx2)
        else:
            raise RuntimeError ("Orbital Hessian has 4 orbital indices; you passed {} orbital ranges".format (len (args)))

    def _call_fullrange (self, p):
        '''Use full permutation symmetry to accelerate it!'''
        norb = [p.shape[-1], p.shape[-1], p.shape[-1], p.shape[-1]]
        if 0 in norb: return np.zeros (norb)
        # Put the orbital ranges in the orthonormal basis for fun and profit
        p = self.moHS @ p
        hess = self._get_Fock2 (p, p, p, p)
        hess -= hess.transpose (1, 0, 2, 3)
        hess -= hess.transpose (0, 1, 3, 2)
        return hess / 2

    def _call_diag (self, p, q, diagx=False):
        hess = np.zeros ([p.shape[-1], q.shape[-1]])
        if diagx: assert (p.shape[-1] == q.shape[-1]), 'diagx error: p and q ranges have different lengths'
        for ix, iy in product (range (p.shape[-1]), range (q.shape[-1])):
            if diagx and ix != iy: continue
            hess[ix,iy] = self._call_general (p[:,ix:ix+1], q[:,iy:iy+1], p[:,ix:ix+1], q[:,iy:iy+1], diagx1=False, diagx2=False)[0,0,0,0]
        if diagx: hess = np.diag (hess)
        return hess

    def _call_general (self, p, q, r, s, diagx1=False, diagx2=False):
        ''' The Hessian E2^pr_qs is F2^pr_qs - F2^qr_ps - F2^ps_qr + F2^qs_pr. 
        Since the orbitals are segmented into separate ranges, you cannot necessarily just calculate
        one of these and transpose. '''
        norb = [p.shape[-1], q.shape[-1], r.shape[-1], s.shape[-1]]
        if 0 in norb: return np.zeros (norb)
        # diagx1 recursion
        if diagx1:
            assert (p.shape[-1] == q.shape[-1]), 'diagx error: p and q ranges have different lengths'
            hess = np.zeros ([p.shape[-1], r.shape[-1]]) if diagx2 else np.zeros ([p.shape[-1], r.shape[-1], s.shape[-1]]) 
            for ix in range (p.shape[-1]):
                hess[ix] = self._call_general (p[:,ix:ix+1], q[:,ix:ix+1], r, s, diagx1=False, diagx2=diagx2)[0,0]
            return hess
        # diagx2 recursion
        if diagx2:
            assert (r.shape[-1] == s.shape[-1]), 'diagx error: r and s ranges have different lengths'
            hess = np.zeros ([p.shape[-1], q.shape[-1], r.shape[-1]]) 
            for ix in range (r.shape[-1]):
                hess[:,:,ix] = self._call_general (p, q, r[:,ix:ix+1], s[:,ix:ix+1], diagx1=False, diagx2=False)[:,:,0,0]
            return hess
        # Put the orbital ranges in the orthonormal basis for fun and profit
        p = self.moHS @ p
        q = self.moHS @ q
        r = self.moHS @ r
        s = self.moHS @ s
        t0, w0 = time.clock (), time.time ()
        self._cache_eri_baaa (p, q, r, s)
        self._cache_eri_pqrs (p, q, r, s)
        print ("Time spent in preliminary cacheing: {:.6f} s clock, {:.6f} s wall".format (time.clock () - t0, time.time () - w0))
        hess  = self._get_Fock2 (p, q, r, s)
        hess -= self._get_Fock2 (q, p, r, s).transpose (1,0,2,3)
        hess -= self._get_Fock2 (p, q, s, r).transpose (0,1,3,2)
        hess += self._get_Fock2 (q, p, s, r).transpose (1,0,3,2)
        return hess / 2

    def _get_eri (self, orbs_list, compact=False):
        ''' First, see if I have a reduced-range array in an orthonormal basis available on this object.
            Only if I don't will I leave this object to look for eris. '''
        if self._eri_kernel is not None and self._eri_kernel_basis is not None:
            b2c = self._eri_kernel_basis.conjugate ().T
            new_orbs_list = [b2c @ c2p for c2p in orbs_list]
            eri = ao2mo.incore.general (self._eri_kernel, new_orbs_list, compact=compact)
            norbs = [o.shape[1] for o in orbs_list]
            if not compact: eri = eri.reshape (*norbs)
            return eri
        return self._get_eri_external (orbs_list, compact=compact)

    def _get_eri_external (self, orbs_list, compact=False):
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

    def _get_collective_basis (self, *args):
        p = np.concatenate (args, axis=-1)
        q = orthonormalize_a_basis (p)
        '''
        ovlp = p.conjugate ().T @ p
        evals, evecs = 
        p2q = linalg.qr (ovlp)[0]
        q = p @ p2q
        '''
        qH = q.conjugate ().T
        q2p = [qH @ arg for arg in args]
        return q, q2p

    def _get_uv (self, q, s):
        oneRDM = sum (self.oneRDMs)
        dms = self.oneRDMs + [oneRDM]
        u = [self._get_entangled (q, dm) for dm in dms]
        u = self._get_collective_basis (*u)[0]
        v = [self._get_entangled (s, dm) for dm in dms]
        v = self._get_collective_basis (*v)[0]
        return u, v

    def _cache_overlapping_inactive (self, *args):
        b = self._get_collective_basis (*args)[0]
        a = np.concatenate (self.mo2amo, axis=1)
        b2a = b.conjugate ().T @ a
        b2u = get_complementary_states (b2a, already_complete_warning=False)
        u = b @ b2u
        dm = sum (self.oneRDMs) # inactive, therefore spin-adapted
        evals, u2i = linalg.eigh (u.conjugate ().T @ dm @ u)
        idx = evals > 1e-8
        self.pqrs_i = u @ u2i
        return

    def _cache_eri_baaa (self, *args):
        b = self.pqrs_b = self._get_collective_basis (*args)[0]
        self.baaa = []
        for t, a, n in zip (self.twoCDM, self.mo2amo, self.ncas):
            self.baaa.append (self._get_eri ([b, a, a, a]))
        return

    def _cache_eri_bboo (self, p, q, r, s):
        self._cache_overlapping_inactive (self, p, q, r, s)
        b1 = self.pq_b = self._get_collective_basis (p, q)[0]
        b2 = self.rs_b = self._get_collective_basis (r, s)[0]
        ia_o = [self.pqrs_i] + self.mo2amo
        self.bboo = []
        self.bobo = []
        # It's important to understand why the combinatoric logic of the two below is different!
        # bboo has an inherent transpose symmetry: pqij = pqji
        # On the other hand, bobo does NOT
        for o1, o2 in combinations_with_replacement (ia_o, 2):
            self.bboo.append (self._get_eri ([b1, b2, o1, o2]))
        for o1, o2 in product (ia_o, repeat=2):
            self.bobo.append (self._get_eri ([b1, o1, b2, o2]))
        # I should make an index to conveniently address the diagonal active-orbital parts only for the 2CDM terms
        diag_idx = np.cumsum (np.arange (self.nas + 1, 1, -1, dtype=np.int))
        self.bbaa = [self.bboo[idx] for idx in diag_idx]
        diag_idx = (np.arange (self.nas + 1, dtype=np.int) * (nas + 2))[1:]
        self.baba = [self.bobo[idx] for idx in diag_idx]
        return

    def _cache_eri_pruv (self, p, r, u, v):
        if v.shape[1] == 0 or u.shape[1] == 0: return
        self.purv = self._get_eri ([p, u, r, v])
        self.prvu = self._get_eri ([p, r, v, u])
        self.pvru = self._get_eri ([p, v, r, u])
        return

    def _cache_eri_pqrs (self, p, q, r, s):
        ''' Cache (pq|rs) and (ps|rq) including all orbitals entangled to p,q,r,s in those ranges via any 1-rdm
            (charge, alpha, or beta) '''
        pu, qu, ru, su = (self._append_entangled (x) for x in [p,q,r,s])
        self.pqrs = self._get_eri ([pu,qu,ru,su])
        self.psrq = self._get_eri ([pu,su,ru,qu])
        self.prsq = self._get_eri ([pu,ru,su,qu])
        self.pqrs_p = p
        self.pqrs_r = r
        self.pqrs_up = pu
        self.pqrs_uq = qu
        self.pqrs_ur = ru
        self.pqrs_us = su
        return

    def _check_pq_perm (self, p, q):
        olap_p = measure_basis_olap (p, self.pqrs_p)[0] / p.shape[1]
        olap_q = measure_basis_olap (q, self.pqrs_p)[0] / q.shape[1]
        return olap_q > olap_p

    def _check_rs_perm (self, r, s):
        olap_r = measure_basis_olap (r, self.pqrs_r)[0] / r.shape[1]
        olap_s = measure_basis_olap (s, self.pqrs_r)[0] / s.shape[1]
        return olap_s > olap_r

    def _get_Fock2 (self, p, q, r, s):
        ''' This calculates one of the terms F2^pr_qs '''

        # Easiest term: 2 f^p_r D^q_s
        f_pr = [p.conjugate ().T @ f @ r for f in self.fock]
        D_qs = [q.conjugate ().T @ D @ s for D in self.oneRDMs]
        hess = 2 * sum ([np.multiply.outer (f, D) for f, D in zip (f_pr, D_qs)])
        hess = hess.transpose (0,2,1,3) # 'pr,qs->pqrs'

        # Generalized Fock matrix terms: delta_qr (F^p_s + F^s_p)
        ovlp_qr = q.conjugate ().T @ r
        if np.amax (np.abs (ovlp_qr)) > 1e-8: # skip if there is no overlap between the q and r ranges
            gf_ps = self._get_Fock1 (p, s) + self._get_Fock1 (s, p).T
            hess += np.multiply.outer (ovlp_qr, gf_ps).transpose (2,0,1,3) # 'qr,ps->pqrs'

        # Explicit CDM contributions:  2 v^pu_rv l^qu_sv  +  2 v^pr_uv (l^qs_uv + l^qv_us)        
        t0, w0 = time.clock (), time.time ()
        for t, a, n in zip (self.twoCDM, self.mo2amo, self.ncas):
            #for t, a, n, bbaa, baba in zip (self.twoCDM, self.mo2amo, self.ncas, self.bbaa, self.baba):
            a2q = a.conjugate ().T @ q
            a2s = a.conjugate ().T @ s
            # If either q or s has no weight on the current active space, skip
            if np.amax (np.abs (a2q)) < 1e-8 or np.amax (np.abs (a2s)) < 1e-8:
                continue
            eri = self._get_eri ([p, r, a, a])
            thess  = np.tensordot (eri, t, axes=((2,3),(2,3)))
            eri = self._get_eri ([p, a, r, a])
            thess += np.tensordot (eri, t + t.transpose (0,1,3,2), axes=((1,3),(1,3)))
            '''
            p2b = p.conjugate ().T @ self.pq_b
            r2b = r.conjugate ().T @ self.rs_b
            braa = np.tensordot (r2b, bbaa, axes=((1),(1))).transpose (1,0,2,3)
            praa = np.tensordot (p2b, braa, axes=1)
            thess  = np.tensordot (praa, t, axes=((2,3),(2,3)))
            bara = np.tensordot (r2b, baba, axes=((1),(2))).transpose (1,2,0,3)
            para = np.tensordot (p2b, bara, axes=1)
            thess += np.tensordot (para, t + t.transpose (0,1,3,2), axes=((1,3),(1,3)))
            '''
            thess = np.tensordot (thess, a2q, axes=(2,0)) # 'prab,aq->prbq' (tensordot always puts output indices in order of the arguments)
            thess = np.tensordot (thess, a2s, axes=(2,0)) # 'prbq,bs->prqs'
            hess += 2 * thess.transpose (0, 2, 1, 3) # 'prqs->pqrs'
        print ("Time spent in 2-CDM terms: {:.6f} s clock, {:.6f} s wall".format (time.clock () - t0, time.time () - w0))

        # Weirdo split-coulomb and split-exchange terms
        # u,v are supersets of q,s 
        perm_pq = self._check_pq_perm (p, q)
        perm_rs = self._check_rs_perm (r, s)
        u = self.pqrs_up if perm_pq else self.pqrs_uq
        v = self.pqrs_ur if perm_rs else self.pqrs_us
        #self._cache_eri_pruv (p, r, u, v)
        hess += 4 * self._get_splitc (p, q, r, s, sum (self.oneRDMs), u, v, perm_pq, perm_rs)
        for dm in self.oneRDMs:
            hess -= 2 * self._get_splitx (p, q, r, s, dm, u, v, perm_pq, perm_rs)
        return hess

    def _get_Fock1 (self, p, q):
        ''' Calculate the "generalized fock matrix" for orbital ranges p and q '''
        gfock = sum ([f @ D for f, D in zip (self.fock, self.oneRDMs)])
        gfock = p.conjugate ().T @ gfock @ q
        p2b = p.conjugate ().T @ self.pqrs_b
        # b must span all of p at least, or else there is a bug
        b2p = p2b.conjugate ().T
        pOp = p2b @ b2p
        for t, a, n, baaa in zip (self.twoCDM, self.mo2amo, self.ncas, self.baaa):
            a2q = a.conjugate ().T @ q
            # If q has no weight on the current active space, skip
            if np.amax (np.abs (a2q)) < 1e-8:
                continue
            eri = np.tensordot (p2b, baaa, axes=1)
            gfock += np.tensordot (eri, t, axes=((1,2,3),(1,2,3))) @ a2q
        return gfock 

    def _get_splitc (self, p, q, r, s, dm, u, v, perm_pq, perm_rs):
        ''' v^pr_uv D^q_u D^s_v
        It shows up because some of the cumulant decompositions put q and s on different 1rdm factors '''
        if u.shape[1] == 0 or v.shape[1] == 0: return 0
        D_uq = u.conjugate ().T @ dm @ q
        D_vs = v.conjugate ().T @ dm @ s
        lp, lr = p.shape[1], r.shape[1]
        pqrs = self.pqrs
        if perm_pq: pqrs = pqrs.transpose (1,0,2,3)
        if perm_rs: pqrs = pqrs.transpose (0,1,3,2)
        purv = pqrs[:lp,:,:,:][:,:,:lr,:]
        hess = np.tensordot (purv, D_uq, axes=(1,0)) # 'purv,uq->prvq'
        hess = np.tensordot (hess, D_vs, axes=(2,0)) # 'prvq,vs->prqs'
        return hess.transpose (0,2,1,3) # 'prqs->pqrs'

    def _get_splitx (self, p, q, r, s, dm, u, v, perm_pq, perm_rs):
        ''' (v^pv_ru + v^pr_vu) g^q_u g^s_v = v^pr_vu g^q_u g^s_v - v^pv_su g^q_u g^r_v
        It shows up because some of the cumulant decompositions put q and s on different 1rdm factors 
        Pay VERY CLOSE ATTENTION to the order of the indices! Remember p-q, r-s are the degrees of freedom
        and the contractions should resemble an exchange diagram from mp2! (I've exploited elsewhere
        the fact that v^pv_ru = v^pu_rv because that's just complex conjugation, but I wrote it as v^pv_ru here
        to make the point because you cannot swap u and v in the other one.)'''
        if u.shape[1] == 0 or v.shape[1] == 0: return 0
        D_uq = u.conjugate ().T @ dm @ q
        D_vs = v.conjugate ().T @ dm @ s
        lp, lr = p.shape[1], r.shape[1]
        perm_count = int (perm_pq) + int (perm_rs)
        # (pr|vu) <- (pr|sq); (pv|ru) <- (ps|rq)
        if perm_count == 2: # p is q and r is s
            prsq = self.prsq.transpose (3,2,1,0)
            psrq = self.psrq.transpose (3,2,1,0)
        elif perm_pq: # p is q
            prsq = self.psrq.transpose (3,2,1,0)
            psrq = self.prsq.transpose (3,2,1,0)
        elif perm_rs: # r is s
            prsq = self.psrq
            psrq = self.prsq
        else:
            prsq = self.prsq
            psrq = self.psrq
        prvu = prsq[:lp,:,:,:][:,:lr,:,:]
        pvru = psrq[:lp,:,:,:][:,:,:lr,:]
        eri = prvu + pvru.transpose (0,2,1,3)
        #eri = self.prvu + self.pvru.transpose (0,2,1,3) #self._get_eri ([p,r,v,u]) + self._get_eri ([p,v,r,u]).transpose (0,2,1,3)
        hess = np.tensordot (eri,  D_uq, axes=(3,0)) # 'prvu,uq->prvq'
        hess = np.tensordot (hess, D_vs, axes=(2,0)) # 'prvq,vs->prqs'
        return hess.transpose (0,2,1,3) # 'prqs->pqrs'

    def _get_entangled (self, p, dm):
        ''' Do SVD of a 1-rdm to get a small number of orbitals that you need to actually pay attention to
        when computing splitc and splitx eris '''
        q = linalg.qr (p)[0]
        qH = q.conjugate ().T
        lvec, sigma, rvec = linalg.svd (qH @ dm @ p, full_matrices=False)
        idx = np.abs (sigma) > 1e-8
        return q @ lvec[:,idx]

    def _append_entangled (self, p):
        ''' Do SVD of 1-rdms to get a small number of orbitals that you need to actually pay attention to
        when computing splitc and splitx eris. Append these extras to the end of p '''
        q = get_complementary_states (p)
        qH = q.conjugate ().T
        lvecs = []
        for dm in self.oneRDMs:
            lvec, sigma, rvec = linalg.svd (qH @ dm @ p, full_matrices=False)
            idx = np.abs (sigma) > 1e-8
            if np.count_nonzero (idx): lvecs.append (lvec[:,idx])
        if len (lvecs):
            lvec = self._get_collective_basis (*lvecs)[0]
            u = q @ lvec
            return np.append (p, u, axis=1)
        return p

    def get_gradient (self, p, q):
        ''' A routine to calculate the gradient because it's convenient to have it here '''
        # Put the orbital ranges in the orthonormal basis for fun and profit
        p = self.moHS @ p
        q = self.moHS @ q
        self._cache_eri_baaa (p, q)
        # One-body part
        e1 = sum ([f @ g for f, g in zip (self.fock, self.oneRDMs)])
        e1 = p.conjugate ().T @ (e1 - e1.T) @ q
        # Two-body part
        for idx, (t, a, n, baaa) in enumerate (zip (self.twoCDM, self.mo2amo, self.ncas, self.baaa)):
            #print ("<a{}|p>: {}, {}".format (idx, *measure_basis_olap (p, a)))
            #print ("<a{}|q>: {}, {}".format (idx, *measure_basis_olap (q, a)))
            a2p = a.conjugate ().T @ p
            a2q = a.conjugate ().T @ q
            p2b = p.conjugate ().T @ self.pqrs_b
            q2b = q.conjugate ().T @ self.pqrs_b
            paaa = np.tensordot (p2b, baaa, axes=1)
            qaaa = np.tensordot (q2b, baaa, axes=1)
            e1 +=  np.tensordot (paaa, t, axes=((1,2,3),(1,2,3))) @ a2q
            e1 -= (np.tensordot (qaaa, t, axes=((1,2,3),(1,2,3))) @ a2p).T
        return e1

    def get_diagonal_step (self, p, q):
        ''' Obtain a gradient-descent approximation for the relaxation of orbitals p in range q using the gradient and
        diagonal elements of the Hessian, x^p_q = -E1^p_q / E2^pp_qq '''
        # First, get the gradient and svd to obtain conjugate orbitals of p in q
        lvec, e1, rvecH = linalg.svd (self.get_gradient (p, q), full_matrices=False)
        rvec = rvecH.conjugate ().T
        print ("compare this to the last print of the gradient", e1)
        p = p @ lvec
        q = q @ rvec
        #e2 = self.__call__(p, q, diagx=True)
        lp = p.shape[-1]
        lq = q.shape[-1]
        lpq = lp * lq
        # Zero gradient escape
        if not np.count_nonzero (np.abs (e1) > 1e-8): return p, np.zeros (lp), q
        pqu = self._append_entangled (np.append (p, q, axis=1))
        self._eri_kernel = self._get_eri ([pqu, pqu, pqu, pqu]) 
        self._eri_kernel_basis = pqu
        # Because this ^ is faster than sectioning it and even with 20+20+20 active/inactive/external, it's still only 100 MB
        e2 = self.__call__(p, q, p, q)
        e2 = np.diag (np.diag (e2.reshape (lpq, lpq)).reshape (lp, lq))
        self._eri_kernel = self._eri_kernel_basis = None
        return p, -e1 / e2, q

    def get_conjugate_gradient (self, p, q, r, s):
        ''' Obtain the gradient for ranges p->q after making an approximate gradient-descent step in r->s:
        E1'^p_q = E1^p_q - E2^pr_qs * x^r_s = E1^p_q + E2^pr_qs * E1^r_s / E2^rr_ss '''
        e1pq = self.get_gradient (p, q)
        r, x_rs, s = self.get_diagonal_step (r, s)
        #e2 = self.__call__(p, q, r, s, diagx1=False, diagx2=True)
        # Zero step escape
        if not np.count_nonzero (np.abs (x_rs) > 1e-8): return e1pq
        lp = p.shape[-1]
        lq = q.shape[-1]
        lr = r.shape[-1]
        ls = s.shape[-1]
        diag_idx = np.arange (lr, dtype=int)
        diag_idx = (diag_idx * lr) + diag_idx
        e2 = self.__call__(p, q, r, s).reshape (lp, lq, lr*ls)[:,:,diag_idx]
        e1pq += np.tensordot (e2, x_rs, axes=1)
        return e1pq

class CASSCFHessianTester (object):
    ''' Use pyscf.mcscf.mc1step.gen_g_hop to test HessianCalculator.
    There are 3 nonredundant orbital rotation sectors: ui, ai, and au
    Therefore there are 6 nonredundant Hessian sectors: uiui, uiai,
    uiau, aiai, aiau, and auau. Note that PySCF chooses to store the
    lower-triangular (p>q) part. Sadly I can't use permutation symmetry
    to accelerate any of these because any E2^pr_qr would necessarily
    refer to a redundant rotation.'''

    def __init__(self, mc):
        oneRDMs = mc.make_rdm1s ()
        casdm1s = mc.fcisolver.make_rdm1s (mc.ci, mc.ncas, mc.nelecas)
        casdm1, casdm2 = mc.fcisolver.make_rdm12 (mc.ci, mc.ncas, mc.nelecas)
        twoCDM = get_2CDM_from_2RDM (casdm2, casdm1s)
        ao2amo = mc.mo_coeff[:,mc.ncore:][:,:mc.ncas]
        self.calculator = HessianCalculator (mc._scf, oneRDMs, twoCDM, ao2amo)
        self.cas = mc
        self.cas_mo = mc.mo_coeff
        self.ncore, self.ncas, self.nelecas = mc.ncore, mc.ncas, mc.nelecas
        self.nocc = self.ncore + self.ncas
        self.nmo = self.cas_mo.shape[1]
        self.hop, self.hdiag = gen_g_hop (mc, self.cas_mo, 1, casdm1, casdm2, mc.ao2mo (self.cas_mo))[2:]

    def __call__(self, pq, rs=None):
        ''' pq, rs = 0 (ui), 1 (ai), 2 (au) '''
        if rs is None: return self._call_diag (pq)
        
        p, q, prange, qrange, np, nq = self._parse_range (pq)
        r, s, rrange, srange, nr, ns = self._parse_range (rs)

        my_hess = self.calculator (p, q, r, s)
        print ("{:8s} {:13s} {:13s} {:13s} {:13}".format ('Idx', 'Mine', "PySCF's", 'Difference', 'Ratio'))
        fmt_str = "{0:d},{1:d},{2:d},{3:d} {4:13.6e} {5:13.6e} {6:13.6e} {7:13.6e}"

        for (ixp, pi), (ixq, qi) in product (enumerate (prange), enumerate (qrange)):
            Py_hess = self._pyscf_hop_call (pi, qi, rrange, srange)
            for (ixr, ri), (ixs, si) in product (enumerate (rrange), enumerate (srange)):
                diff = my_hess[ixp,ixq,ixr,ixs] - Py_hess[ixr,ixs]
                rat = my_hess[ixp,ixq,ixr,ixs] / Py_hess[ixr,ixs]
                print (fmt_str.format (pi, qi, ri, si, my_hess[ixp,ixq,ixr,ixs], Py_hess[ixr,ixs], diff, rat))

    def _call_diag (self, pq):
        offs = ix = np = nq = 0
        while ix < pq:
            np, nq = self._parse_range (ix)[4:]
            offs += np*nq
            ix += 1
        p, q, prange, qrange, np, nq = self._parse_range (pq)
        Py_hdiag = self.hdiag[offs:][:np*nq].reshape (np, nq)
        my_hdiag = self.calculator (p, q)

        print ("{:8s} {:13s} {:13s} {:13s} {:13}".format ('Idx', 'Mine', "PySCF's", 'Difference', 'Ratio'))
        fmt_str = "{0:d},{1:d},{0:d},{1:d} {2:13.6e} {3:13.6e} {4:13.6e} {5:13.6e}"
        for (ixp, pi), (ixq, qi) in product (enumerate (prange), enumerate (qrange)):
            diff = my_hdiag[ixp,ixq] - Py_hdiag[ixp,ixq]
            rat = my_hdiag[ixp,ixq] / Py_hdiag[ixp,ixq]
            print (fmt_str.format (pi, qi, my_hdiag[ixp,ixq], Py_hdiag[ixp,ixq], diff, rat))

    def _parse_range (self, pq):
        if pq == 0: # ui
            p = self.cas_mo[:,self.ncore:self.nocc]
            q = self.cas_mo[:,:self.ncore]
            prange = range (self.ncore,self.nocc)
            qrange = range (self.ncore)
            np = self.ncas
            nq = self.ncore
        elif pq == 1: # ai
            p = self.cas_mo[:,self.nocc:]
            q = self.cas_mo[:,:self.ncore]
            prange = range (self.nocc, self.nmo)
            qrange = range (self.ncore)
            np = self.nmo - self.nocc
            nq = self.ncore
        elif pq == 2: # au
            p = self.cas_mo[:,self.nocc:]
            q = self.cas_mo[:,self.ncore:self.nocc]
            prange = range (self.nocc, self.nmo)
            qrange = range (self.ncore,self.nocc)
            np = self.nmo - self.nocc
            nq = self.ncas
        else: 
            raise RuntimeError ("Undefined range {}".format (pq))
        return p, q, prange, qrange, np, nq

    def _pyscf_hop_call (self, ip, iq, rrange, srange):
        kappa = np.zeros ([self.nmo, self.nmo])
        kappa[ip,iq] = 1
        kappa = self.cas.pack_uniq_var (kappa)
        py_hess = self.hop (kappa)
        py_hess = self.cas.unpack_uniq_var (py_hess)
        return py_hess[rrange,:][:,srange]

class LASSCFHessianCalculator (HessianCalculator):

    def __init__(self, ints, oneRDM_loc, all_frags, fock_c):
        self.ints = ints
        active_frags = [f for f in all_frags if f.norbs_as]
        self._eri_kernel = self._eri_kernel_basis = None 

        # Global things
        self.nmo = self.nao = ints.norbs_tot
        self.mo = self.moH = self.Smo = self.moHS = np.eye (self.nmo)
        oneSDM_loc = sum ([f.oneSDMas_loc for f in active_frags])
        self.oneRDMs = [(oneRDM_loc + oneSDM_loc)/2, (oneRDM_loc - oneSDM_loc)/2]
        #fock_c = ints.loc_rhf_fock_bis (oneRDM_loc)
        fock_s = -ints.loc_rhf_k_bis (oneSDM_loc) / 2 if isinstance (oneSDM_loc, np.ndarray) else 0
        self.fock = [fock_c + fock_s, fock_c - fock_s]

        # Fragment things
        self.mo2amo = [f.loc2amo for f in active_frags]
        self.twoCDM = [f.twoCDMimp_amo for f in active_frags]
        self.ncas = [f.norbs_as for f in active_frags]

    def _get_eri_external (self, orbs_list, compact=False):
        return self.ints.general_tei (orbs_list, compact=compact)
        
    
