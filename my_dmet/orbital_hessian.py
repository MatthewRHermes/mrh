import numpy as np
from pyscf import ao2mo
from pyscf.mcscf.mc1step import gen_g_hop
from mrh.util.basis import represent_operator_in_basis, is_basis_orthonormal
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

    def __call__(self, *args):
        if len (args) == 0:
            ''' If no orbital ranges are passed, return the full mo-basis Hessian '''
            return self._call_diag (self.mo, self.mo)
        elif len (args) == 1:
            ''' If one orbital range is passed, return the Hessian with all four indices in that range'''
            return self._call_diag (args[0], args[0])
        elif len (args) == 2:
            ''' If two orbital ranges are passed, return the Hessian with the first range specifying rotation 1
            and the second specifying rotation 2 (be careful when doing this; it's ^pq_pq, not ^pp_qq, because the
            former is the configuration where you can use the permutation symmetry to speed up the thing '''
            return self._call_diag (args[0], args[1])
        elif len (args) == 3:
            ''' First two orbital ranges specify rotation 1; third specifies rotation 2 '''
            return self._call_semidiag (args[0], args[1], args[2])
        elif len (args) == 4:
            ''' If all four orbital ranges are passed, return the Hessian so specified. No permutation symmetry can be exploited. '''
            return self._call_full (args[0], args[1], args[2], args[3])
        else:
            raise RuntimeError ("Orbital Hessian has 4 orbital indices; you passed {} orbital ranges".format (len (args)))

    def _call_diag (self, p, q):
        ''' The Hessian E2^pq_p'q' is F2^pq_p'q' - F2^p'q_pq' - F2^pq'_p'q + F2^p'q'_pq 
        Use full permutation symmetry to accelerate it!'''
        norb = [p.shape[-1], p.shape[-1], q.shape[-1], q.shape[-1]]
        if 0 in norb: return np.zeros (norb)
        # Put the orbital ranges in the orthonormal basis for fun and profit
        p = self.moHS @ p
        q = self.moHS @ q
        hess = self._get_Fock2 (p, p, q, q)
        hess -= hess.transpose (1, 0, 2, 3)
        hess -= hess.transpose (0, 1, 3, 2)
        return hess / 2

    def _call_semidiag (self, p, q, r):
        ''' The Hessian E2^pr_qr' is F2^pr_qr' - F2^qr_pr' - F2^pr'_qr + F2^qr'_pr 
        Use permutation symmetry on the second rotation'''
        norb = [p.shape[-1], q.shape[-1], r.shape[-1], r.shape[-1]]
        if 0 in norb: return np.zeros (norb)
        # Put the orbital ranges in the orthonormal basis for fun and profit
        p = self.moHS @ p
        q = self.moHS @ q
        r = self.moHS @ r
        hess  = self._get_Fock2 (p, q, r, r)
        hess -= self._get_Fock2 (q, p, r, r).transpose (1,0,2,3)
        hess -= hess.transpose (0,1,3,2)
        return hess / 2

    def _call_full (self, p, q, r, s):
        ''' The Hessian E2^pr_qs is F2^pr_qs - F2^qr_ps - F2^ps_qr + F2^qs_pr. 
        Since the orbitals are segmented into separate ranges, you cannot necessarily just calculate
        one of these and transpose. '''
        norb = [p.shape[-1], q.shape[-1], r.shape[-1], s.shape[-1]]
        if 0 in norb: return np.zeros (norb)
        # Put the orbital ranges in the orthonormal basis for fun and profit
        p = self.moHS @ p
        q = self.moHS @ q
        r = self.moHS @ r
        s = self.moHS @ s
        hess  = self._get_Fock2 (p, q, r, s)
        hess -= self._get_Fock2 (q, p, r, s).transpose (1,0,2,3)
        hess -= self._get_Fock2 (p, q, s, r).transpose (0,1,3,2)
        hess += self._get_Fock2 (q, p, s, r).transpose (1,0,3,2)
        return hess / 2

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
        for t, a, n in zip (self.twoCDM, self.mo2amo, self.ncas):
            a2q = a.conjugate ().T @ q
            a2s = a.conjugate ().T @ s
            # If either q or s has no weight on the current active space, skip
            if np.amax (np.abs (a2q)) < 1e-8 or np.amax (np.abs (a2s)) < 1e-8:
                continue
            eri = self._get_eri ([p, r, a, a])
            thess  = np.tensordot (eri, t, axes=((2,3),(2,3)))
            eri = self._get_eri ([p, a, r, a])
            thess += np.tensordot (eri, t + t.transpose (0,1,3,2), axes=((1,3),(1,3)))
            thess = np.tensordot (thess, a2q, axes=(2,0)) # 'prab,aq->prbq' (tensordot always puts output indices in order of the arguments)
            thess = np.tensordot (thess, a2s, axes=(2,0)) # 'prbq,bs->prqs'
            hess += 2 * thess.transpose (0, 2, 1, 3) # 'prqs->pqrs'

        # Weirdo split-coulomb and split-exchange terms
        hess += 4 * self._get_splitc (p, q, r, s, self.oneRDMs[0] + self.oneRDMs[1])
        for dm in self.oneRDMs:
            hess -= 2 * self._get_splitx (p, q, r, s, dm)

        return hess

    def _get_Fock1 (self, p, q):
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

    def _get_splitc (self, p, q, r, s, dm):
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

    def _get_splitx (self, p, q, r, s, dm):
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

    def _get_entangled (self, p, dm):
        ''' Do SVD of a 1-rdm to get a small number of orbitals that you need to actually pay attention to
        when computing splitc and splitx eris '''
        q = linalg.qr (p)[0]
        qH = q.conjugate ().T
        lvec, sigma, rvec = linalg.svd (qH @ dm @ p, full_matrices=False)
        idx = np.abs (sigma) > 1e-8
        return q @ lvec[:,idx]


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

    def __call__(self, pq, rs):
        ''' pq, rs = 0 (ui), 1 (ai), 2 (au) '''
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

