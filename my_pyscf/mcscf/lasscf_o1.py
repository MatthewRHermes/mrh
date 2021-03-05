import time
import numpy as np
from scipy import linalg
from pyscf import gto, lib, ao2mo
from mrh.my_pyscf.mcscf import lasci, lasscf_o0
from functools import partial

# Let's finally implement, in the more pure LASSCF rewrite, the ERI-
# related cost savings that I made such a big deal about in JCTC 2020,
# 16, 4923

def make_schmidt_spaces (h_op):
    ''' Build the spaces which active orbitals will explore in this
    macrocycle, based on gradient and Hessian SVDs and Schmidt
    decompositions. Must be called after __init__ is complete

    Args:
        h_op: LASSCF Hessian operator instance

    Returns:
        uschmidt : nfrag ndarrays of shape (nmo, *)
            The number of subspaces built is equal to the
            product of the number of irreps in the molecule
            and the number of fragments, minus the number
            of null spaces.

    '''
    las = h_op.las
    ugg = h_op.ugg
    mo_coeff = h_op.mo_coeff
    nao, nmo = mo_coeff.shape
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    dm1 = h_op.dm1s.sum (0)
    g_vec = h_op.get_grad ()
    prec_op = h_op.get_prec ()
    hx_vec = prec_op (-g_vec)
    gorb1 = ugg.unpack (g_vec)[0]
    gorb2 = gorb1 + ugg.unpack (hx_vec)[0]

    def _svd (metric, q, p):
        m, n = q.shape[1], p.shape[1]
        k = min (m, n)
        qh = q.conj ().T
        u, svals, vh = linalg.svd (qh @ metric @ p)
        idx_sort = np.argsort (-np.abs (svals))
        svals = svals[idx_sort]
        u[:,:k] = u[:,idx_sort]
        return q @ u, svals

    def _check (tag, umat_p, umat_q):
        np, nq = umat_p.shape[1], umat_q.shape[1]
        k = min (np, nq)
        lib.logger.debug (las, '%s size of pspace = %d, qspace = %d', tag, np, nq)
        return k

    def _grad_svd (tag, geff, umat_p, umat_q, ncoup=0):
        umat_q, svals = _svd (geff, umat_q, umat_p)
        umat_p = np.append (umat_p, umat_q[:,:ncoup], axis=1)
        umat_q = umat_q[:,ncoup:]
        k = _check ('after {} SVD'.format (tag), umat_p, umat_q)
        lib.logger.debug (las, '%s SVD lowest eigenvalue = %e', tag, svals[ncoup-1])
        return umat_p, umat_q, k

    def _schmidt (tag, umat_p, umat_q, thresh=1e-8):
        umat_q, svals = _svd (dm1, umat_q, umat_p)
        ncoup = np.count_nonzero (np.abs (svals) > thresh)
        umat_p = np.append (umat_p, umat_q[:,:ncoup], axis=1)
        umat_q = umat_q[:,ncoup:]
        dm_pp = umat_p.conj ().T @ dm1 @ umat_p
        lib.logger.debug (las, 'number of electrons in p-space after %s Schmidt = %e', tag, np.trace (dm_pp))
        k = _check ('after {} Schmidt'.format (tag), umat_p, umat_q)
        return umat_p, umat_q, k

    def _make_single_space (mask):
        # Active orbitals other than the current should be in
        # neither the p-space nor the q-space
        # This is because I am still assuming we have all 
        # eris of type aaaa
        nlas = np.count_nonzero (mask)
        umat_p = np.diag (mask.astype (mo_coeff.dtype))[:,mask]
        umat_q = np.eye (nmo)
        umat_q = np.append (umat_q[:,:ncore], umat_q[:,nocc:], axis=1)
        # At any of these steps we might run out of orbitals...
        # The Schmidt steps might turn out to be completely unnecessary
        k = _check ('initial', umat_p, umat_q)
        if k == 0: return umat_p
        umat_p, umat_q, k = _grad_svd ('g', gorb1, umat_p, umat_q, ncoup=k)
        if k == 0: return umat_p
        umat_p, umat_q, k = _schmidt ('first', umat_p, umat_q) 
        if k == 0: return umat_p
        umat_p, umat_q, k = _grad_svd ('g+hx', gorb2, umat_p, umat_q, ncoup=min(k,2*nlas))
        if k == 0: return umat_p
        umat_p, umat_q, k = _schmidt ('second', umat_p, umat_q)
        return umat_p

    orbsym = getattr (mo_coeff, 'orbsym', np.zeros (nmo))
    uschmidt = []
    for ilas in range (len (las.ncas_sub)):
        i = sum (las.ncas_sub[:ilas]) + ncore
        j = i + las.ncas_sub[ilas]
        irreps, idx_irrep = np.unique (orbsym[i:j], return_inverse=True)
        ulist = []
        for ix in range (len (irreps)):
            idx = np.squeeze (np.where (idx_irrep==ix)) + i
            idx_mask = np.zeros (nmo, dtype=np.bool_)
            idx_mask[idx] = True
            ulist.append (_make_single_space (idx_mask))
        uschmidt.append (np.concatenate (ulist, axis=1))

    return uschmidt

class LASSCF_HessianOperator (lasscf_o0.LASSCF_HessianOperator):

    make_schmidt_spaces = make_schmidt_spaces

    def _init_eri (self):
        lasci._init_df_(self)
        t0 = (time.clock (), time.time ())
        self.uschmidt = uschmidt = self.make_schmidt_spaces ()
        t1 = lib.logger.timer (self.las, 'build schmidt spaces', *t0)
        if isinstance (self.las, lasci._DFLASCI):
            eri = self.las.with_df.ao2mo
        elif getattr (self.las._scf, '_eri', None) is not None:
            eri = partial (ao2mo.full, self.las._scf._eri)
        else:
            eri = partial (ao2mo.full, self.las.mol)
        self.eri_imp = []
        for ix, umat in enumerate (uschmidt):
            nimp = umat.shape[1]
            mo = self.mo_coeff @ umat
            self.eri_imp.append (ao2mo.restore (1, eri (mo), nimp))
            t1 = lib.logger.timer (self.las, 'schmidt-space {} eri array'.format (ix), *t1)
        # eri_cas is taken from h2eff_sub
        lib.logger.timer (self.las, '_init_eri', *t0)

    def split_veff (self, veff_mo, dm1s_mo):
        veff_c = veff_mo.copy ()
        ncore = self.ncore
        nocc = self.nocc
        sdm = dm1s_mo[0] - dm1s_mo[1]
        veff_s = np.zeros_like (veff_c)
        # (H.x_pa)_aa
        veff_s[ncore:nocc,ncore:nocc] = np.tensordot (self.h2eff_sub,
            sdm[:,ncore:nocc], axes=((0,3),(0,1)))
        # (H.x_ua)_ua
        for uimp, eri in zip (self.uschmidt, self.eri_imp):
            s = uimp.conj ().T @ sdm @ uimp
            v = np.tensordot (eri, s, axes=((1,2),(0,1)))
            veff_s += uimp @ v @ uimp.conj ().T
        veff_s[:,:] *= -0.5
        veffa = veff_c + veff_s
        veffb = veff_c - veff_s
        return np.stack ([veffa, veffb], axis=0)

    def orbital_response (self, kappa, odm1fs, ocm2, tdm1frs, tcm2, veff_prime):
        ''' Parent class does everything except va/ac degrees of freedom
        (c: closed; a: active; v: virtual; p: any) '''
        ncore, nocc, nmo = self.ncore, self.nocc, self.nmo
        gorb = lasci.LASCI_HessianOperator.orbital_response (self, kappa, odm1fs,
            ocm2, tdm1frs, tcm2, veff_prime)
        f1_prime = np.zeros ((self.nmo, self.nmo), dtype=self.dtype)
        # (H.x_ua)_ua, (H.x_ua)_vc
        for uimp, eri in zip (self.uschmidt, self.eri_imp):
            uimp_cas = uimp[ncore:nocc,:]
            cm2 = np.tensordot (ocm2, uimp_cas, axes=((2),(0))) # pqrs -> pqsr
            cm2 = np.tensordot (cm2, uimp, axes=((2),(0))) # pqsr -> pqrs
            cm2 = np.tensordot (uimp_cas.conj (), cm2, axes=((0),(1))) # pqrs -> qprs
            cm2 = np.tensordot (uimp_cas.conj (), cm2, axes=((0),(1))) # qprs -> pqrs
            cm2 += cm2.transpose (1,0,3,2)
            cm2 += cm2.transpose (2,3,0,1)
            f1 = np.tensordot (eri, cm2, axes=((1,2,3),(1,2,3)))
            f1_prime += uimp @ f1 @ uimp.conj ().T
        # (H.x_aa)_ua
        ecm2 = ocm2[:,:,:,ncore:nocc] + ocm2[:,:,:,ncore:nocc].transpose (1,0,3,2)
        ecm2 += ecm2.transpose (2,3,0,1) + tcm2
        f1_prime[:ncore,ncore:nocc] += np.tensordot (self.h2eff_sub[:ncore], ecm2, axes=((1,2,3),(1,2,3)))
        f1_prime[nocc:,ncore:nocc] += np.tensordot (self.h2eff_sub[nocc:], ecm2, axes=((1,2,3),(1,2,3)))
        # (H.x_ua)_aa
        ecm2 = ocm2.copy ()
        f1_aa = f1_prime[ncore:nocc,ncore:nocc]
        f1_aa[:,:] += (np.tensordot (self.h2eff_sub[:ncore], ocm2[:,:,:,:ncore], axes=((0,2,3),(3,0,1)))
                     + np.tensordot (self.h2eff_sub[nocc:],  ocm2[:,:,:,nocc:],  axes=((0,2,3),(3,0,1))))
        f1_aa[:,:] += (np.tensordot (self.h2eff_sub[:ncore], ocm2[:,:,:,:ncore], axes=((0,1,3),(3,2,1)))
                     + np.tensordot (self.h2eff_sub[nocc:],  ocm2[:,:,:,nocc:],  axes=((0,1,3),(3,2,1))))
        f1_aa[:,:] += (np.tensordot (self.h2eff_sub[:ncore], ocm2[:,:,:,:ncore], axes=((0,1,2),(3,2,0)))
                     + np.tensordot (self.h2eff_sub[nocc:],  ocm2[:,:,:,nocc:],  axes=((0,1,2),(3,2,0))))
        return gorb + (f1_prime - f1_prime.T)

class LASSCFNoSymm (lasscf_o0.LASSCFNoSymm):
    _hop = LASSCF_HessianOperator

class LASSCFSymm (lasscf_o0.LASSCFSymm):
    _hop = LASSCF_HessianOperator

def LASSCF (mf_or_mol, ncas_sub, nelecas_sub, **kwargs):
    if isinstance(mf_or_mol, gto.Mole):
        mf = scf.RHF(mf_or_mol)
    else:
        mf = mf_or_mol
    if mf.mol.symmetry: 
        las = LASSCFSymm (mf, ncas_sub, nelecas_sub, **kwargs)
    else:
        las = LASSCFNoSymm (mf, ncas_sub, nelecas_sub, **kwargs)
    if getattr (mf, 'with_df', None):
        las = lasci.density_fit (las, with_df = mf.with_df) 
    return las

        

