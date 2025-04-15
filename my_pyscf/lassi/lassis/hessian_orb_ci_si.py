import numpy as np
from pyscf import lib
from mrh.my_pyscf.lassi.lassis import coords
from mrh.my_pyscf.lassi import op_o0, op_o1
from mrh.my_pyscf.lassi.spaces import list_spaces
from itertools import permutations
from scipy.sparse import linalg as sparse_linalg

op = (op_o0, op_o1)

class HessianOperator (sparse_linalg.LinearOperator):
    '''Engine for mapping a step vector to an extended model space in order to evaluate
    Hessian-vector products efficiently'''

    def __init__(self, ugg):
        self.ugg = ugg
        self.nmo = ugg.nmo
        self.mo_coeff = mo_coeff = ugg.mo_coeff
        self.nfrags = ugg.nfrags
        self.ci = ugg.ci
        self.nroots = len (ugg.ci[0])
        self.lsi = lsi = ugg.lsi
        self.opt = lsi.opt
        self.si = ugg.raw2orth.H (ugg.si)
        self.nprods = ugg.nprods
        self.nroots_si = ugg.nroots_si
        self.shape = (ugg.nvar_tot,ugg.nvar_tot)
        self.dtype = ugg.mo_coeff.dtype # TODO: generalize this for SOC
        self.eris = self.lsi.get_casscf_eris (self.mo_coeff)
        ncore, ncas = lsi.ncore, lsi.ncas
        nocc = ncore + ncas
        las = lsi._las
        self.h2_paaa = []
        for i in range (self.nmo):
            self.h2_paaa.append (self.eris.ppaa[i][ncore:nocc])
        
        self.casdm1, self.casdm2 = lsi.make_casdm12 (
            ci=self.ci, si=self.si, opt=self.opt
        )
        self.h2_paaa = np.stack (self.h2_paaa, axis=0)
        dm0 = 2*mo_coeff[:,:ncore] @ mo_coeff[:,:ncore].conj ().T
        self.veff_c = np.squeeze (las.get_veff (dm=dm0))
        self.ham_2q = lsi.ham_2q (mo_coeff)
        h1 = lsi._las.get_hcore () + self.veff_c
        self.h1 = mo_coeff.conj ().T @ h1 @ mo_coeff
        self.fock1 = self.get_fock1 (self.h1, self.h2_paaa, self.casdm1, self.casdm2)
        self.spaces = list_spaces (lsi)
        self.e_roots_si = np.zeros (self.nroots_si)
        self.e_roots_si = np.dot (self.si.conj ().T, self.hsi_op (self.ham_2q, self.ci, self.si))

    def get_fock1 (self, h1, h2_paaa, casdm1, casdm2, _coreocc=2):
        ncore, ncas = self.lsi.ncore, self.lsi.ncas
        nocc = ncore + ncas
        dm1 = _coreocc*np.eye (self.nmo).astype (casdm1.dtype)
        try:
            dm1[ncore:nocc,ncore:nocc] = casdm1
        except ValueError as err:
            print (self.si.shape)
            raise (err)
        fock1 = np.dot (h1, dm1)
        fock1[:,ncore:nocc] += lib.einsum ('pbcd,abcd->pa', h2_paaa, casdm2)
        return fock1

    def to_hop (self, x):
        xorb, xci_ref, xci_sf, xci_ch, xsi = self.ugg.unpack (x)
        xci = self.lsi.prepare_model_states (xci_ref, xci_sf, xci_ch)[0].ci
        ci1 = [c*(self.nfrags+1) for c in self.ci]
        n = len (xci[0])
        for i in range (self.nfrags):
            ci1[i][n*(i+1):n*(i+2)] = xci[i]
        si0 = np.tile (self.si, (self.nfrags+1,1))
        si1 = si0.copy ()
        si0[self.nprods:,:] = 0.0
        si1[:self.nprods,:] = xsi[:]
        return xorb, ci1, si0, si1

    def from_hop (self, kappa=None, ci_10=None, si_10=None, ci_01=None, si_01=None):
        '''Add the orbital internal indices into ?i_01'''
        if kappa is None:
            kappa = np.zeros ((self.nmo, self.nmo), dtype=self.mo_coeff.dtype)
        ci1 = [[0.0 for ci_ij in ci_i] for ci_i in self.ci]

        # TODO: SA generalization

        if ci_01 is not None:
            for i in range (self.nfrags):
                for r in range (self.nroots):
                    ci1[i][r] += ci_01[i][r][0,...,0]
        if ci_10 is not None:
            for i,j in permutations (range (self.nfrags), 2):
                ci_10_i = ci_10[i][self.nroots*(j+1):self.nroots*(j+2)]
                for r in range (self.nroots):
                    ci1[i][r] += ci_10_i[r][0,...,0]
            for i in range (self.nfrags):
                for r in range (self.nroots):
                    ci1[i][r] += ci_10[i][r][0,...,0]
        for i in range (self.nfrags):
            for r in range (self.nroots):
                ci1[i][r] += ci1[i][r].conj () # + h.c.
        ci1_ref, ci1_sf, ci1_ch = coords.sum_hci (self.lsi, ci1)

        si1 = np.zeros_like (self.si)
        if si_01 is not None:
            si1 += si_01[:self.nprods,:]
        if si_10 is not None:
            si1 += si_10.reshape (self.nfrags+1,self.nprods,self.nroots_si)[1:].sum (0)
        si1 += si1.conj () # + h.c.

        return self.ugg.pack (kappa, ci1_ref, ci1_sf, ci1_ch, si1)

    def get_nelec_frs (self, nr=None):
        nelec_frs = self.lsi.get_nelec_frs ()
        nr0 = nelec_frs.shape[1]
        if nr is not None:
            assert (nr%nr0==0)
            nelec_frs = np.concatenate ([nelec_frs,]*(nr//nr0), axis=1)
        return nelec_frs 

    def hci_op (self, ham_2q, ci, si_bra, si_ket):
        nelec_frs = self.get_nelec_frs (nr=len(ci[0]))
        ncore, ncas = self.lsi.ncore, self.lsi.ncas
        nocc = ncore+ncas
        h0, h1_pp, h2_paaa = ham_2q
        h1 = h1_pp[ncore:nocc,ncore:nocc]
        h2 = h2_paaa[ncore:nocc]
        return op[self.opt].contract_ham_ci (
            self.lsi, h1, h2, ci, nelec_frs, ci, nelec_frs,
            si_bra=si_bra, si_ket=si_ket, h0=h0
        )

    def hsi_op (self, ham_2q, ci, si):
        nelec_frs = self.get_nelec_frs (nr=len(ci[0]))
        ncore, ncas = self.lsi.ncore, self.lsi.ncas
        nocc = ncore+ncas
        h0, h1_pp, h2_paaa = ham_2q
        h1 = h1_pp[ncore:nocc,ncore:nocc]
        h2 = h2_paaa[ncore:nocc]
        ham_op, _, ovlp_op = op[self.opt].gen_contract_op_si_hdiag (
            self.lsi, h1, h2, ci, nelec_frs
        )[:3]
        return ham_op (si) - (self.e_roots_si - h0) * si

    def get_xham_2q (self, kappa):
        return xham_2q (self.lsi, kappa, mo_coeff=self.mo_coeff, eris=self.eris,
                        veff_c=self.veff_c)

    def _matvec (self, x):
        kappa, ci1, si0, si1 = self.to_hop (x)
        xham_2q = self.get_xham_2q (kappa)
        ham_2q = self.ham_2q

        rorb = self.hoo (xham_2q, kappa)
        rorb += self.hoa (ci1, si0, si1)

        # TODO: why divide by 2?
        xham_2q = [h/2 for h in xham_2q]

        rci_01 = self.hci_op (xham_2q, ci1, si0, si0)
        rci_001 = self.hci_op (ham_2q, ci1, si0, si1)
        for i in range (self.nfrags):
            for j in range (len (rci_01[i])):
                rci_01[i][j] += rci_001[i][j]
        rci_10 = self.hci_op (ham_2q, ci1, si1, si0)

        rsi_01 = self.hsi_op (xham_2q, ci1, si0)
        rsi_01 += self.hsi_op (ham_2q, ci1, si1)
        rsi_10 = self.hsi_op (ham_2q, ci1, si0)

        return self.from_hop (rorb, rci_10, rsi_10, rci_01, rsi_01)

    def hoo (self, xham_2q, kappa):
        h0, h1, h2 = xham_2q
        fx = self.get_fock1 (h1, h2, self.casdm1, self.casdm2)
        return (fx - fx.T) / 2

    def hoa (self, ci1, si0, si1):
        casdm1, casdm2 = self.lsi.trans_casdm12 (ci=ci1, si_bra=si0, si_ket=si1,
                                                 spaces=self.spaces*(self.nfrags+1), opt=self.opt)
        casdm1 += casdm1.T
        casdm2 += casdm2.transpose (1,0,3,2)
        fx = self.get_fock1 (self.h1, self.h2_paaa, casdm1, casdm2, _coreocc=0)
        return fx - fx.T

def xham_2q (lsi, kappa, mo_coeff=None, eris=None, veff_c=None):
    las = lsi._las
    if mo_coeff is None: mo_coeff=lsi.mo_coeff
    if eris is None: eris = lsi.get_casscf_eris (mo_coeff)
    nao, nmo = mo_coeff.shape
    ncore, ncas = lsi.ncore, lsi.ncas
    nocc = ncore + ncas
    if veff_c is None:
        if ncore:
            dm0 = 2*mo_coeff[:,:ncore] @ mo_coeff[:,:ncore].conj ().T
            veff_c = np.squeeze (las.get_veff (dm=dm0))
        else:
            veff_c = 0

    h0 = 0

    mo0 = mo_coeff
    mo0H = mo0.conj ().T
    mo1 = np.dot (mo0, kappa)
    mo1H = np.dot (kappa, mo0H)
    h1_0 = las.get_hcore () + veff_c
    h1 = mo0H @ h1_0 @ mo1 - mo1H @ h1_0 @ mo0
    if ncore:
        dm1 = 2*np.eye (nmo)
        dm1[ncore:] = 0
        dm1 = kappa @ dm1
        dm1 += dm1.T
        dm1 = mo0 @ dm1 @ mo0H
        h1_1 = np.squeeze (las.get_veff (dm=dm1))
        h1_1 = mo0H @ h1_1 @ mo0
        h0 = 2*np.sum (h1_1.diagonal ()[:ncore])
        h1 += h1_1

    h2 = np.stack ([eris.ppaa[i] for i in range (nmo)], axis=0)
    if ncas:
        h2 = -lib.einsum ('pq,qabc->pabc',kappa,h2)
        kpa = kappa[:,ncore:nocc]
        kap = kappa[ncore:nocc,:]
        for i in range (nmo):
            h2[i] += lib.einsum ('pab,pq->qab',eris.ppaa[i],kpa)
            h2[i] -= lib.einsum ('pq,aqb->apb',kap,eris.papa[i])
            h2[i] += lib.einsum ('apb,pq->abq',eris.papa[i],kpa)

    return h0, h1, h2

 
