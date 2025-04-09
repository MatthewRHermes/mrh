import numpy as np
from pyscf import lib
from mrh.my_pyscf.lassi.lassis import coords
from itertools import permutations

class HessianIndexEngine (object):
    '''Engine for mapping a step vector to an extended model space in order to evaluate
    Hessian-vector products efficiently'''

    def __init__(self, ugg):
        self.ugg = ugg
        self.nmo = ugg.nmo
        self.mo_coeff = ugg.mo_coeff
        self.nfrags = ugg.nfrags
        self.ci = ugg.ci
        self.nroots = len (ugg.ci[0])
        self.lsi = ugg.lsi
        self.si = ugg.si
        self.nprods = ugg.nprods
        self.nroots_si = ugg.nroots_si

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
        ci1 = [[np.zeros_like (ci_ij) for ci_ij in ci_i] for ci_i in self.ci]

        if ci_01 is not None:
            for i in range (self.nfrags):
                for r in range (self.nroots):
                    ci1[i][r] += ci_01[i][r]
        if ci_10 is not None:
            for i,j in permutations (range (self.nfrags), 2):
                ci0 = ci_10[i][self.nroots*(j+1):self.nroots*(j+2)]
                for r in range (self.nroots):
                    ci1[i][r] += ci0[i][r]
        for i in range (self.nfrags):
            for r in range (self.nroots):
                ci1[i][r] += ci1[i][r].conj () # + h.c.
        ci1_ref, ci1_sf, ci1_ch = coords.sum_hci (self.lsi, ci1)

        si1 = np.zeros_like (self.si)
        if si_01 is not None:
            si1 += si_01[:n,:]
        if si_10 is not None:
            si1 += si_10.reshape (self.nfrags,self.nprods,self.nroots_si)[1:].sum (0)
        si1 += si1.conj () # + h.c.

        return self.ugg.pack (kappa, ci1_ref, ci1_sf, ci1_ch, si1)

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
            veff_c = np.squeeze (las.get_veff (dm1s=dm0))
        else:
            veff_c = 0

    h0 = 0

    mo0 = mo_coeff
    mo0H = mo.conj ().T
    mo1 = np.dot (mo0, kappa)
    mo1H = np.dot (kappa, mo0H)
    h1_0 = las.get_hcore () + veff_c
    h1 = mo0H @ h1_0 @ mo - mo1H @ h1_0 @ mo0
    if ncore:
        dm1 = 2*np.eye (nmo)
        dm1[ncore:] = 0
        dm1 = kappa @ dm1
        dm1 += dm1.T
        dm1 = mo0 @ dm1 @ mo0H
        h1_1 = np.squeeze (las.get_veff (dm1s=dm1))
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

 
