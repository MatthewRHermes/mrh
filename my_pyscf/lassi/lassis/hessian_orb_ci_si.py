import numpy as np
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
        
