import numpy as np
from scipy import linalg
from pyscf import lib
import copy
from mrh.my_pyscf.mcscf import lasscf_sync_o0
from mrh.my_pyscf.lassi import lassis
from mrh.my_pyscf.lassi.spaces import list_spaces
from mrh.my_pyscf.fci.csfstring import CSFTransformer

class UnitaryGroupGenerators (lasscf_sync_o0.LASSCF_UnitaryGroupGenerators):
    def __init__(self, lsi, mo_coeff, ci_ref, ci_sf, ci_ch, si):
        self.mol = lsi.mol
        self.nmo = mo_coeff.shape[-1]
        self.nfrags = lsi.nfrags
        self.frozen = lsi.frozen
        self._init_orb (lsi, mo_coeff, ci_ref)
        self._init_ci (lsi, ci_ref, ci_sf, ci_ch)
        self._init_si (lsi, si)

    def _init_ci (self, lsi, ci_ref, ci_sf, ci_ch):
        sp = list_spaces (lsi.get_las_of_ci_ref (ci_ref))[0]
        self.ci_ref = copy.deepcopy (ci_ref)
        self.ci_sf = copy.deepcopy (ci_sf)
        self.ci_ch = copy.deepcopy (ci_ch)
        self.t_ref = []
        self.t_sf = []
        self.t_ch_i = []
        self.t_ch_a = []
        for no, na, nb, s in zip (sp.nlas, sp.nelecu, sp.nelecd, sp.smults):
            self.t_ref.append (CSFTransformer (no, na, nb, s))
            # Spin flip
            ti = [None, None]
            if na > nb: ti[0] = CSFTransformer (no, na-1, nb+1, s-2)
            if (nb > 0) and (na < no): ti[1] = CSFTransformer (no, na+1, nb-1, s+2)
            self.t_sf.append (ti)
            # Donating electron
            ti = [None, None]
            if na>nb: ti[0] = CSFTransformer (no, na-1, nb, s-1)
            if nb>0: ti[1] = CSFTransformer (no, na, nb-1, s+1)
            self.t_ch_i.append (ti)
            # Receiving electron
            ti = [None, None]
            if na<no: ti[0] = CSFTransformer (no, na+1, nb, s+1)
            if na>nb: ti[1] = CSFTransformer (no, na, nb+1, s+1)
            self.t_ch_a.append (ti)
        for i in range (self.nfrags):
            c = self.ci_ref[i]
            t = self.t_ref[i]
            self.ci_ref[i] = t.vec_det2csf (c, normalize=False)
            for s in range (2):
                c = self.ci_sf[i][s]
                t = self.t_sf[i][s]
                if c is not None: self.ci_sf[i][s] = t.vec_det2csf (c, normalize=False)
            for a in range (self.nfrags):
                for s in range (4):
                    # p = 0: i
                    c = self.ci_ch[i][a][s][0]
                    t = self.t_ch_i[i][s//2]
                    if c is not None: self.ci_ch[i][a][s][0] = t.vec_det2csf (c, normalize=False)
                    # p = 1: a
                    c = self.ci_ch[i][a][s][1]
                    t = self.t_ch_a[a][s%2]
                    if c is not None: self.ci_ch[i][a][s][1] = t.vec_det2csf (c, normalize=False)

    def _init_si (self, lsi, si):
        self.nprods = lsi.get_nprods ()
        self.si = np.asfortranarray (si.reshape (self.nprods, -1))
        self.nroots_si = self.si.shape[1]

    def pack (self, kappa, ci_ref, ci_sf, ci_ch, si):
        kappa = .5 * (kappa - kappa.T)
        x = kappa[self.uniq_orb_idx]
        # ci_ref part
        for i in range (self.nfrags):
            t = self.t_ref[i]
            c0 = self.ci_ref[i]
            c1 = t.vec_det2csf (ci_ref[i], normalize=False)
            c1 -= c0 * np.dot (c0.conj (), c1)
            x = np.append (x, c1)
        # ci_sf part
        for i in range (self.nfrags):
            for s in range (2):
                t = self.ti_sf[i][s]
                c0 = self.ci_sf[i][s]
                if c0 is not None:
                    c1 = t.vec_det2csf (ci_sf[i][0], normalize=False)
                    c1 -= np.dot (np.dot (c1, c0.conj ().T), c0)
                    x = np.append (x, c1.ravel ())
        # ci_ch part
        for i in range (self.nfrags):
            for a in range (self.nfrags):
                for s in range (4):
                    # p = 0: i
                    c0 = self.ci_ch[i][a][s][0]
                    t = self.t_ch_i[i][s//2]
                    if c0 is not None:
                        c1 = t.vec_det2csf (ci_ch[i][a][s][0], normalize=False)
                        c1 -= np.dot (np.dot (c1, c0.conj ().T), c0)
                        x = np.append (x, c1.ravel ())
                    # p = 1: a
                    c0 = self.ci_ch[i][a][s][1]
                    t = self.t_ch_a[a][s%2]
                    if c0 is not None:
                        c1 = t.vec_det2csf (ci_ch[i][a][s][1], normalize=False)
                        c1 -= np.dot (np.dot (c1, c0.conj ().T), c0)
                        x = np.append (x, c1.ravel ())
        # si part internal
        si = si.reshape (self.nprods, self.nroots_si)
        z = self.si.conj ().T @ si
        if self.nroots_si>1:
            x = np.append (x, (z-z.T)[np.tril_indices (self.nroots_si)])
        # si part external
        si -= self.si @ z
        x = np.append (x, si)
        return x

    def unpack (self, x):
        kappa = np.zeros ((self.nmo, self.nmo), dtype=x.dtype)
        kappa[self.uniq_orb_idx] = x[:self.nvar_orb]
        kappa = kappa - kappa.T
        y = x[self.nvar_orb:]
        ci_ref = []
        # ci_ref part
        for i in range (self.nfrags):
            t = self.t_ref[i]
            ci_ref.append (t.vec_csf2det (y[:t.ncsf], normalize=False))
            y = y[t.ncsf:]
        # ci_sf part
        ci_sf = []
        for i in range (self.nfrags):
            c_i = [None, None]
            for s in range (2):
                t = self.ti_sf[i][s]
                c0 = self.ci_sf[i][s]
                if c0 is not None:
                    c_i[s] = t.vec_csf2det (y[:t.ncsf], normalize=False)
                    y = y[t.ncsf:] 
            ci_sf.append (c_i)
        # ci_ch part
        ci_ch = []
        for i in range (self.nfrags):
            ci = []
            for a in range (self.nfrags):
                cia = [[None, None] for s in range (4)]
                for s in range (4):
                    # p = 0: i
                    c0 = self.ci_ch[i][a][s][0]
                    t = self.t_ch_i[i][s//2]
                    if c0 is not None:
                        cia[s][0] = t.vec_csf2det (y[:t.ncsf], normalize=False)
                        y = y[t.ncsf:] 
                    # p = 1: a
                    c0 = self.ci_ch[i][a][s][1]
                    t = self.t_ch_a[a][s%2]
                    if c0 is not None:
                        cia[s][1] = t.vec_csf2det (y[:t.ncsf], normalize=False)
                        y = y[t.ncsf:] 
                ci.append (cia)
            ci_ch.append (ci)
        # si part internal
        si = np.zeros_like (self.si)
        nz = self.nroots_si * (self.nroots_si - 1) // 2
        if nz:
            z = np.zeros ((self.nroots_si, self.nroots_si), dtype=x.dtype)
            z[np.tril_indices (self.nroots_si)] = y[:nz]
            z = 0.5 * (z - z.T)
            si += self.si @ z
            y = y[nz:]
        # si part external
        si += y.reshape (self.nprods, self.nroots_si)
        return kappa, ci_ref, ci_sf, ci_ch, si

    @property
    def ncsf_ref (self):
        return np.asarray ([t.ncsf for t in self.t_ref])

    @property
    def ncsf_sf (self):
        ncsf = np.zeros ((self.nfrags, 2), dtype=int)
        for i in range (self.nfrags):
            for s in range (2):
                if self.ci_sf[i][s] is not None:
                    ncsf[i][s] = self.t_sf[i][s].ncsf
        return ncsf

    @property
    def ncsf_ch (self):
        ncsf = np.zeros ((self.nfrags, self.nfrags, 4, 2), dtype=int)
        for i in range (self.nfrags):
            for a in range (self.nfrags):
                for s in range (4):
                    # p = 0: i
                    if self.ci_ch[i][a][s][0] is not None:
                        

def _update_mo (mo0, kappa):
    umat = linalg.expm (kappa/2)
    mo1 = mo0 @ umat
    if hasattr (mo0, 'orbsym'):
        mo1 = lib.tag_array (mo1, orbsym=mo0.orbsym)
    return mo1

def _update_civecs (ci0, dci):
    old_shape = ci0.shape
    if ci0.ndim==2:
        nroots=1
    else:
        nroots = ci0.shape[0]
    ci0 = ci0.reshape (nroots,-1)
    dci = dci.reshape (nroots,-1)
    dci -= np.dot (np.dot (dci, ci0.conj ().T), ci0)
    phi = linalg.norm (dc, axis=1)
    cosp = np.cos (phi)
    sinp = np.ones_like (cosp)
    i = np.abs (phi) > 1e-8
    sinp[i] = np.sin (phi[i]) / phi[i]
    ci1 = cosp[:,None]*ci0 + sinp[:,None]*dci
    return ci1.reshape (old_shape)

def _update_sivecs (si0, dsi):
    if si0.ndim==2:
        si0 = si0.T
        dsi = dsi.T
    si1 = _update_civecs (si0, dsi)
    if si1.ndim==2:
        si1 = si1.T
    return si1

def update_ci_ref (lsi, ci0, ci1):
    ci2 = []
    for c0, c1 in zip (ci0, ci1):
        ci2.append (_update_civecs (c0, c1))
    return ci2

def update_ci_sf (lsi, ci0, ci1):
    ci2 = []
    for a0, a1 in zip (ci0, ci1):
        a2 = []
        for b0, b1 in zip (a0, a1):
            a2.append (_update_civecs (b0, b1))
        ci2.append (a2)
    return ci2

def update_ci_ch (lsi, ci0, ci1):
    ci2 = []
    for a0, a1 in zip (ci0, ci1):
        a2 = []
        for b0, b1 in zip (a0, a1):
            b2 = []
            for c0, c1 in zip (b0, b1):
                c2 = []
                for d0, d1 in zip (c0, c1):
                    c2.append (_update_civecs (d0, d1))
                b2.append (c2)
            a2.append (b2)
        ci2.append (a2)
    return ci2

