import numpy as np
from scipy import linalg
from pyscf import lib
import copy
from mrh.my_pyscf.mcscf import lasscf_sync_o0
from mrh.my_pyscf.lassi import lassis, op_o0, op_o1
from mrh.my_pyscf.lassi.spaces import list_spaces
from mrh.my_pyscf.fci.csfstring import CSFTransformer
from mrh.my_pyscf.fci.spin_op import mup

op = (op_o0, op_o1)

class UnitaryGroupGenerators (lasscf_sync_o0.LASSCF_UnitaryGroupGenerators):
    def __init__(self, lsi, mo_coeff, ci_ref, ci_sf, ci_ch, si):
        self.mol = lsi.mol
        self.lsi = lsi
        self.nmo = mo_coeff.shape[-1]
        self.nfrags = lsi.nfrags
        self.frozen = None
        self.mo_coeff = mo_coeff
        self._init_orb (lsi, mo_coeff, ci_ref)
        self._init_ci (lsi, ci_ref, ci_sf, ci_ch)
        self._init_si (lsi, ci_ref, ci_sf, ci_ch, si)

    def _init_ci (self, lsi, ci_ref, ci_sf, ci_ch):
        sp = list_spaces (lsi.get_las_of_ci_ref (ci_ref))[0]
        self.ci_ref = copy.deepcopy (ci_ref)
        self.ci_sf = copy.deepcopy (ci_sf)
        self.ci_ch = copy.deepcopy (ci_ch)
        self.ci = self.lsi.prepare_model_states (ci_ref, ci_sf, ci_ch)[0].ci
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
            if na>nb: ti[0] = CSFTransformer (no, na, nb+1, s-1)
            if na<no: ti[1] = CSFTransformer (no, na+1, nb, s+1)
            self.t_ch_a.append (ti)
        for i in range (self.nfrags):
            c = self.ci_ref[i]
            t = self.t_ref[i]
            self.ci_ref[i] = t.vec_det2csf (c, normalize=False)[0]
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

    def _init_si (self, lsi, ci_ref, ci_sf, ci_ch, si):
        self.nprods = lsi.get_nprods ()
        si = np.asfortranarray (si.reshape (self.nprods, -1))
        self.nroots_si = si.shape[1]
        self.raw2orth = lsi.get_raw2orth (ci_ref=ci_ref, ci_sf=ci_sf, ci_ch=ci_ch)
        self.north = self.raw2orth.shape[0]
        self.si = self.raw2orth (si)

    def pack (self, kappa, ci_ref, ci_sf, ci_ch, si):
        kappa = .5 * (kappa - kappa.T)
        x = kappa[self.uniq_orb_idx]
        # ci_ref part
        for i in range (self.nfrags):
            t = self.t_ref[i]
            c0 = self.ci_ref[i]
            c1 = t.vec_det2csf (ci_ref[i], normalize=False)[0]
            c1 -= c0 * np.dot (c0.conj (), c1)
            x = np.append (x, c1)
        # ci_sf part
        ncsf_sf = self.ncsf_sf
        for i in range (self.nfrags):
            for s in range (2):
                t = self.t_sf[i][s]
                c0 = self.ci_sf[i][s]
                if c0 is not None and (ncsf_sf[i,s,0]<ncsf_sf[i,s,1]):
                    c1 = t.vec_det2csf (ci_sf[i][s], normalize=False)
                    c1 -= np.dot (np.dot (c1, c0.conj ().T), c0)
                    x = np.append (x, c1.ravel ())
        # ci_ch part
        ncsf_ch = self.ncsf_ch
        for i in range (self.nfrags):
            for a in range (self.nfrags):
                for s in range (4):
                    # p = 0: i
                    c0 = self.ci_ch[i][a][s][0]
                    t = self.t_ch_i[i][s//2]
                    if c0 is not None and (ncsf_ch[i,a,s,0,0]<ncsf_ch[i,a,s,0,1]):
                        c1 = t.vec_det2csf (ci_ch[i][a][s][0], normalize=False)
                        c1 -= np.dot (np.dot (c1, c0.conj ().T), c0)
                        x = np.append (x, c1.ravel ())
                    # p = 1: a
                    c0 = self.ci_ch[i][a][s][1]
                    t = self.t_ch_a[a][s%2]
                    if c0 is not None and (ncsf_ch[i,a,s,1,0]<ncsf_ch[i,a,s,1,1]):
                        c1 = t.vec_det2csf (ci_ch[i][a][s][1], normalize=False)
                        c1 -= np.dot (np.dot (c1, c0.conj ().T), c0)
                        x = np.append (x, c1.ravel ())
        # si part internal
        si = self.raw2orth (si.reshape (self.nprods, self.nroots_si))
        z = self.si.conj ().T @ si
        if self.nroots_si>1:
            x = np.append (x, (z-z.T)[np.tril_indices (self.nroots_si, k=-1)])
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
            ci_ref[-1] = ci_ref[-1].reshape (t.ndeta, t.ndetb)
            y = y[t.ncsf:]
        # ci_sf part
        ci_sf = [[None for s in range (2)] for i in range (self.nfrags)]
        ncsf_sf = self.ncsf_sf
        for i in range (self.nfrags):
            for s in range (2):
                t = self.t_sf[i][s]
                c0 = self.ci_sf[i][s]
                if c0 is not None and (ncsf_sf[i,s,0]<ncsf_sf[i,s,1]):
                    ci_sf[i][s] = t.vec_csf2det (y[:c0.size].reshape (c0.shape), normalize=False)
                    ci_sf[i][s] = ci_sf[i][s].reshape (-1, t.ndeta, t.ndetb)
                    y = y[c0.size:]
                elif c0 is not None:
                    ci_sf[i][s] = np.zeros ((ncsf_sf[i,s,0],t.ndeta,t.ndetb), dtype=c0.dtype)
        # ci_ch part
        ci_ch = []
        ncsf_ch = self.ncsf_ch
        for i in range (self.nfrags):
            ci = []
            for a in range (self.nfrags):
                cia = [[None, None] for s in range (4)]
                for s in range (4):
                    # p = 0: i
                    c0 = self.ci_ch[i][a][s][0]
                    t = self.t_ch_i[i][s//2]
                    if c0 is not None and (ncsf_ch[i,a,s,0,0]<ncsf_ch[i,a,s,0,1]):
                        cia[s][0] = t.vec_csf2det (y[:c0.size].reshape (c0.shape), normalize=False)
                        cia[s][0] = cia[s][0].reshape (-1, t.ndeta, t.ndetb)
                        y = y[c0.size:] 
                    elif c0 is not None:
                        cia[s][0] = np.zeros ((ncsf_ch[i,a,s,0,0],t.ndeta,t.ndetb), dtype=c0.dtype)
                    # p = 1: a
                    c0 = self.ci_ch[i][a][s][1]
                    t = self.t_ch_a[a][s%2]
                    if c0 is not None and (ncsf_ch[i,a,s,1,0]<ncsf_ch[i,a,s,1,1]):
                        cia[s][1] = t.vec_csf2det (y[:c0.size].reshape (c0.shape), normalize=False)
                        cia[s][1] = cia[s][1].reshape (-1, t.ndeta, t.ndetb)
                        y = y[c0.size:] 
                    elif c0 is not None:
                        cia[s][1] = np.zeros ((ncsf_ch[i,a,s,0,0],t.ndeta,t.ndetb), dtype=c0.dtype)
                ci.append (cia)
            ci_ch.append (ci)
        # si part internal
        si = np.zeros_like (self.si)
        nz = self.nroots_si * (self.nroots_si - 1) // 2
        if nz:
            z = np.zeros ((self.nroots_si, self.nroots_si), dtype=x.dtype)
            z[np.tril_indices (self.nroots_si, k=-1)] = y[:nz]
            z = 0.5 * (z - z.T)
            si += self.si @ z
            y = y[nz:]
        # si part external
        si += y.reshape (self.north, self.nroots_si)
        si = self.raw2orth.H (si)
        return kappa, ci_ref, ci_sf, ci_ch, si

    @property
    def ncsf_ref (self):
        return np.asarray ([t.ncsf for t in self.t_ref])

    @property
    def ncsf_sf (self):
        ncsf = np.zeros ((self.nfrags, 2, 2), dtype=int)
        for i in range (self.nfrags):
            for s in range (2):
                if self.ci_sf[i][s] is not None:
                    ncsf[i,s,:] = self.ci_sf[i][s].shape
        return ncsf

    @property
    def ncsf_ch (self):
        ncsf = np.zeros ((self.nfrags, self.nfrags, 4, 2, 2), dtype=int)
        for i in range (self.nfrags):
            for a in range (self.nfrags):
                for s in range (4):
                    # p = 0: i
                    for p in range (2):
                        c = self.ci_ch[i][a][s][0]
                        if c is not None: ncsf[i,a,s,p,:] = c.shape
        return ncsf

    def sum_ncsf (self, ncsf):
        ncsf = ncsf.reshape (-1,2)
        ncsf[ncsf[:,0]==ncsf[:,1],:] = 0
        return np.dot (ncsf[:,0], ncsf[:,1])

    @property
    def nvar_csf_ref (self): return self.ncsf_ref.sum ()

    @property
    def nvar_csf_sf (self): return self.sum_ncsf (self.ncsf_sf)

    @property
    def nvar_csf_ch (self): return self.sum_ncsf (self.ncsf_ch)

    @property
    def nvar_si (self):
        nz = self.nroots_si * (self.nroots_si - 1) // 2
        return nz, self.north*self.nroots_si

    @property
    def ncsf_all (self):
        return self.ncsf_ref.sum () + self.nvar_csf_sf + self.nvar_csf_ch

    @property
    def nvar_tot (self):
        return self.nvar_orb + self.ncsf_all + sum (self.nvar_si)

    def get_sector_offsets (self):
        lens = [self.nvar_orb, self.ncsf_ref.sum (), self.nvar_csf_sf, self.nvar_csf_ch]
        lens += list (self.nvar_si)
        lens = np.asarray (lens)
        offs = np.cumsum (lens)
        offs = np.stack ((offs-lens, offs), axis=1)
        return offs

    def update_wfn (self, x):
        kappa, dcir, dcis, dcic, dsi = self.unpack (x)
        mo1 = _update_mo (self.mo_coeff, kappa)
        ci_ref = [t.vec_csf2det (c).reshape (t.ndeta, t.ndetb)
                  for c, t in zip (self.ci_ref, self.t_ref)]
        ci_sf = [[None,None] for i in range (self.nfrags)]
        ci_ch = [[[[None,None] for s in range (4)]
                  for a in range (self.nfrags)]
                 for i in range (self.nfrags)]
        for i in range (self.nfrags):
            for s in range (2):
                t = self.t_sf[i][s]
                c = self.ci_sf[i][s]
                if c is not None: ci_sf[i][s] = t.vec_csf2det (c).reshape (-1,t.ndeta,t.ndetb)
            for a in range (self.nfrags):
                for s in range (4):
                    # p = 0: i
                    t = self.t_ch_i[i][s//2]
                    c = self.ci_ch[i][a][s][0]
                    if c is not None:
                        ci_ch[i][a][s][0] = t.vec_csf2det (c).reshape (-1,t.ndeta,t.ndetb)
                    # p = 1: a
                    t = self.t_ch_a[a][s%2]
                    c = self.ci_ch[i][a][s][1]
                    if c is not None:
                        ci_ch[i][a][s][1] = t.vec_csf2det (c).reshape (-1,t.ndeta,t.ndetb)
        ci_ref = _update_ci_ref (ci_ref, dcir)
        ci_sf = _update_ci_sf (ci_sf, dcis)
        ci_ch = _update_ci_ch (ci_ch, dcic)
        si = self.raw2orth.H (self.si)
        si = _update_sivecs (si, dsi)
        return mo1, ci_ref, ci_sf, ci_ch, si


def _update_mo (mo0, kappa):
    umat = linalg.expm (kappa/2)
    mo1 = mo0 @ umat
    if hasattr (mo0, 'orbsym'):
        mo1 = lib.tag_array (mo1, orbsym=mo0.orbsym)
    return mo1

def _update_civecs (ci0, dci):
    if ci0 is None and dci is None: return None
    old_shape = ci0.shape
    if ci0.ndim==2:
        nroots=1
    else:
        nroots = ci0.shape[0]
    ci0 = ci0.reshape (nroots,-1)
    dci = dci.reshape (nroots,-1)
    dci -= np.dot (np.dot (dci, ci0.conj ().T), ci0)
    phi = linalg.norm (dci, axis=1)
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

def _update_ci_ref (ci0, ci1):
    ci2 = []
    for c0, c1 in zip (ci0, ci1):
        ci2.append (_update_civecs (c0, c1))
    return ci2

def _update_ci_sf (ci0, ci1):
    ci2 = []
    for a0, a1 in zip (ci0, ci1):
        a2 = []
        for b0, b1 in zip (a0, a1):
            a2.append (_update_civecs (b0, b1))
        ci2.append (a2)
    return ci2

def _update_ci_ch (ci0, ci1):
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

def sum_hci (lsi, hci_fr):
    '''Add hci vectors into LASSIS-shape arrays

    Args:
        lsi: object of `class`:LASSIS:
        hci_fr: list of length nfrags of lists of length nroots of ndarrays
            Contains <E(k)_p,a,b|H|Psi> where |E(k)_p> = <k_p|Psi>

    Returns:
        hci_ref: list of length nfrags of ndarrays
        hci_sf: nested list of shape (nfrags,2) of ndarrays
        hci_ch: nested list of shape (nfrags,nfrags,4,2) of ndarrays
    '''
    no = lsi.ncas_sub
    smult = lsi.get_smult_fr ()
    hci_ref = [0,]*lsi.nfrags
    hci_sf = [[0,0] for i in range (lsi.nfrags)]
    hci_ch = [[[[0,0] for s in range (4)]
               for a in range (lsi.nfrags)]
              for i in range (lsi.nfrags)]
    for i in range (lsi.nfrags):
        for p, ne in zip (*lsi.get_ref_fbf_rootspaces (i)):
            hci_ref[i] += mup (hci_fr[i][p], no[i], ne, smult[i][p])
        for s in range (2):
            for p, ne in zip (*lsi.get_sf_fbf_rootspaces (i,s)):
                hci_sf[i][s] += mup (hci_fr[i][p], no[i], ne, smult[i][p])
        for a in range (lsi.nfrags):
            for s in range (4):
                for p, nei, nea in zip (*lsi.get_ch_fbf_rootspaces (i, a, s)):
                    hci_ch[i][a][s][0] += mup (hci_fr[i][p], no[i], nei, smult[i][p])
                    hci_ch[i][a][s][1] += mup (hci_fr[a][p], no[a], nea, smult[a][p])
    return hci_ref, hci_sf, hci_ch


