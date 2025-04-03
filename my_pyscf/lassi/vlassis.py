import numpy as np
from scipy import linalg
from pyscf import lib

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

