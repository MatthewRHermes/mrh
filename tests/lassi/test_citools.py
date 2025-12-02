import copy
import unittest
import numpy as np
from scipy import linalg
from mrh.my_pyscf.lassi import citools
from pyscf import lib
from pyscf.scf.addons import canonical_orth_
import itertools

def setUpModule():
    global rng
    rng = np.random.default_rng ()

def tearDownModule():
    global rng
    del rng
    
def case_umat_dot_1frag (ks, rng, nroots, nfrags, nvecs, lroots):
    nstates = np.prod (lroots, axis=0).sum ()
    if nvecs > nstates: return
    si = np.empty ((0,0))
    for i in range (100):
        si = rng.random (size=(nstates,nvecs))
        si = si @ canonical_orth_(si.conj ().T @ si)
        if si.shape[1] == nvecs: break
    ks.assertEqual (si.shape[1], nvecs)
    si0 = si.copy ()
    UUsi = si.copy ()
    siT = si.T.copy ()
    UUsiT = siT.copy ()
    for ifrag, iroot in itertools.product (range (nfrags), range (nroots)):
        lr = lroots[ifrag,iroot]
        umat = linalg.qr (rng.random (size=(lr,lr)))[0]
        #umat = umat @ canonical_orth_(umat.conj ().T @ umat)
        ks.assertEqual (umat.shape[1], lr)
        ks.assertAlmostEqual (lib.fp (umat.conj ().T @ umat), lib.fp (np.eye (lr)))
        ks.assertAlmostEqual (lib.fp (umat @ umat.conj ().T), lib.fp (np.eye (lr)))
        si = citools.umat_dot_1frag_(si, umat, lroots, ifrag, iroot, axis=0)
        UUsi = citools.umat_dot_1frag_(UUsi, umat, lroots, ifrag, iroot, axis=0)
        UUsi = citools.umat_dot_1frag_(UUsi, umat.conj ().T, lroots, ifrag, iroot, axis=0)
        siT = citools.umat_dot_1frag_(siT, umat, lroots, ifrag, iroot, axis=1)
        UUsiT = citools.umat_dot_1frag_(UUsiT, umat, lroots, ifrag, iroot, axis=1)
        UUsiT = citools.umat_dot_1frag_(UUsiT, umat.conj ().T, lroots, ifrag, iroot, axis=1)
    ovlp = si.conj ().T @ si
    ovlpT = siT.conj () @ siT.T
    ks.assertAlmostEqual (lib.fp (ovlp), lib.fp (np.eye (nvecs)))
    ks.assertAlmostEqual (lib.fp (ovlpT), lib.fp (np.eye (nvecs)))
    ks.assertAlmostEqual (lib.fp (UUsi), lib.fp (si0))
    ks.assertAlmostEqual (lib.fp (UUsiT), lib.fp (si0.conj ().T))
    return

class KnownValues(unittest.TestCase):

    def test_umat_dot_1frag (self):
        nroots, nfrags, nvecs = tuple (rng.integers (1, high=6, size=3))
        for nroots, nfrags, nvecs in itertools.product (range (1,6), repeat=3):
            lroots = rng.integers (1, high=10, size=(nfrags,nroots))
            with self.subTest (nroots=nroots, nfrags=nfrags, nvecs=nvecs, lroots=lroots):
                case_umat_dot_1frag (self, rng, nroots, nfrags, nvecs, lroots)

if __name__ == "__main__":
    print("Full Tests for LASSI citools module functions")
    unittest.main()


