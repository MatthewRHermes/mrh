import copy
import unittest
import numpy as np
from scipy import linalg
from mrh.my_pyscf.lassi import citools, op_o0
from mrh.my_pyscf.fci.spin_op import mdown
from pyscf import lib
from pyscf.scf.addons import canonical_orth_
from pyscf.csf_fci.csfstring import CSFTransformer
import itertools

def setUpModule():
    pass

def tearDownModule():
    pass

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

def random_orthrows (nrows, ncols):
    x = 2 * np.random.rand (nrows, ncols) - 1
    Q = linalg.orth (x)
    x = Q.T @ x
    x /= linalg.norm (x, axis=1)[:,None]
    return x

class KnownValues(unittest.TestCase):

    def test_umat_dot_1frag (self):
        rng = np.random.default_rng ()
        nroots, nfrags, nvecs = tuple (rng.integers (1, high=6, size=3))
        for nroots, nfrags, nvecs in itertools.product (range (1,6), repeat=3):
            lroots = rng.integers (1, high=10, size=(nfrags,nroots))
            with self.subTest (nroots=nroots, nfrags=nfrags, nvecs=nvecs, lroots=lroots):
                case_umat_dot_1frag (self, rng, nroots, nfrags, nvecs, lroots)

    def test_get_orth_basis (self):
        norb_f = np.array ([4,4])
        nelec_frs = np.array ([[[2,2],[2,2],[3,1],[3,1],[2,2],[2,2],[1,3],[1,3]],
                               [[2,2],[2,2],[1,3],[1,3],[2,2],[2,2],[3,1],[3,1]]])
        smult_fr = np.array ([[1,1,3,3,3,3,3,3],]*2)
        ci = [[None for i in range (8)] for j in range (2)]
        ci_trip = [[None for i in range (2)] for j in range (2)]
        t1 = CSFTransformer (4, 2, 2, 1)
        t3 = CSFTransformer (4, 3, 1, 3)
        for i in range (2):
            for j in range (2):
                ci[i][j] = t1.vec_csf2det (random_orthrows (2, t1.ncsf))
                ci[i][j] = ci[i][j].reshape (-1, t1.ndeta, t1.ndetb)
                ci_trip[i][j] = t3.vec_csf2det (random_orthrows (2, t3.ncsf))
                ci_trip[i][j] = ci_trip[i][j].reshape (-1, t3.ndeta, t3.ndetb)
        ci[0][2:4] = ci_trip[0]
        ci[0][4:6] = [mdown (x, 4, (2, 2), 3) for x in ci_trip[0]]
        ci[0][6:8] = [mdown (x, 4, (1, 3), 3) for x in ci_trip[0]]
        ci[1][2:4] = [mdown (x, 4, (1, 3), 3) for x in ci_trip[1]]
        ci[1][4:6] = [mdown (x, 4, (2, 2), 3) for x in ci_trip[1]]
        ci[1][6:8] = ci_trip[1]
        ovlp = op_o0.get_ovlp (ci, norb_f, nelec_frs)
        raw2orth_spin = citools.get_orth_basis (ci, norb_f, nelec_frs, smult_fr=smult_fr)
        raw2orth_nospin = citools.get_orth_basis (ci, norb_f, nelec_frs)
        with self.subTest ('size'):
            # Exact number may change if there are sometimes lindeps
            self.assertLess (raw2orth_spin.get_nbytes (), raw2orth_nospin.get_nbytes ())
        for raw2orth, lbl in ((raw2orth_spin, 'using smult'), (raw2orth_spin, 'not using smult')):
            with self.subTest (lbl + ' orth'):
                xox = raw2orth (raw2orth (ovlp.T).T)
                xox -= np.eye (raw2orth.shape[0])
                err = np.amax (np.abs (xox))
                self.assertLess (err, 1e-8)
            with self.subTest (lbl + ' reverse'):
                o1 = 2 * np.random.rand (raw2orth.shape[0]) - 1
                o1 /= linalg.norm (o1)
                r1 = raw2orth.H (o1)
                self.assertAlmostEqual (r1.dot (ovlp @ r1), 1.0)
                # TODO: understand why I can't go back and forth

if __name__ == "__main__":
    print("Full Tests for LASSI citools module functions")
    unittest.main()


