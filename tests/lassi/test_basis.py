import unittest
import numpy as np
from scipy import linalg
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
from mrh.my_pyscf.lassi import op_o0, op_o1, basis
from mrh.my_pyscf.fci.spin_op import mdown
from pyscf import lib, gto
from pyscf.csf_fci.csfstring import CSFTransformer
import itertools

op = (op_o0, op_o1)

def setUpModule():
    global rng, orth_bases, ham_raw, h_op_raw, hdiag_raw
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
    ovlp0 = op_o0.get_ovlp (ci, norb_f, nelec_frs)
    raw2orth_fullspin = basis.get_orth_basis (ci, norb_f, nelec_frs, smult_fr=smult_fr)
    raw2orth_nospin = basis.get_orth_basis (ci, norb_f, nelec_frs)
    ci1 = [[ci_ij for ci_ij in ci_i] for ci_i in ci]
    for i in range (2):
        for j in range (2+4*i,4+4*i):
            ci1[i][j] = t3.vec_csf2det (random_orthrows (2, t3.ncsf))
            ci1[i][j] = ci1[i][j].reshape (-1, t3.ndeta, t3.ndetb)
    ovlp1 = op_o0.get_ovlp (ci1, norb_f, nelec_frs)
    raw2orth_semispin = basis.get_orth_basis (ci1, norb_f, nelec_frs, smult_fr=smult_fr)
    orth_bases = {'no spin': (ovlp0, raw2orth_nospin),
                  'semi-spin': (ovlp1, raw2orth_semispin),
                  'full spin': (ovlp0, raw2orth_fullspin)}
    mol = gto.M (verbose=0, output='/dev/null')
    las = LASSCF (mol, norb_f, norb_f)
    rng = np.random.default_rng ()
    h1 = 1 - (2*rng.random ((8,8)))
    h1 += h1.T
    h2 = 1 - (2*rng.random ((8,8,8,8)))
    h2 += h2.transpose (1,0,2,3)
    h2 += h2.transpose (0,1,3,2)
    h2 += h2.transpose (2,3,0,1)
    ham_raw = op[1].ham (las, h1, h2, ci, nelec_frs, smult_fr=smult_fr)[0]
    ops = [op[opt].gen_contract_op_si_hdiag (las, h1, h2, ci, nelec_frs, smult_fr=smult_fr)
           for opt in (0,1)]
    h_op_raw = [ops[0][0], ops[1][0]]
    hdiag_raw = [ops[0][3], ops[1][3]]

def tearDownModule():
    global rng, orth_bases, ham_raw, h_op_raw, hdiag_raw
    del rng, orth_bases, ham_raw, h_op_raw, hdiag_raw

def random_orthrows (nrows, ncols):
    x = 2 * np.random.rand (nrows, ncols) - 1
    Q = linalg.orth (x)
    x = Q.T @ x
    x /= linalg.norm (x, axis=1)[:,None]
    return x

def possible_smults (smult_f, spin):
    s2_f = smult_f-1
    s2_f = s2_f[s2_f!=0]
    if len (s2_f) == 0: return np.arange (1,2,dtype=int)
    s2_f = np.sort (s2_f)[::-1]
    smult_min = max (2*s2_f[0] - s2_f.sum (), abs (spin)) + 1
    smult_max = (smult_f-1).sum () + 1
    return np.arange (smult_min, smult_max+1, 2, dtype=int)

class KnownValues(unittest.TestCase):

    def test_get_orth_basis_size (self):    
        # Exact number may change if there are sometimes lindeps
        self.assertLess (orth_bases['full spin'][1].get_nbytes (),
                         orth_bases['semi-spin'][1].get_nbytes ())
        self.assertLess (orth_bases['semi-spin'][1].get_nbytes (),
                         orth_bases['no spin'][1].get_nbytes ())

    def test_get_orth_basis (self):
        for lbl, (ovlp, raw2orth) in orth_bases.items ():
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

    def case_hdiag_orth (self, raw2orth, opt):
        ham_orth = raw2orth (ham_raw.T).T
        ham_orth = raw2orth (ham_orth.conj ()).conj ()
        hdiag_orth_ref = ham_orth.diagonal ()
        hdiag_orth_test = op[opt].get_hdiag_orth (hdiag_raw[opt], h_op_raw[opt], raw2orth)
        self.assertAlmostEqual (lib.fp (hdiag_orth_ref), lib.fp (hdiag_orth_test), 8)

    def case_pspace_ham (self, raw2orth, opt):
        ham_orth = raw2orth (ham_raw.T).T
        ham_orth = raw2orth (ham_orth.conj ()).conj ()
        addrs = rng.choice (raw2orth.shape[0], 5, replace=False)
        ham_ref = ham_orth[addrs,:][:,addrs]
        ham_test = op[opt].pspace_ham (h_op_raw[opt], raw2orth, addrs)
        self.assertAlmostEqual (lib.fp (ham_test), lib.fp (ham_ref), 8)

    def test_hdiag_orth_nospin_o0 (self):
        self.case_hdiag_orth (orth_bases['no spin'][1], 0)

    def test_hdiag_orth_fullspin_o0 (self):
        self.case_hdiag_orth (orth_bases['full spin'][1], 0)

    def test_hdiag_orth_nospin_o1 (self):
        self.case_hdiag_orth (orth_bases['no spin'][1], 1)

    def test_hdiag_orth_fullspin_o1 (self):
        self.case_hdiag_orth (orth_bases['full spin'][1], 1)

    def test_pspace_ham_fullspin_o0 (self):
        self.case_pspace_ham (orth_bases['full spin'][1], 0)

    def test_pspace_ham_fullspin_o1 (self):
        self.case_pspace_ham (orth_bases['full spin'][1], 1)

    def test_pspace_ham_nospin_o0 (self):
        self.case_pspace_ham (orth_bases['no spin'][1], 0)

    def test_pspace_ham_nospin_o1 (self):
        self.case_pspace_ham (orth_bases['no spin'][1], 1)

    def iterate_spins (self, max_nfrag, max_smult_f, max_smult=None):
        for nfrag in range (1,max_nfrag+1):
            for smults in itertools.product (range (1,max_smult_f+1), repeat=nfrag):
                smult_f = np.asarray (smults)
                max_spin1 = (smult_f-1).sum ()
                if max_smult is None:
                    max_spin = max_spin1
                else:
                    max_spin = max_smult - 1
                if ((max_spin1-max_spin)%2) != 0:
                    max_spin -= 1
                assert (max_spin >= 0)
                for spin in range (-max_spin, max_spin+1, 2):
                    with self.subTest (smult_f=smult_f, spin=spin):
                        yield smult_f, spin

    def check_s2mat (self, s2mat, smult_f, spin):
        with self.subTest ('check_s2mat'):
            evals = linalg.eigh (s2mat)[0]
            idx = np.zeros (evals.shape[0], dtype=bool)
            for smult in possible_smults (smult_f, spin):
                s = (smult-1) * .5
                s2eval = s * (s + 1)
                idx[np.isclose (evals, s2eval)] = True
            self.assertTrue (np.all (idx))

    def check_umat (self, umat):
        with self.subTest ('check_umat'):
            ovlp = umat.conj ().T @ umat
            ref = np.eye (umat.shape[1])
            self.assertAlmostEqual (lib.fp (ovlp), lib.fp (ref), 7)

    def check_umat_s2mat (self, umat, s2mat, smult):
        with self.subTest ('check_umat_s2mat'):
            s = (smult-1) / 2
            s2eval = s * (s + 1)
            ref = np.eye (umat.shape[1]) * s2eval
            test = umat.conj ().T @ s2mat @ umat
            self.assertAlmostEqual (lib.fp (test), lib.fp (ref), 7)

    def test_spincoup (self):
        for smult_f, spin in self.iterate_spins (4, 4):
            s2mat = basis.make_s2mat (smult_f, spin)
            self.check_s2mat (s2mat, smult_f, spin)
            for smult in possible_smults (smult_f, spin):
                with self.subTest ('umat', smult_lsf=smult):
                    umat = basis.get_spincoup_umat (smult_f, spin, smult)
                    self.check_umat (umat)
                    self.check_umat_s2mat (umat, s2mat, smult)

if __name__ == "__main__":
    print("Full Tests for LASSI basis module functions")
    unittest.main()


