import unittest
import numpy
from pyscf import gto, dft, lib

# A geometry step through the GPU DF J/K: SCF at G0, then the gradient scanner at G1 (what geomeTRIC does),
# against a fresh CPU calculation at G1. libgpu caches the 3-center integrals by id(dfobj); the scanner keeps the
# DF object, so without a reload exact exchange keeps the G0 integrals (|dE| ~ 2.5 Ha after one step).

G0 = "O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59"
G1 = "O 0 0 0.05; H 0 0.80 0.55; H 0 -0.72 0.63"
CASES = [("pbe", 0), ("pbe0", 0), ("wb97m-v", 0), ("pbe0", 1)]   # (xc, spin); spin 1 = ROKS cation

def _make (atom, xc, spin, basis='cc-pvdz'):
    mol = gto.M(atom=atom, basis=basis, charge=spin, spin=spin, verbose=0)
    mf = (dft.RKS(mol) if spin == 0 else dft.ROKS(mol)).density_fit()
    mf.xc = xc
    mf.conv_tol = 1e-11
    mf.grids.level = 2
    mf.nlcgrids.level = 1
    return mf

def _reference (xc, spin):
    mf = _make(G1, xc, spin)
    e = mf.kernel()
    return e, mf.nuc_grad_method().kernel()

def _scanned (xc, spin):
    mf = _make(G0, xc, spin)
    mf.kernel()
    scan = mf.nuc_grad_method().as_scanner()
    mol1 = mf.mol.set_geom_(gto.M(atom=G1, unit="Angstrom").atom_coords(), unit="Bohr", inplace=False)
    return scan(mol1)

def setUpModule():
    global refs, ref_tz
    lib.param.use_gpu = None
    refs = {c: _reference(*c) for c in CASES}
    ref_tz = _make(G1, 'pbe0', 0, basis='cc-pvtz').kernel()

def tearDownModule():
    global refs, ref_tz
    del refs, ref_tz

class KnownValues (unittest.TestCase):

    @classmethod
    def setUpClass (cls):
        from mrh.my_pyscf.gpu import libgpu
        from gpu4mrh import patch_pyscf
        cls.libgpu = libgpu
        cls.gpu = libgpu.init()

    def setUp (self):
        lib.param.use_gpu = self.gpu   # tests/gpu/conftest.py resets it after every test

    @classmethod
    def tearDownClass (cls):
        cls.libgpu.destroy_device(cls.gpu)
        lib.param.use_gpu = None

    def test_scanner_step (self):
        for c in CASES:
            e_ref, g_ref = refs[c]
            e, g = _scanned(*c)
            with self.subTest (xc=c[0], spin=c[1]):
                self.assertAlmostEqual (e, e_ref, 9)
                self.assertLess (numpy.abs(g - g_ref).max(), 1e-6)

    def test_same_dfobj_new_basis (self):
        # cc-pVDZ -> cc-pVTZ on the same DF object (same id): the cached blocks change shape
        mf = _make(G0, 'pbe0', 0)
        mf.kernel()
        mf_tz = _make(G1, 'pbe0', 0, basis='cc-pvtz')
        mf_tz.with_df = mf.with_df
        mf_tz.with_df.reset(mf_tz.mol)
        self.assertAlmostEqual (mf_tz.kernel(), ref_tz, 9)

if __name__ == "__main__":
    print("Tests for GPU DF J/K after a geometry change")
    unittest.main()
