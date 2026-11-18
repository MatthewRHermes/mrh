import gc
import unittest
import numpy
from pyscf import gto, scf, dft, lib

# A geometry step through the GPU DF J/K: SCF at G0, then the gradient scanner at G1 (what geomeTRIC does),
# against a fresh CPU calculation at G1. libgpu caches the 3-center integrals by id(dfobj); the scanner keeps the
# DF object, so without a reload exact exchange keeps the G0 integrals (|dE| ~ 2.5 Ha after one step).

G0 = "O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59"
G1 = "O 0 0 0.05; H 0 0.80 0.55; H 0 -0.72 0.63"
CASES = [("pbe", 0), ("pbe0", 0), ("wb97m-v", 0), ("pbe0", 1)]   # (xc, spin); spin 1 = ROKS cation

def _make (atom, xc, spin, basis='cc-pvdz', use_gpu=None):
    # the device handle travels on the Mole (never lib.param.use_gpu); the CPU references are built
    # before gpu4mrh patches gto.M, so the keyword is only passed when there is a handle
    kwargs = {} if use_gpu is None else {'use_gpu': use_gpu}
    mol = gto.M(atom=atom, basis=basis, charge=spin, spin=spin, verbose=0, **kwargs)
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

def _scanned (xc, spin, use_gpu):
    mf = _make(G0, xc, spin, use_gpu=use_gpu)
    mf.kernel()
    scan = mf.nuc_grad_method().as_scanner()
    mol1 = mf.mol.set_geom_(gto.M(atom=G1, unit="Angstrom").atom_coords(), unit="Bohr", inplace=False)
    return scan(mol1)

def _hf (atom, use_gpu=None):
    kwargs = {} if use_gpu is None else {'use_gpu': use_gpu}
    mf = scf.ROHF(gto.M(atom=atom, basis='cc-pvdz', spin=1, verbose=0, **kwargs)).density_fit()
    mf.conv_tol = 1e-10
    return mf

def _nblk_cached (gpu, addr):
    arg = numpy.array([-1, -1, -1, -1], dtype=numpy.int32)
    libgpu.get_dfobj_status(gpu, addr, arg)
    return arg[2] # -1: nothing cached for addr

GOH = ["O 0 0 0; H 0 0.76 0.59", "O 0 0 0.05; H 0 0.80 0.55"]

def setUpModule():
    global refs, ref_tz, ref_oh
    assert getattr(lib.param, 'use_gpu', None) is None
    refs = {c: _reference(*c) for c in CASES}
    ref_tz = _make(G1, 'pbe0', 0, basis='cc-pvtz').kernel()
    ref_oh = [_hf(g).kernel() for g in GOH]

def tearDownModule():
    global refs, ref_tz, ref_oh
    del refs, ref_tz, ref_oh

class KnownValues (unittest.TestCase):

    @classmethod
    def setUpClass (cls):
        global libgpu
        from mrh.my_pyscf.gpu import libgpu
        from gpu4mrh import patch_pyscf
        cls.gpu = libgpu.init()

    @classmethod
    def tearDownClass (cls):
        libgpu.destroy_device(cls.gpu)

    def test_scanner_step (self):
        for c in CASES:
            e_ref, g_ref = refs[c]
            e, g = _scanned(*c, self.gpu)
            with self.subTest (xc=c[0], spin=c[1]):
                self.assertAlmostEqual (e, e_ref, 9)
                self.assertLess (numpy.abs(g - g_ref).max(), 1e-6)

    def test_same_dfobj_new_basis (self):
        # cc-pVDZ -> cc-pVTZ on the same DF object (same id): the cached blocks change shape
        mf = _make(G0, 'pbe0', 0, use_gpu=self.gpu)
        mf.kernel()
        mf_tz = _make(G1, 'pbe0', 0, basis='cc-pvtz', use_gpu=self.gpu)
        mf_tz.with_df = mf.with_df
        mf_tz.with_df.reset(mf_tz.mol)
        self.assertAlmostEqual (mf_tz.kernel(), ref_tz, 9)

    def test_reset_evicts (self):
        mf = _hf(GOH[0], use_gpu=self.gpu)
        mf.kernel()
        addr = id(mf.with_df)
        self.assertGreater (_nblk_cached(self.gpu, addr), 0)
        mf.with_df.reset()
        self.assertEqual (_nblk_cached(self.gpu, addr), -1)

    def test_gc_evicts (self):
        # the blocks of a garbage-collected DF object must not outlive it: its id() can be reused
        mf = _hf(GOH[0], use_gpu=self.gpu)
        mf.kernel()
        addr = id(mf.with_df)
        self.assertGreater (_nblk_cached(self.gpu, addr), 0)
        del mf
        gc.collect()
        self.assertEqual (_nblk_cached(self.gpu, addr), -1)

    def test_fresh_objects_scan (self):
        # a scan written with a fresh mol/mf per point never calls DF.reset; when a new DF object
        # landed on a freed one's address it was handed the old geometry's integrals (|dE| ~ 1.3 Ha)
        for it in range(30):
            k = it % 2
            mf = _hf(GOH[k], use_gpu=self.gpu)
            with self.subTest (it=it):
                self.assertAlmostEqual (mf.kernel(), ref_oh[k], 8)
            del mf
            gc.collect()

    def test_untracked_dfobj_raises (self):
        # libgpu refuses to cache blocks of a DF object nobody tracks (nothing would evict them)
        from unittest import mock
        from mrh.my_pyscf.gpu import eri_cache
        gpu = libgpu.init() # own device: the failed get_jk leaves it mid-call
        try:
            mf = _hf(GOH[0], use_gpu=gpu)
            with mock.patch.object (eri_cache, 'track', lambda dfobj: None):
                with self.assertRaisesRegex (RuntimeError, 'not registered'):
                    mf.kernel()
        finally:
            libgpu.destroy_device(gpu)

if __name__ == "__main__":
    print("Tests for GPU DF J/K after a geometry change")
    unittest.main()
