import unittest
import pyscf
from pyscf import gto, scf, tools, mcscf,lib
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
from pyscf.mcscf import avas
from mrh.tests.gpu.geometry_generator import generator
from pyscf.fci import rdm, cistring
from pyscf.fci.addons import _unpack_nelec
import numpy as np


def setUpModule():
    global nfrags, basis
    nfrags = 4
    basis = '631g'
    
def tearDownModule():
    global nfrags, basis
    del nfrags, basis

def _run_mod(gpu_run, norb, nelec, link_index, cibra, ciket):
    if gpu_run: 
        from mrh.my_pyscf.gpu import libgpu
        from gpu4mrh import patch_pyscf
        gpu = libgpu.init()
        mol = gto.M(use_gpu = gpu, atom=generator(nfrags), basis = basis)
        from pyscf.lib import param
        param.use_gpu = gpu
    else:
        mol = gto.M(atom=generator(nfrags), basis = basis)
    mf = scf.RHF(mol)
    mf=mf.density_fit()
    mf.with_df.auxbasis = pyscf.df.make_auxbasis(mol)
    mf.max_cycle=1
    mf.kernel()
    
    rdm1a = rdm.make_rdm1_spin1('FCItrans_rdm1a', cibra, ciket, norb, nelec, link_index)
    rdm1b = rdm.make_rdm1_spin1('FCItrans_rdm1b', cibra, ciket, norb, nelec, link_index)
    rdm1aa, rdm2aa = rdm.make_rdm12_spin1('FCItdm12kern_a', cibra, ciket, norb, nelec, link_index)
    rdm1bb, rdm2bb = rdm.make_rdm12_spin1('FCItdm12kern_b', cibra, ciket, norb, nelec, link_index)
    _, rdm2ab = rdm.make_rdm12_spin1('FCItdm12kern_ab', cibra, ciket, norb, nelec, link_index) 
    rdm1sf, rdm2sf = rdm.make_rdm12_spin1('FCItdm12kern_sf', cibra, ciket, norb, nelec, link_index)
    return rdm1a, rdm1b, rdm1aa, rdm2aa, rdm1bb, rdm2bb, rdm2ab, rdm1sf, rdm2sf



class KnownValues(unittest.TestCase):
    def test_implementations(self):

        norb = 9
        nelec = 5
        neleca, nelecb = _unpack_nelec(nelec)
        link_indexa = link_indexb = cistring.gen_linkstr_index(range(norb), neleca)
        if neleca!=nelecb: link_indexb= cistring.gen_linkstr_index(range(norb), nelecb)
        na = link_indexa.shape[0]
        nb = link_indexb.shape[0]
        cibra =np.random.random((na,nb))
        ciket =np.random.random((na,nb))
        link_index = (link_indexa, link_indexb)

        rdm1a_cpu, rdm1b_cpu, rdm1aa_cpu, rdm2aa_cpu, rdm1bb_cpu, rdm2bb_cpu, rdm2ab_cpu, rdm1sf_cpu, rdm2sf_cpu = _run_mod(False, norb, nelec, link_index, cibra, ciket)
        rdm1a_gpu, rdm1b_gpu, rdm1aa_gpu, rdm2aa_gpu, rdm1bb_gpu, rdm2bb_gpu, rdm2ab_gpu, rdm1sf_gpu, rdm2sf_gpu = _run_mod(True, norb, nelec, link_index, cibra, ciket)
        
        rdm1a_diff = np.max(np.abs(rdm1a_cpu-rdm1a_gpu))
        rdm1b_diff = np.max(np.abs(rdm1b_cpu-rdm1b_gpu))
        rdm1aa_diff = np.max(np.abs(rdm1aa_cpu-rdm1aa_gpu))
        rdm2aa_diff = np.max(np.abs(rdm2aa_cpu-rdm2aa_gpu))
        rdm1bb_diff = np.max(np.abs(rdm1bb_cpu-rdm1bb_gpu))
        rdm2bb_diff = np.max(np.abs(rdm2bb_cpu-rdm2bb_gpu))
        rdm2ab_diff = np.max(np.abs(rdm2ab_cpu-rdm2ab_gpu))
        rdm1sf_diff = np.max(np.abs(rdm1sf_cpu-rdm1sf_gpu))
        rdm2sf_diff = np.max(np.abs(rdm2sf_cpu-rdm2sf_gpu))

        with self.subTest ('FCI trans_rdm1a'):
            self.assertAlmostEqual(rdm1a_diff, 0.0, 7)
        with self.subTest ('FCI trans_rdm1b'):
            self.assertAlmostEqual(rdm1b_diff, 0.0, 7)
        with self.subTest ('FCI tdm12kern_a'):
            self.assertAlmostEqual(rdm1aa_diff, 0.0, 7)
            self.assertAlmostEqual(rdm2aa_diff, 0.0, 7)
        with self.subTest ('FCI tdm12kern_b'):
            self.assertAlmostEqual(rdm1bb_diff, 0.0, 7)
            self.assertAlmostEqual(rdm2bb_diff, 0.0, 7)
        with self.subTest ('FCI tdm12kern_ab'):
            self.assertAlmostEqual(rdm2ab_diff, 0.0, 7)
        with self.subTest ('FCI tdm12kern_ab'):
            self.assertAlmostEqual(rdm1sf_diff, 0.0, 7)
            self.assertAlmostEqual(rdm2sf_diff, 0.0, 7)

if __name__ == "__main__":
    print("Tests for GPU accelerated TDMs")
    unittest.main()
