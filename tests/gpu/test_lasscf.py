#miniapp to debug and test performance of ao2mo versions
'''
all versions do the following
   b_{u1p2}^P = b^P_{u1u2}m^{p2}_{u2}        ---- 1
   b_{p1p2}^P = b^P_{u1p2}m^{p1}_{u1}        ---- 2
   g^{p1p2}_{a1a2} = b^P_{p1p2}b^P_{a1a2}    ---- 3
   g^{p1a1}_{p2a2} = b^P_{p1a1}b^P_{p2a2}    ---- 4
   
  
   gpu_v3 does 1, 2 and 3 and pulls back g^{p1p2}_{a1a2} and b^P_{p1a1}
   gpu_v4 does 1, 2, 3 and 4 and pulls back g^{p1p2}_{a1a2} and g^{p1a1}_{p2a2} 

''' 
import unittest
from pyscf import gto, scf, tools, mcscf,lib
from pyscf.mcscf import avas
from mrh.tests.gpu.geometry_generator import generator
from pyscf.ao2mo import _ao2mo
import ctypes
import numpy 
from mrh.my_pyscf.gpu import libgpu

def setUpModule():
    global gpu, mf, mo_guess
    from gpu4mrh import patch_pyscf
    gpu = libgpu.init ()
    lib.param.use_gpu = gpu
    mol=gto.M(atom=generator(1),
              basis='6-31g',
              verbose=0,
              output='/dev/null')
    mf=scf.RHF(mol)
    mf=mf.density_fit()
    mf.run()
    ncas, nelecas, mo_guess = avas.kernel(mf, ['C 2pz'])
    
def tearDownModule():
    global gpu, mf, mo_guess
    libgpu.destroy_device (gpu)
    del gpu, mf, mo_guess
    lib.param.use_gpu = None

REFERENCE_E = -78.0335541872522

class KnownValues (unittest.TestCase):

    def test_sync (self):
        from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
        las = LASSCF(mf, (2,), (2,))
        mo_coeff = las.localize_init_guess ([[1,2],], mo_guess)
        las.kernel (mo_coeff)
        self.assertTrue (las.converged)
        self.assertAlmostEqual (las.e_tot, REFERENCE_E, 6)

    def test_async (self):
        from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
        las = LASSCF(mf, (2,), (2,))
        mo_coeff = las.set_fragments_([[1,2],], mo_guess)
        las.kernel (mo_coeff)
        self.assertTrue (las.converged)
        self.assertAlmostEqual (las.e_tot, REFERENCE_E, 6)

if __name__ == "__main__":
    print("Tests for GPU accelerated LASSCF kernels")
    unittest.main()
