import sys
from pyscf import gto, scf, tools, dft, lib
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf import mcpdft
import unittest

class KnownValues(unittest.TestCase):
    def test_h4 (self):
        mol = gto.M()
        mol.atom='''H -5.10574 2.01997 0.00000;
                    H -4.29369 2.08633 0.00000;
                    H -3.10185 2.22603 0.00000;
                    H -2.29672 2.35095 0.00000'''
        mol.basis='sto3g'
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.build()

        mf = scf.ROHF(mol).newton().run()

        mc = mcpdft.LASSCF(mf, 'tPBE', (2, 2), (2, 2), spin_sub=(3,1), grids_level=1)
        mc = mc.state_average([0.5,0.5],spins=[[2,0],[0,2]],smults=[[3,1],[1,3]],charges=[[0,0],[0,0]])
        mo0 = mc.localize_init_guess (([0, 1] , [2, 3]))
        eref = mc.kernel(mo0)[0]

        las = LASSCF(mf,(2, 2), (2, 2), spin_sub=(3,1))
        las = las.state_average_([0.5,0.5],spins=[[2,0],[0,2]],smults=[[3,1],[1,3]],charges=[[0,0],[0,0]])
        mo0 = las.localize_init_guess (([0, 1] , [2, 3]))
        las.kernel(mo0)[0]

        mclas = mcpdft.LASSCF(las, 'tPBE', DoPreLASSI=True, grids_level=1)
        ecal = mclas.kernel()[0]

        self.assertAlmostEqual (eref[0], ecal[0], 8)
        self.assertAlmostEqual (eref[1], ecal[1], 8)

if __name__ == "__main__":
    print("Full Tests for PreLASSI, LAS-PDFT")
    unittest.main()


