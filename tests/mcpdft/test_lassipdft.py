import sys
from pyscf import gto, scf, tools, dft, lib
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_dmet import localintegrals, dmet, fragments
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf import lassi
import unittest

class KnownValues(unittest.TestCase):
    def test_ethene (self):
        mol = gto.M()
        mol.atom = '''C  -0.662958  0.000000  0.000000;
                            C   0.662958  0.000000  0.000000;
                            H  -1.256559  -0.924026  0.000000;
                            H  1.256559  -0.924026  0.000000;
                            H  -1.256559  0.924026  0.000000;
                            H  1.256559  0.924026  0.000000'''
        mol.basis='sto3g'
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.build()
        mf = scf.ROHF(mol).newton().run()
        
        las = LASSCF(mf, (2,),(2,))
        las = las.state_average([1, 0], spins=[[0,], [2, ]], smults=[[1, ], [3, ]], charges=[[0, ],[0, ]])
        mo0 = las.localize_init_guess(([0, 1],), las.sort_mo([8, 9]))
        las.kernel(mo0)
        lsi = lassi.LASSI(las)
        lsi.kernel()
        mc = mcpdft.LASSI(lsi, 'tPBE', (2, ), (2, ))
        mc.kernel()
        elassi = mc.e_mcscf[0]
        epdft = mc.e_tot[0]
        self.assertAlmostEqual (elassi , -77.1154672717181, 7) # Reference values of CASSCF and CAS-PDFT
        self.assertAlmostEqual (epdft , -77.49805221093968, 7)

if __name__ == "__main__":
    print("Full Tests for LASSI-PDFT")
    unittest.main()


