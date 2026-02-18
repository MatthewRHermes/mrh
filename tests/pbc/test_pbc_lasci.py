import numpy
import sys
import unittest
from pyscf import mcscf
from pyscf.pbc import gto, scf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCFNoSymm as LASSCF

class KnownValues(unittest.TestCase):
    def test_h2(self):
        cell = gto.M(a = numpy.eye(3)*5,
        atom = '''
            H         -6.37665        2.20769        3.00000
            H         -5.81119        2.63374        3.00000
        ''',
        basis = 'sto3g',
        verbose = 1, max_memory=10000)
        cell.output = '/dev/null'
        cell.build()

        mf = scf.RHF(cell).density_fit() # GDF
        mf.exxdiv = None
        emf = mf.kernel()
        
        mc = mcscf.CASCI(mf, 2, 2)
        ecasci = mc.kernel()[0]

        las = LASSCF(mf, (2,), (2,))
        mo0 = las.localize_init_guess((list(range(2)), ))
        elasci = las.lasci(mo0)[1]

        self.assertAlmostEqual (elasci, ecasci, 7)

if __name__ == "__main__":
    print("Full Tests for PBC-LASCI")
    unittest.main()
