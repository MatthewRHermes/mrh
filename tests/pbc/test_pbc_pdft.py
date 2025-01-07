import numpy
import sys
from pyscf.pbc import gto, scf, dft
from mrh.my_pyscf import mcpdft
import unittest

class KnownValues(unittest.TestCase):
    def test_h2(self):
        cell = gto.M(a = numpy.eye(3)*5,
        atom = '''
            H         -6.37665        2.20769        3.00000
            H         -5.81119        2.63374        3.00000
        ''',
        basis = '6-31g',
        verbose = 1, max_memory=10000)
        cell.output = '/dev/null'
        cell.build()

        mf = dft.RKS(cell).density_fit() # GDF
        mf.xc = 'pbe'
        mf.exxdiv = None
        emf = mf.kernel()
        
        mc = mcpdft.CASCI(mf,'tPBE', 1, 2)
        ecasci = mc.kernel(mf.mo_coeff)[0]

        self.assertAlmostEqual (emf, ecasci, 7)

if __name__ == "__main__":
    print("Full Tests for PBC-PDFT")
    unittest.main()
