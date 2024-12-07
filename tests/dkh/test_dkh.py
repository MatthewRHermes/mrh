#
# Author: Bhavnesh Jangid <email: jangidbhavnesh@uchicago.edu>
#

import unittest
from pyscf import gto, scf
from mrh.my_pyscf.dkh import dkh

class KnownValues(unittest.TestCase):

    def test_DKH2(self):
        mol = gto.Mole(atom='''Ne 0 0 0''',basis='cc-pvdz-dk',verbose=0)
        mol.build()
        mfdkh = scf.RHF(mol)
        mfdkh.get_hcore = lambda *args: dkh(mol,dkhord=2,c=137.0359895) # Orca's Speed of Light
        mfdkh.kernel()
        self.assertAlmostEqual(mfdkh.e_tot, -128.62532864034782, 7)

    def test_DKH3(self):
        mol = gto.Mole(atom='''Ne 0 0 0''',basis='cc-pvdz-dk',verbose=0)
        mol.build()
        mfdkh = scf.RHF(mol)
        mfdkh.get_hcore = lambda *args: dkh(mol,dkhord=3,c=137.0359895) # Orca's Speed of Light
        mfdkh.kernel()
        self.assertAlmostEqual(mfdkh.e_tot, -128.62538501869074, 7)

    def test_DKH4(self):
        mol = gto.Mole(atom='''Ne 0 0 0''',basis='cc-pvdz-dk',verbose=0)
        mol.build()
        mfdkh = scf.RHF(mol)
        mfdkh.get_hcore = lambda *args: dkh(mol,dkhord=4,c=137.0359895) # Orca's Speed of Light
        mfdkh.kernel()
        self.assertAlmostEqual(mfdkh.e_tot, -128.62538062385389, 7)

if __name__ == "__main__":
    print("Test for DKH Scalar Relativisitic Effects")
    unittest.main()

