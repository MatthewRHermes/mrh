import unittest
from pyscf import gto, scf

def setUpModule():
    pass

def tearDownModule():
    pass

class KnownValues (unittest.TestCase):

    def test_get_single_state_las (self):
        from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
        from mrh.my_pyscf.lassi import states as lassi_states
        xyz='''N 0 0 0,
               N 3 0 0'''
        mol = gto.M (atom=xyz, basis='cc-pvdz', symmetry=False, verbose=0, output='/dev/null')
        mf = scf.RHF (mol).run ()
        las = LASSCF (mf, (6,6), ((3,0),(0,3)))
        
        mo = las.set_fragments_(([0],[1]))
        las = lassi_states.spin_shuffle (las)
        las.weights = [1.0/las.nroots,]*las.nroots
        las.kernel (mo)
        for i in range (len (las.e_states)):
            las1 = las.get_single_state_las (state=i)
            las1.lasci ()
            self.assertAlmostEqual (las1.e_tot, las.e_states[i], 6)
            e_lasci = las1.e_tot
            las1.kernel ()
            self.assertLessEqual (las1.e_tot, las.e_states[i])

if __name__ == "__main__":
    print("Full Tests for LASSCF/LASCI miscellaneous")
    unittest.main()
