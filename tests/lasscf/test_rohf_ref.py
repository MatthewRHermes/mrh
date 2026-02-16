import unittest
from pyscf import gto, scf, mcscf, lib
from mrh.my_pyscf.mcscf import lasscf_async as asyn
from mrh.my_pyscf.mcscf import lasscf_sync_o0 as syn

class KnownValues (unittest.TestCase):

    def test_c2h6n4 (self):
        from mrh.tests.lasscf.c2h6n4_struct import structure as struct
        mol = struct (0.2, 0.2, '6-31g', symmetry=False)
        mol.spin = 4
        mol.output = '/dev/null'
        mol.verbose = 0
        mol.build ()
        mf = scf.RHF (mol).run (max_cycle=100)
        
        mc = mcscf.CASCI (mf, 4, 4).run ()
        self.assertAlmostEqual (mf.e_tot, mc.e_tot, 6)

        las = syn.LASSCF (mf, (4, 4), ((4,2), (4,2)), spin_sub=(3,3))
        las.mo_coeff = las.localize_init_guess ([[1,2],[9,10]], mf.mo_coeff, freeze_cas_spaces=True)
        las.ci = las.get_init_guess_ci ()
        e_tot = las.energy_elec () + las.energy_nuc ()

        with self.assertRaises (AssertionError):
            self.assertAlmostEqual (e_tot, mf.e_tot, 6)
        
        las = syn.LASSCF (mf, (4, 4), ((4,2), (4,2)), spin_sub=(3,3))
        las.mo_coeff = las.localize_init_guess ([[1,2],[9,10]], mf.mo_coeff, mo_occ=mf.mo_occ)
        las.ci = las.get_init_guess_ci ()
        e_tot = las.energy_elec () + las.energy_nuc ()

        self.assertAlmostEqual (e_tot, mf.e_tot, 6)

    def test_c2h4n4 (self):
        from mrh.tests.lasscf.c2h4n4_struct import structure as struct
        mol = struct (0.2, 0.2, '6-31g', symmetry='Cs')
        mol.spin = 4
        mol.output = '/dev/null'
        mol.verbose = 0
        mol.build ()
        mf = scf.RHF (mol).run (max_cycle=100)
        
        mc = mcscf.CASCI (mf, 4, 4).run ()
        self.assertAlmostEqual (mf.e_tot, mc.e_tot, 6)

        las = syn.LASSCF (mf, (4, 4), ((4,2), (4,2)), spin_sub=(3,3))
        las.mo_coeff = las.localize_init_guess ([[1,2],[7,8]], mf.mo_coeff, freeze_cas_spaces=True)
        las.lasci_() # need to initialize some FCISolver attributes
        las.ci = las.get_init_guess_ci ()
        e_tot = las.energy_elec () + las.energy_nuc ()

        with self.assertRaises (AssertionError):
            self.assertAlmostEqual (e_tot, mf.e_tot, 6)
        
        las = syn.LASSCF (mf, (4, 4), ((4,2), (4,2)), spin_sub=(3,3))
        las.mo_coeff = las.localize_init_guess ([[1,2],[7,8]], mf.mo_coeff, mo_occ=mf.mo_occ)
        las.lasci_() # need to initialize some FCISolver attributes
        las.ci = las.get_init_guess_ci ()
        e_tot = las.energy_elec () + las.energy_nuc ()

        self.assertAlmostEqual (e_tot, mf.e_tot, 6)

if __name__ == "__main__":
    print ("Full Tests for LASSCF orbital localization from ROHF references")
    unittest.main()
