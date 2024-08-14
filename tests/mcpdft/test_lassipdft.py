import unittest
from pyscf import lib, gto, scf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import LASSI
from mrh.my_pyscf.lassi.spaces import all_single_excitations
from mrh.my_pyscf.mcscf.lasci import get_space_info

class KnownValues(unittest.TestCase):
    def test_ethene_casci_limit (self):
        xyz='''H 0 0 0
               H 1 0 0
               H 3 0 0
               H 4 0 0'''

        mol = gto.M (atom=xyz, basis='sto3g', symmetry=False, verbose=0, output='/dev/null')
        mf = scf.RHF (mol).run ()
        
        # LASSCF and LASSI
        las = LASSCF (mf, (2,2), (2,2), spin_sub=(1,1))
        las.lasci ()
        las1 = las
        for i in range (2): las1 = all_single_excitations (las1)
        charges, spins, smults, wfnsyms = get_space_info (las1)
        lroots = 4 - smults
        idx = (charges!=0) & (lroots==3)
        lroots[idx] = 1
        las1.conv_tol_grad = las.conv_tol_self = 9e99
        las1.lasci (lroots=lroots.T)
        las1.dump_spaces ()
        # CASCI limit
        from pyscf import mcpdft
        mc = mcpdft.CASCI (mf, 'tPBE', 4, 4).run ()
        lsi = LASSI (las1)
        lsi.kernel ()
        self.assertAlmostEqual (lsi.e_roots[0], mc.e_mcscf, 7)
        for opt in range (3):
            with self.subTest (opt=opt):
                from mrh.my_pyscf import mcpdft
                lsipdft = mcpdft.LASSI (lsi, 'tPBE')
                lsipdft.opt = opt
                lsipdft.kernel()
                self.assertAlmostEqual (lsipdft.e_tot[0], mc.e_tot, 7)

if __name__ == "__main__":
    print("Full Tests for LASSI-PDFT")
    unittest.main()


